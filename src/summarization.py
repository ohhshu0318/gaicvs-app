"""要約タスク実行モジュール。
Summarization task execution module.

このモジュールは、生命保険会社の苦情要望データの要約処理を行います。
This module performs summarization processing of life insurance company complaint data.

処理の流れ (Processing flow):
1. 入力CSVの取得と検証 (Get and validate input CSV)
2. テキストデータのクリーニング (Clean text data)
3. テキストの結合とプロンプト生成 (Combine text and generate prompts)
4. Florence APIによる要約生成 (Generate summaries using Florence API)
5. 結果のS3保存とDB書き込み (Save results to S3 and write to DB)
"""

import argparse
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
from sqlalchemy import MetaData, Table
from sqlalchemy.exc import OperationalError

from classes.db_operation import db_operator
from classes.error_handle import CustomError, error_context, error_handler, safe_execute
from classes.florence_operation import florence_operator
from classes.logging import s3_logger
from classes.prompt_operation import s3_prompt
from classes.storage_operation import storage_operator
from common.const import (
    S3_ENDFILE_PATH,
    S3_INPUT_PATH,
    S3_LOG_PATH,
    S3_SUMMARIZATION_RESULTS_PATH,
    SUMMARIZATION_PROMPT_PATH,
    SUMMARIZATION_RESULST_CSV_HEADER,
)
from common.utils import from_utc_to_jst
from utils.data_utils import DataUtils
from utils.prompt_utils import PromptUtils
from utils.s3_utils import S3Utils


def init_args() -> argparse.Namespace:
    """コマンドライン引数を初期化する。
    Initialize command line arguments.

    Returns:
        解析されたコマンドライン引数 / Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="要約タスクを実行する")

    parser.add_argument(
        "--run_date",
        type=str,
        dest="run_date",
        required=True,
        help="タスク実行日 / Task execution date",
    )
    parser.add_argument(
        "--execution_id",
        type=str,
        dest="execution_id",
        required=True,
        help="Step Function execution ID",
    )
    
    return parser.parse_args()


@error_handler(
    "logs/summarization.log", "ERR", "02", "1", "001", "001"
)
def read_csv_from_s3(input_s3_key: str) -> pd.DataFrame:
    """S3からCSVファイルを読み込む。
    Read CSV file from S3.

    Args:
        input_s3_key: 入力ファイルのS3キー / S3 key of input file

    Returns:
        読み込んだDataFrame / Loaded DataFrame
    """
    s3_logger.info(f"{input_s3_key} からCSVデータを取得します")
    s3_logger.info(f"Getting CSV data from {input_s3_key}")
    
    csv_data = storage_operator.get_object(input_s3_key)
    df = pd.read_csv(csv_data)
    
    if df.empty:
        s3_logger.error("入力CSVファイルにデータがありません")
        s3_logger.error("Input CSV file contains no data")
        raise ValueError("Input CSV file is empty")
    
    return df


@error_handler(
    "logs/summarization.log", "ERR", "02", "1", "004", "003"
)
def generate_summaries(df: pd.DataFrame, prompt_template: str) -> pd.DataFrame:
    """テキストの要約を生成する。
    Generate summaries of texts.

    Args:
        df: 入力データフレーム / Input DataFrame
        prompt_template: プロンプトテンプレート / Prompt template

    Returns:
        要約結果を含むDataFrame / DataFrame with summary results
    """
    # テキストカラムの結合 / Combine text columns
    df["combined_text"] = DataUtils.combine_text_columns(
        df, ["mosd_naiyou", "taioushousai", "kujo_partybiko"]
    )

    s3_logger.info("行の結合処理が正常に完了しました")
    s3_logger.info("Row combination processing completed successfully")

    # プロンプト生成 / Generate prompts
    df = PromptUtils.generate_summarization_prompt(
        df, prompt_template, "combined_text"
    )

    input_prompts = df["genai_summary_prompt"].tolist()
    s3_logger.info(f"{len(input_prompts)} 件のプロンプトを生成しました")
    s3_logger.info(f"Generated {len(input_prompts)} prompts")
    
    # Florence APIにリクエスト / Request to Florence API
    s3_logger.info("Florence APIに要約リクエストを送信します")
    s3_logger.info("Sending summarization request to Florence API")
    
    summary_result = florence_operator.chat(input_prompts)
    
    s3_logger.info("APIから要約を取得しました")
    s3_logger.info("Retrieved summaries from API")

    # レスポンス処理 / Process responses
    results = PromptUtils.extract_api_responses(
        summary_result,
        error_code_success="",
        error_code_timeout="WRN_02_1_004_006 要約処理システムエラーFlorenceAPI Timeout",
        error_code_failed="WRN_02_1_004_009 要約処理業務エラーFlorence要約できない"
    )
    
    df["genai_summary_text"] = [result["data"] for result in results]
    df["error_message"] = [result["error_message"] for result in results]

    # タイムスタンプ追加 / Add timestamp
    current_utc_time = datetime.now(timezone.utc).isoformat()
    jst_time = from_utc_to_jst(current_utc_time)
    df["genai_summary_run4md"] = jst_time.strftime("%Y-%m-%d %H:%M:%S")

    s3_logger.info("要約生成が完了しました")
    s3_logger.info("Summary generation completed")
    
    return df


@error_handler(
    "logs/summarization.log", "ERR", "02", "1", "001", "002"
)
def save_output_to_s3(df: pd.DataFrame, output_key: str) -> None:
    """生成された要約結果をS3に保存する。
    Save generated summary results to S3.

    Args:
        df: 要約結果を含むDataFrame / DataFrame with summary results
        output_key: 出力ファイルのS3キー / S3 key of output file
    """
    s3_logger.info(f"結果を {output_key} に保存します")
    s3_logger.info(f"Saving results to {output_key}")
    
    output_df = df[SUMMARIZATION_RESULST_CSV_HEADER]
    storage_operator.put_df_to_s3(output_df, output_key)
    
    s3_logger.info(f"結果を {output_key} に正常に保存しました")
    s3_logger.info(f"Successfully saved results to {output_key}")


@error_handler(
    "logs/summarization.log", "ERR", "02", "1", "002", "002"
)
def write_results_to_db(
    df: pd.DataFrame, 
    table_name: str, 
    schema: str, 
    index_elements: List[str]
) -> None:
    """処理結果をDBに書き込む。
    Write processing results to DB.

    Args:
        df: 要約結果を含むDataFrame / DataFrame with summary results
        table_name: DB表名 / DB table name
        schema: DBスキーマ名 / DB schema name
        index_elements: キー列のリスト / List of key columns
    """
    s3_logger.info(f"処理結果を{schema}.{table_name}に書き込みます")
    s3_logger.info(f"Writing processing results to {schema}.{table_name}")
    
    records = df.to_dict(orient="records")
    metadata = MetaData()
    table = Table(
        table_name,
        metadata,
        schema=schema,
        autoload_with=db_operator.engine,
    )
    
    current_utc_time = datetime.now(timezone.utc).isoformat()
    jst_time = from_utc_to_jst(current_utc_time)
    current_timestamp = jst_time.strftime("%Y-%m-%d %H:%M:%S")

    for record in records:
        set_values = {
            "genai_summary_text": record["genai_summary_text"],
            "finalupd_tstnp": current_timestamp,
        }

        insert_values = {
            "kjoinfo_tsuban": record["kjoinfo_tsuban"],
            "tourokutstnp": current_timestamp,
            "trksha_shzkc": "gen-ai",
            "trksha_shzkmei": "gen-ai",
            "tourokushacode": "gen-ai",
            "tourokushamei": "gen-ai",
            "finalupopshzkc": "gen-ai",
            "finluposhzkmei": "gen-ai",
            "final_upd_op_c": "gen-ai",
            "saishuksnsmei": "gen-ai",
            "sakujo_flg": "0",
            **set_values,
        }

        db_operator.do_upsert(
            table_object=table,
            records=[insert_values],
            index_elements=index_elements,
            update_columns=list(set_values.keys()),
        )

    s3_logger.info("処理結果をDBに正常に書き込みました")
    s3_logger.info("Successfully wrote processing results to DB")


def main() -> None:
    """メイン関数。
    Main function.
    """
    # 初期化 / Initialization
    args = init_args()
    execution_id = args.execution_id
    run_date = from_utc_to_jst(args.run_date).strftime("%Y%m%d")
    
    # ログパスの生成 / Generate log path
    log_path = (
        S3_LOG_PATH.replace("@run_date", run_date)
        .replace("@execution_id", execution_id)
        .replace("@type", "summarization")
    )
    
    # S3パスの生成 / Generate S3 paths
    input_s3_key = (
        S3_INPUT_PATH.replace("@run_date", run_date)
        .replace("@execution_id", execution_id)
        .replace("@type", "summarization")
    )
    
    result_s3_key = S3_SUMMARIZATION_RESULTS_PATH.replace(
        "@run_date", run_date
    ).replace("@execution_id", execution_id)
    
    # 共通ユーティリティのインスタンス化 / Instantiate common utilities
    s3_utils = S3Utils(storage_operator)
    
    # S3から入力CSVファイルを取得 / Get input CSV file from S3
    input_data = read_csv_from_s3(input_s3_key)
    
    # データクリーニング / Data cleaning
    input_data, missing_data_rows = DataUtils.clean_text_data(input_data, ["mosd_naiyou"])
    if not missing_data_rows.empty:
        s3_logger.info(f"{len(missing_data_rows)} 行のデータに欠損値があります")
        s3_logger.info(f"{len(missing_data_rows)} rows have missing values")
    
    # S3から要約プロンプトを取得 / Get summarization prompt from S3
    s3_logger.info(f"要約プロンプトを取得: {SUMMARIZATION_PROMPT_PATH}")
    s3_logger.info(f"Retrieving summarization prompt: {SUMMARIZATION_PROMPT_PATH}")
    
    summarization_prompt = s3_prompt.get_prompt(SUMMARIZATION_PROMPT_PATH)
    
    # 要約生成 / Generate summaries
    summary_result_df = generate_summaries(input_data, summarization_prompt)
    
    # S3に結果を保存 / Save results to S3
    save_output_to_s3(summary_result_df, result_s3_key)
    
    # DBに結果を書き込み / Write results to DB
    write_results_to_db(
        summary_result_df,
        table_name="tblgensummary",
        schema="kujo",
        index_elements=["kjoinfo_tsuban"],
    )
    
    # アウトプットファイル存在チェック / Check output file existence
    s3_logger.info("アウトプットCSVファイル存在チェックを行う")
    s3_logger.info("Checking output CSV file existence")
    
    s3_utils.check_file_exists(
        key=result_s3_key,
        log_path=log_path,
        error_type="ERR",
        error_main="02",
        error_category="1",
        error_module="001",
        error_detail="001"
    )
    
    # アウトプットファイルヘッダーチェック / Check output file headers
    s3_logger.info("アウトプットCSVファイルのヘッダーチェックを行う")
    s3_logger.info("Checking output CSV file headers")
    
    s3_utils.check_file_headers(
        key=result_s3_key,
        expected_headers=SUMMARIZATION_RESULST_CSV_HEADER,
        log_path=log_path,
        error_type="ERR",
        error_main="02",
        error_category="1",
        error_module="001",
        error_detail="008"
    )
    
    # 前回実行日時の取得 / Get last run date
    endfile_s3_key_prefix = S3_ENDFILE_PATH.replace("@type", "summarization")
    
    last_run_date, last_run_endfile = s3_utils.get_last_run_date(
        endfile_prefix=endfile_s3_key_prefix,
        log_path=log_path,
        error_type="ERR",
        error_main="02",
        error_category="1",
        error_module="001",
        error_detail="001"
    )
    
    # endfileの管理（バックアップと新規作成）/ Manage endfile (backup and create new)
    s3_utils.manage_endfile(
        endfile_prefix=endfile_s3_key_prefix,
        last_run_endfile=last_run_endfile,
        run_date=args.run_date,
        log_path=log_path,
        error_type="ERR",
        error_main="02",
        error_category="1",
        error_module="001",
        error_detail="002"
    )
    
    s3_logger.info("要約処理が完了しました")
    s3_logger.info("Summarization processing completed")


if __name__ == "__main__":
    main()

import argparse
import asyncio
from datetime import datetime, timezone
from io import StringIO

import pandas as pd
from classes.db_operation import db_operator
from classes.error_handler import error_handler
from classes.florence_operation import florence_operator
from classes.logging_handler import application_logger
from classes.prompt_operation import s3_prompt
from classes.storage_operation import storage_operator
from common.const import (S3_ENDFILE_PATH, S3_INPUT_PATH, S3_LOG_PATH,
                          S3_SUMMARIZATION_RESULTS_PATH,
                          SUMMARIZATION_PROMPT_PATH,
                          SUMMARIZATION_RESULTS_CSV_HEADER)
from sqlalchemy import MetaData, Table
from utils.data_utils import DataUtils
from utils.jst_utils import from_utc_to_jst
from utils.prompt_utils import PromptUtils
from utils.s3_utils import S3Utils


def init_args():
    parser = argparse.ArgumentParser(description="要約タスクを実行する")

    parser.add_argument(
        "--run_date",
        type=str,
        dest="run_date",
        required=True,
        help="タスク実行日",
    )
    parser.add_argument(
        "--execution_id",
        type=str,
        dest="execution_id",
        required=True,
        help="Step Function execution ID",
    )
    args = parser.parse_args()

    return args


def generate_s3_path(
    base_path: str, run_date: str, execution_id: str, type: str = ""
) -> str:
    """
    生成S3パス。typeが指定されていない場合、無視する。
    """
    if type:
        return (
            base_path.replace("@run_date", run_date)
            .replace("@execution_id", execution_id)
            .replace("@type", type)
        )
    else:
        return base_path.replace("@run_date", run_date).replace(
            "@execution_id", execution_id
        )


@error_handler("summarization", "read_csv")
def read_csv_from_s3(
    run_date: str, execution_id: str, type: str, log_path: str
):
    """
    S3からCSVデータを読み込む
    """
    input_file_name = generate_s3_path(
        S3_INPUT_PATH, run_date, execution_id, type
    )

    csv_data = storage_operator.get_object(input_file_name)
    df = pd.read_csv(StringIO(csv_data))

    if df.empty:
        application_logger.error(f"CSVデータが空です: {input_file_name}")
        raise ValueError("CSVデータが空です")

    application_logger.info(f"{input_file_name} からCSVデータを取得しました")

    # DataUtilsを使用してデータをクリーニング
    df_cleaned, missing_rows = DataUtils.clean_text_data(
        df, columns=["mosd_naiyou", "taioushousai"]
    )

    if not missing_rows.empty:
        application_logger.info(
            f"欠損値や空の値を含む行が {len(missing_rows)}件 削除されました"
        )

    return df_cleaned


@error_handler("summarization", "generate_summaries")
def summarization(df, summarization_prompt, log_path):
    """
    特定の列を結合し、Promptを生成してAPIで要約を取得する
    """
    # テキスト列を結合
    df["combined_text"] = DataUtils.combine_text_columns(
        df, ["mosd_naiyou", "taioushousai"]
    )

    application_logger.info("行の結合処理が正常に完了しました")

    # PromptUtilsを使用してプロンプトを生成
    df = PromptUtils.create_summary_prompt(
        df, summarization_prompt, "combined_text"
    )

    # 不要な列を削除
    df.drop(
        columns=["mosd_naiyou", "taioushousai", "combined_text"], inplace=True
    )

    input_prompts = df["genai_summary_prompt"].tolist()
    application_logger.info(f"生成したプロンプト: {input_prompts}")

    # APIで要約を取得
    summary_result = asyncio.run(florence_operator.chat(input_prompts))
    application_logger.info(f"要約結果: {summary_result}")
    application_logger.info("APIから要約を取得しました")

    # PromptUtilsを使用してAPIレスポンスを処理
    processed_results = PromptUtils.extract_api_responses(
        summary_result,
        error_code_success="",
        error_code_timeout="WRN_02_1_004_006 要約処理システムエラーFlorenceAPI Timeout",
        error_code_failed="WRN_02_2_004_009 要約処理業務エラーFlorence要約できない",
    )

    # 結果をDataFrameに追加
    df["genai_summary_text"] = [result["data"] for result in processed_results]
    df["error_message"] = [
        result["error_message"] for result in processed_results
    ]

    # 空の要約をカウント
    nan_count = df["genai_summary_text"].isna().sum()
    blank_count = (df["genai_summary_text"] == "空白").sum()

    if nan_count > 0:
        application_logger.info(
            f"要約結果内のNaNは{nan_count}件検出されました"
        )
    if blank_count > 0:
        application_logger.info(
            f"要約結果内の「空白」文字は{blank_count}件検出されました"
        )

    # 空の要約を処理
    df["genai_summary_text"] = df["genai_summary_text"].apply(
        lambda x: "" if pd.isna(x) or x.strip() == "" or x == "空白" else x
    )

    # タイムスタンプを追加
    current_utc_time = datetime.now(timezone.utc).isoformat()
    jst_time = from_utc_to_jst(current_utc_time)
    df["genai_summary_run4md"] = jst_time.strftime("%Y-%m-%d %H:%M:%S")

    application_logger.info("要約生成が完了しました")
    return df


@error_handler("summarization", "save_output")
def save_output_to_s3(df, run_date, execution_id, log_path):
    """
    生成された要約結果をS3に保存する
    """
    output_key = generate_s3_path(
        S3_SUMMARIZATION_RESULTS_PATH, run_date, execution_id
    )

    output_df = df[SUMMARIZATION_RESULTS_CSV_HEADER]
    storage_operator.put_df_to_s3(output_df, output_key)

    application_logger.info(f"結果を {output_key} に正常に保存しました")
    return output_key


@error_handler("summarization", "write_db")
def write_results_to_db(log_path, df, table_name, schema, index_elements):
    """
    処理結果をDBに書き込む
    """
    records = df.to_dict(orient="records")
    metadata = MetaData()
    table = Table(
        table_name, metadata, schema=schema, autoload_with=db_operator.engine
    )
    current_utc_time = datetime.now(timezone.utc).isoformat()
    jst_time = from_utc_to_jst(current_utc_time)

    for record in records:
        set_values = {
            "genai_summary_text": record["genai_summary_text"],
            "finalupd_tstnp": jst_time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        insert_values = {
            "kjoinfo_tsuban": record["kjoinfo_tsuban"],
            "tourokutstnp": jst_time.strftime("%Y-%m-%d %H:%M:%S"),
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

    application_logger.info("処理結果をDBに正常に書き込みました")


def main():
    """
    ECSタスクのメイン関数。要約生成のプロセスを処理する
    """
    args = init_args()
    execution_id = args.execution_id
    run_date = from_utc_to_jst(args.run_date).strftime("%Y%m%d")
    log_path = (
        S3_LOG_PATH.replace("@run_date", run_date)
        .replace("@execution_id", execution_id)
        .replace("@type", "summarization")
    )

    # アプリケーションコンテキストを設定
    application_logger.set_execution_context("summarization", execution_id)

    # S3ユーティリティの初期化
    s3_utils = S3Utils(storage_operator)

    # CSVデータの読み込み
    input_data = read_csv_from_s3(
        run_date, execution_id, "summarization", log_path
    )

    # プロンプトの取得
    summarization_prompt = s3_prompt.get_prompt(SUMMARIZATION_PROMPT_PATH)

    # 要約処理
    summary_result_df = summarization(
        input_data, summarization_prompt, log_path
    )

    # 結果をS3に保存
    result_s3_key = save_output_to_s3(
        summary_result_df, run_date, execution_id, log_path
    )

    # 結果をDBに書き込み
    write_results_to_db(
        log_path,
        summary_result_df,
        table_name="tblgensummary",
        schema="kujo",
        index_elements=["kjoinfo_tsuban"],
    )

    # ファイル存在チェック
    application_logger.info("出力CSVファイルの存在チェックを実施します")
    s3_utils.check_file_exists(
        key=result_s3_key,
        execution_id=execution_id,
        module_type="summarization",
        operation="check_file",
    )

    # ヘッダーチェック
    application_logger.info("出力CSVファイルのヘッダーチェックを実施します")
    s3_utils.check_file_headers(
        key=result_s3_key,
        expected_headers=SUMMARIZATION_RESULTS_CSV_HEADER,
        execution_id=execution_id,
        module_type="summarization",
        operation="check_header",
    )

    # 前回のendfileを取得
    application_logger.info("前回のendfileを取得します")
    endfile_s3_key_prefix = S3_ENDFILE_PATH.replace("@type", "summarization")
    last_run_date, last_run_endfile = s3_utils.get_last_run_date(
        endfile_prefix=endfile_s3_key_prefix,
        execution_id=execution_id,
        module_type="summarization",
        operation="get_last_run",
    )

    application_logger.info(f"前回のendfileを取得しました: {last_run_endfile}")

    # endfileの管理（バックアップと新規作成）
    application_logger.info("endfileを更新します")
    s3_utils.manage_endfile(
        endfile_prefix=endfile_s3_key_prefix,
        last_run_endfile=last_run_endfile,
        run_date=args.run_date,
        execution_id=execution_id,
        module_type="summarization",
        operation="s3_upload",
    )

    application_logger.info("要約処理が正常に完了しました")


if __name__ == "__main__":
    main()

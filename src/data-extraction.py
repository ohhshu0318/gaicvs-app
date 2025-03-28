"""データ抽出タスク実行モジュール。
Data extraction task execution module.

このモジュールは、生命保険会社の苦情要望データの抽出処理を行います。
This module performs data extraction processing of life insurance company complaint data.

処理の流れ (Processing flow):
1. 前回実行日時の取得 (Get the date of the last run)
2. DBからデータ抽出 (Extract data from DB)
3. S3への結果保存 (Save results to S3)
"""

import argparse
from typing import Optional

import pandas as pd
from sqlalchemy.exc import OperationalError

from classes.db_operation import db_operator
from classes.error_handle import CustomError, error_context, error_handler, safe_execute
from classes.logging import s3_logger
from classes.storage_operation import storage_operator
from common.const import (
    RESULTS_BUCKET_NAME,
    S3_ENDFILE_PATH,
    S3_INPUT_PATH,
    S3_LOG_PATH,
)
from common.SQL_template import (
    query_classification_template,
    query_summarization_template,
)
from common.utils import from_utc_to_jst
from utils.s3_utils import S3Utils


def init_args() -> argparse.Namespace:
    """コマンドライン引数を初期化する。
    Initialize command line arguments.

    Returns:
        解析されたコマンドライン引数 / Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="データ抽出タスクを実行する")

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
    parser.add_argument(
        "--type",
        type=str,
        choices=["summarization", "classification"],
        required=True,
        help="タスクのタイプ: summarization または classification / Task type: summarization or classification",
    )
    return parser.parse_args()


@error_handler(
    "logs/data_extraction.log", "ERR", "01", "2", "002", "007"
)
def get_data(
    last_run_date: str, 
    current_run_date: str, 
    task_type: str,
    log_path: str
) -> pd.DataFrame:
    """昨日の日付を取得し、DBからデータを抽出する。
    Get yesterday's date and extract data from DB.

    Args:
        last_run_date: 前回実行日時 / Last run date
        current_run_date: 今回実行日時 / Current run date
        task_type: タスクのタイプ / Task type
        log_path: ログファイルのパス / Log file path

    Returns:
        抽出されたデータフレーム / Extracted DataFrame
    """
    s3_logger.debug(f"取得した昨日の日付: {current_run_date}, タイプ: {task_type}")
    s3_logger.debug(f"Retrieved yesterday's date: {current_run_date}, type: {task_type}")
    
    # SQLテンプレートの日付を設定 / Set dates in SQL template
    query_summarization = query_summarization_template.replace(
        "@last_run_date", last_run_date
    ).replace("@run_date", current_run_date)
    
    query_classification = query_classification_template.replace(
        "@last_run_date", last_run_date
    ).replace("@run_date", current_run_date)

    # DBクエリを実行 / Execute DB query
    try:
        if task_type == "summarization":
            s3_logger.debug(f"実行するクエリ (summarization):{query_summarization}")
            s3_logger.debug(f"Executing query (summarization):{query_summarization}")
            result = db_operator.execute_statement("sql", query_summarization)
        else:  # classification
            s3_logger.debug(f"実行するクエリ (classification):{query_classification}")
            s3_logger.debug(f"Executing query (classification):{query_classification}")
            result = db_operator.execute_statement("sql", query_classification)
            
        if result.rowcount == 0:
            s3_logger.error(f"期間内のデータが見つかりません: {last_run_date} -> {current_run_date}")
            s3_logger.error(f"No data found in period: {last_run_date} -> {current_run_date}")
            raise ValueError("No data found in specified date range")
            
    except CustomError:
        raise
    except OperationalError as e:
        s3_logger.error(f"DB操作エラー: {str(e)}")
        s3_logger.error(f"DB operation error: {str(e)}")
        raise
    except Exception as e:
        s3_logger.error(f"データ取得エラー: {str(e)}")
        s3_logger.error(f"Data retrieval error: {str(e)}")
        raise

    # 結果をDataFrameに変換 / Convert results to DataFrame
    df = pd.DataFrame.from_records(result.fetchall(), columns=result.keys())
    s3_logger.info(f"DBから{len(df)}件のデータを取得しました")
    s3_logger.info(f"Retrieved {len(df)} records from DB")

    return df


@error_handler(
    "logs/data_extraction.log", "ERR", "01", "1", "001", "002"
)
def upload_to_s3(
    df: pd.DataFrame,
    s3_key: str,
    log_path: str
) -> None:
    """DataFrameをS3にアップロードする。
    Upload DataFrame to S3.

    Args:
        df: アップロードするDataFrame / DataFrame to upload
        s3_key: S3のキー / S3 key
        log_path: ログファイルのパス / Log file path
    """
    s3_logger.info(f"データをS3にアップロード: {RESULTS_BUCKET_NAME}/{s3_key}")
    s3_logger.info(f"Uploading data to S3: {RESULTS_BUCKET_NAME}/{s3_key}")
    
    # S3Utilsのインスタンス化 / Instantiate S3Utils
    s3_utils = S3Utils(storage_operator)
    
    # S3にDataFrameをアップロード / Upload DataFrame to S3
    storage_operator.put_df_to_s3(df=df, key=s3_key)
    
    # アップロードしたファイルの検証 / Validate uploaded file
    s3_utils.check_file_exists(
        key=s3_key,
        log_path=log_path,
        error_type="ERR",
        error_main="01",
        error_category="1",
        error_module="001",
        error_detail="002"
    )
    
    # ヘッダーの検証 / Validate headers
    s3_logger.info("アップロードしたCSVファイルのヘッダーチェック")
    s3_logger.info("Checking headers of uploaded CSV file")
    
    df_uploaded = pd.read_csv(storage_operator.get_object(key=s3_key))
    s3_logger.info(f"アップロードしたファイルのヘッダー: {df_uploaded.columns.to_list()}")
    s3_logger.info(f"Headers of uploaded file: {df_uploaded.columns.to_list()}")
    
    s3_logger.info(f"データが正常にS3にアップロードされました: {RESULTS_BUCKET_NAME}/{s3_key}")
    s3_logger.info(f"Data successfully uploaded to S3: {RESULTS_BUCKET_NAME}/{s3_key}")


def main() -> None:
    """メイン関数。
    Main function.
    """
    # 初期化 / Initialization
    args = init_args()
    run_date = args.run_date
    run_date_jst = from_utc_to_jst(run_date)
    run_date_no_dash = run_date_jst.strftime("%Y%m%d")
    task_type = args.type
    execution_id = args.execution_id
    
    # ログパスの生成 / Generate log path
    log_path = (
        S3_LOG_PATH.replace("@run_date", run_date_no_dash)
        .replace("@execution_id", execution_id)
        .replace("@type", task_type)
    )
    
    # S3Utilsのインスタンス化 / Instantiate S3Utils
    s3_utils = S3Utils(storage_operator)
    
    # 前回実行日時の取得 / Get last run date
    s3_logger.info(f"前回の{task_type}実行日時を取得")
    s3_logger.info(f"Getting last {task_type} run date")
    
    endfile_s3_key_prefix = S3_ENDFILE_PATH.replace("@type", task_type)
    
    last_run_date, last_run_endfile = s3_utils.get_last_run_date(
        endfile_prefix=endfile_s3_key_prefix,
        log_path=log_path,
        error_type="ERR",
        error_main="01",
        error_category="1",
        error_module="001",
        error_detail="001"
    )
    
    s3_logger.info(f"前回のendfileを取得しました: {last_run_endfile}")
    s3_logger.info(f"Retrieved last endfile: {last_run_endfile}")
    
    # DBからデータ取得 / Get data from DB
    df = get_data(last_run_date, run_date_jst.isoformat(), task_type, log_path)
    s3_logger.info(f"取得したデータ: {task_type} {len(df)}件")
    s3_logger.info(f"Retrieved data: {task_type} {len(df)} records")
    
    # S3にアップロード / Upload to S3
    s3_key = (
        S3_INPUT_PATH.replace("@run_date", run_date_no_dash)
        .replace("@execution_id", execution_id)
        .replace("@type", task_type)
    )
    
    upload_to_s3(df, s3_key, log_path)
    
    s3_logger.info("データ抽出処理が完了しました")
    s3_logger.info("Data extraction processing completed")


if __name__ == "__main__":
    main()

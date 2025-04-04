"""
Data Extraction Module.

This module extracts data from a database and uploads it to S3.
It supports both summarization and classification.

Contents:
---------
Functions:
    - init_args               : Initialize command line arguments
    - get_data                : Extract data from database based on date range
    - upload_to_s3            : Upload data to S3 storage
    - main                    : Main execution function
"""

import argparse
from io import StringIO

import pandas as pd
from classes.db_operation import db_operator
from classes.error_handler import CustomError, error_handler
from classes.logging_handler import application_logger
from classes.storage_operation import storage_operator
from common.const import RESULTS_BUCKET_NAME, S3_ENDFILE_PATH, S3_INPUT_PATH
from common.SQL_template import (query_classification_template,
                                 query_summarization_template)
from sqlalchemy.exc import OperationalError
from utils.jst_utils import from_utc_to_jst
from utils.s3_utils import S3Utils


def init_args() -> argparse.Namespace:
    """
    Initialize command line arguments.

    Returns:
        argparse.Namespace    : Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Execute data extraction task"
    )

    parser.add_argument(
        "--run_date",
        type=str,
        dest="run_date",
        required=True,
        help="Task execution date",
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
        help="Task type: summarization or classification",
    )
    return parser.parse_args()


@error_handler("data_extraction", "db_query")
def get_data(
    last_run_date: str,
    current_run_date: str,
    task_type: str,
    execution_id: str,
) -> pd.DataFrame:
    """
    Get yesterday's date and extract data from DB.

    Args:
        last_run_date         : Last run date
        current_run_date      : Current run date
        task_type             : Task type (summarization or classification)
        execution_id          : Execution ID for logging
    Returns:
        pd.DataFrame          : Extracted data as DataFrame
    """
    # Set application context for logging
    application_logger.set_execution_context("data_extraction", execution_id)

    application_logger.debug(
        f"Retrieved yesterday's date: {current_run_date}, type: {task_type}"
    )

    # Set dates in SQL template
    query_summarization = query_summarization_template.replace(
        "@last_run_date", last_run_date
    ).replace("@run_date", current_run_date)

    query_classification = query_classification_template.replace(
        "@last_run_date", last_run_date
    ).replace("@run_date", current_run_date)

    # Execute DB query
    try:
        if task_type == "summarization":
            application_logger.debug(
                f"Executing query (summarization):{query_summarization}"
            )
            result = db_operator.execute_statement("sql", query_summarization)
        else:  # classification
            application_logger.debug(
                f"Executing query (classification):{query_classification}"
            )
            result = db_operator.execute_statement("sql", query_classification)

        if result.rowcount == 0:
            application_logger.error(
                f"No data found : {last_run_date} -> {current_run_date}"
            )
            raise ValueError("No data found in specified date range")

    except CustomError:
        raise
    except OperationalError as error:
        application_logger.error(f"DB operation error: {str(error)}")
        raise
    except Exception as error:
        application_logger.error(f"Data retrieval error: {str(error)}")
        raise

    # Convert results to DataFrame
    df = pd.DataFrame.from_records(result.fetchall(), columns=result.keys())
    application_logger.info(f"Retrieved {len(df)} records from DB")

    return df


@error_handler("data_extraction", "s3_upload")
def upload_to_s3(df: pd.DataFrame, s3_key: str, execution_id: str) -> None:
    """
    Upload DataFrame to S3.

    Args:
        df                    : DataFrame to upload
        s3_key                : S3 key for the file
        execution_id          : Execution ID for logging
    """
    application_logger.info(
        f"Uploading data to S3: {RESULTS_BUCKET_NAME}/{s3_key}"
    )

    # Instantiate S3Utils
    s3_utils = S3Utils(storage_operator)

    try:
        storage_operator.put_df_to_s3(df=df, key=s3_key)

        # Validate uploaded file
        s3_utils.check_file_exists(
            key=s3_key,
            execution_id=execution_id,
            module_type="data_extraction",
            operation="check_file",
        )

        application_logger.info("Checking headers of uploaded CSV file")

        csv_content = storage_operator.get_object(key=s3_key)
        df_uploaded = pd.read_csv(StringIO(csv_content))

        application_logger.info(
            f"Headers of uploaded file: {df_uploaded.columns.to_list()}"
        )

        application_logger.info(
            f"Data successfully uploaded to S3: {RESULTS_BUCKET_NAME}/{s3_key}"
        )
    except Exception as e:
        application_logger.error(f"Error uploading to S3: {e}")
        raise


def main() -> None:
    """
    Main function for data extraction process.
    Parses arguments, retrieves data from DB and uploads to S3.
    """
    # Initialization
    args = init_args()
    run_date = args.run_date
    run_date_jst = from_utc_to_jst(run_date)
    run_date_no_dash = run_date_jst.strftime("%Y%m%d")
    task_type = args.type
    execution_id = args.execution_id

    # Initialize logger with context
    application_logger.set_execution_context("data_extraction", execution_id)

    # Instantiate S3Utils
    s3_utils = S3Utils(storage_operator)

    # Get last run date
    application_logger.info(f"Getting last {task_type} run date")

    endfile_s3_key_prefix = S3_ENDFILE_PATH.replace("@type", task_type)

    last_run_date, last_run_endfile = s3_utils.get_last_run_date(
        endfile_prefix=endfile_s3_key_prefix,
        execution_id=execution_id,
        module_type="data_extraction",
        operation="get_last_run",
    )

    application_logger.info(f"Retrieved last endfile: {last_run_endfile}")

    # Get data from DB
    df = get_data(
        last_run_date, run_date_jst.isoformat(), task_type, execution_id
    )
    application_logger.info(f"Retrieved data: {task_type} {len(df)} records")

    # Upload to S3
    s3_key = (
        S3_INPUT_PATH.replace("@run_date", run_date_no_dash)
        .replace("@execution_id", execution_id)
        .replace("@type", task_type)
    )

    upload_to_s3(df, s3_key, execution_id)

    application_logger.info("Data extraction processing completed")


if __name__ == "__main__":
    main()

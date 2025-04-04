"""
S3 Utility module.

This module provides common utilities for S3 operations.

Contents:
---------
Classes:
    - S3Utils                 : Common utility class for S3 operations
    - check_file_exists       : Check if a file exists on S3
    - check_file_headers      : Check headers of a CSV file on S3
    - get_last_run_date       : Get the date of the last run
    - manage_endfile          : Manage endfile (backup and create new)
"""

import os
from io import StringIO
from typing import Any, List, Tuple

import pandas as pd
from classes.error_handler import CustomError, error_handler
from classes.logging_handler import application_logger
from utils.jst_utils import from_utc_to_jst


class S3Utils:
    """
    Common utility class for S3 operations.
    """

    def __init__(self, storage_operator: Any) -> None:
        """
        Initialize S3Utils.

        Args:
            storage_operator  : Operator for S3 operations
        """

        self.storage_operator = storage_operator

    @error_handler("data_extraction", "check_file")
    def check_file_exists(
        self,
        key: str,
        execution_id: str,
        module_type: str = "data_extraction",
        operation: str = "check_file",
    ) -> bool:
        """
        Check if a file exists on S3.

        Args:
            key               : S3 object key
            execution_id      : Execution ID for error context
            module_type       : Module type for error handling
            operation         : Operation name for error handling
        Returns:
            bool              : True if the file exists
        """

        try:
            # Direct storage operation
            file_exists = self.storage_operator.check_object_exists(key=key)

            # Log output
            application_logger.info(
                f"File existence check: {key}, result: {file_exists}"
            )

            # Error handling if file doesn't exist
            if not file_exists:
                application_logger.error(f"File not found: {key}")
                raise ValueError(f"File not found: {key}")

            return True

        except CustomError:
            # Re-raise CustomError
            raise
        except Exception as error:
            # Handle other exceptions
            application_logger.error(
                f"Error checking file existence: {str(error)}"
            )
            raise

    @error_handler("data_extraction", "check_header")
    def check_file_headers(
        self,
        key: str,
        expected_headers: List[str],
        execution_id: str,
        module_type: str = "data_extraction",
        operation: str = "check_header",
    ) -> bool:
        """
        Check headers of a CSV file on S3.

        Args:
            key               : S3 object key
            expected_headers  : List of expected headers
            execution_id      : Execution ID for error context
            module_type       : Module type for error handling
            operation         : Operation name for error handling
        Returns:
            bool              : True if headers match expectations
        """

        try:
            # Get CSV data
            csv_data = self.storage_operator.get_object(key=key)

            # Handle empty CSV file
            if not csv_data or not csv_data.strip():
                application_logger.warning(f"Empty or invalid CSV file: {key}")
                raise ValueError(f"Empty or invalid CSV file: {key}")

            # Convert CSV to DataFrame
            df = pd.read_csv(StringIO(csv_data))

            # Handle case with no columns
            if df.empty or len(df.columns) == 0:
                application_logger.warning(
                    f"No columns found in CSV file: {key}"
                )
                raise ValueError(f"No columns found in CSV file: {key}")

            # Normalize headers case-insensitively
            actual_headers = [str(col).lower().strip() for col in df.columns]
            expected_normalized = [
                str(h).lower().strip() for h in expected_headers
            ]

            application_logger.info(f"File header check: {key}")
            application_logger.info(f"Actual headers: {actual_headers}")
            application_logger.info(f"Expected headers: {expected_normalized}")

            # Check headers (full match, not partial)
            if set(expected_normalized) != set(actual_headers):
                application_logger.error(
                    f"Header mismatch. Expected: {expected_normalized}, "
                    f"Actual: {actual_headers}"
                )
                raise ValueError(
                    f"Header mismatch. Expected: {expected_normalized}, "
                    f"Actual: {actual_headers}"
                )

            return True
        except CustomError:
            # Re-raise CustomError
            raise
        except Exception as error:
            # Handle other exceptions
            application_logger.error(
                f"Error checking file headers: {str(error)}"
            )
            raise

    @error_handler("data_extraction", "get_last_run")
    def get_last_run_date(
        self,
        endfile_prefix: str,
        execution_id: str,
        module_type: str = "data_extraction",
        operation: str = "get_last_run",
    ) -> Tuple[str, str]:
        """
        Get the date of the last run.

        Args:
            endfile_prefix    : Prefix for endfile
            execution_id      : Execution ID for error context
            module_type       : Module type for error handling
            operation         : Operation name for error handling
        Returns:
            Tuple[str, str]   : Last run date and path to last endfile
        """

        try:
            # Get endfile list (only .txt files)
            txt_files = list(
                filter(
                    lambda x: ".txt" in x,
                    self.storage_operator.list_objects(prefix=endfile_prefix),
                )
            )
            application_logger.debug(f"Detected endfiles: {txt_files}")

            # Handle case with no endfiles
            if not txt_files:
                application_logger.error("No endfile found")
                raise ValueError("No endfile found")

            # Select latest endfile (sort in descending order)
            last_run_endfile = sorted(txt_files, reverse=True)[0]
            application_logger.info(
                f"Retrieved last endfile: {last_run_endfile}"
            )

            # Extract date from filename (without extension)
            last_run_date = os.path.splitext(
                os.path.basename(last_run_endfile)
            )[0]

            return last_run_date, last_run_endfile

        except CustomError:
            # Re-raise CustomError
            raise
        except ValueError:
            # Re-raise ValueError (no endfile found)
            raise
        except Exception as error:
            # Handle other exceptions
            application_logger.error(
                f"Error getting last run date: {str(error)}"
            )
            raise

    @error_handler("data_extraction", "s3_upload")
    def manage_endfile(
        self,
        endfile_prefix: str,
        last_run_endfile: str,
        run_date: str,
        execution_id: str,
        module_type: str = "data_extraction",
        operation: str = "s3_upload",
    ) -> str:
        """
        Manage endfile (backup and create new).

        Args:
            endfile_prefix    : Prefix for endfile
            last_run_endfile  : Path to last endfile
            run_date          : Run date
            execution_id      : Execution ID for error context
            module_type       : Module type for error handling
            operation         : Operation name for error handling
        Returns:
            str               : Path to newly created endfile
        """

        try:
            # Generate destination key for backup
            destination_key = str(last_run_endfile).replace(
                endfile_prefix, os.path.join(endfile_prefix, "backup/")
            )

            # Log file movement
            application_logger.info(
                f"Moving endfile: {last_run_endfile} -> {destination_key}"
            )

            # Backup previous endfile
            self.storage_operator.move_object(
                destination_key=destination_key,
                source_key=last_run_endfile,
            )

            # Generate key for new endfile
            # Account for special characters and long paths
            safe_run_date = str(run_date).replace(":", "-").replace(" ", "_")
            endfile_key = os.path.join(
                endfile_prefix,
                f"{from_utc_to_jst(safe_run_date).isoformat()}.txt",
            )

            # Create new endfile
            application_logger.info(f"Creating new endfile: {endfile_key}")
            self.storage_operator.put_object(key=endfile_key, body=b"")

            return endfile_key

        except CustomError:
            # Re-raise CustomError
            raise
        except Exception as error:
            # Handle other exceptions
            application_logger.error(f"Error managing endfile: {str(error)}")
            raise

"""
Logging module.

This module centrally manages logging for the entire application.
Provides functionalities for standard logging, S3 logging, and error logging.

Contents:
---------
Helper Functions:
    - _format_error_message   : Format error codes into messages
    - _get_log_path           : Generate log file paths

Classes:
    - ApplicationLogger       : Main logger for application components

Utility Functions:
    - warning_logger          : Simplified warning logging for legacy support
"""

import logging
import os
from datetime import datetime, timezone
from logging import Formatter, Handler, Logger, getLogger
from typing import Dict, Optional

from classes.storage_operation import storage_operator
from common.error_map import (CATEGORY_MAP, ERROR_MESSAGE_MAP, MODULE_MAP,
                              PROCESSING_MAP, TYPE_MAP)
from utils.jst_utils import from_utc_to_jst

# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------
logger: Logger = getLogger("application")
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    handler = logging.StreamHandler()  # 型ヒントなし
    logger.addHandler(handler)

log_format = (
    "[%(asctime)s,%(msecs)d]"
    "{%(filename)s:%(lineno)s}"
    "%(levelname)s:%(message)s"
)
date_format = "%Y-%m-%d %H:%M:%S"
formatter = Formatter(fmt=log_format, datefmt=date_format)

for handler in logger.handlers:  # type: ignore
    handler_obj: Handler = handler
    handler_obj.setFormatter(formatter)

# Module specific settings
MODULE_CONFIG: Dict[str, Dict[str, str]] = {
    "data_extraction": {
        "error_main": "01",
        "log_path_template": "logs/data_extraction/data_extraction_{execution_id}.log",
    },
    "summarization": {
        "error_main": "02",
        "log_path_template": "logs/type=summarization/summarization_{execution_id}.log",
    },
    "classification": {
        "error_main": "03",
        "log_path_template": "logs/type=classification/classification_{execution_id}.log",
    },
}


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def _format_error_message(
    error_type: str,
    error_main: str,
    error_category: str,
    error_module: str,
    error_detail: str,
    extra_message: str = "",
) -> str:
    """
    Generate error message from error codes.

    Args:
        error_type       : Error type (ERR/WRN)
        error_main       : Error main (01-03)
        error_category   : Error category (1-2)
        error_module     : Error module (001-006)
        error_detail     : Error detail (001-013)
        extra_message    : Additional message (default: "")

    Returns:
        str              : Formatted error message
    """
    try:
        message = (
            f"{TYPE_MAP[error_type]} - {PROCESSING_MAP[error_main]} - "
            f"{CATEGORY_MAP[error_category]} - {MODULE_MAP[error_module]}: "
            f"{ERROR_MESSAGE_MAP[error_detail]}"
        )
    except KeyError:
        message = "UNKNOWN ERROR/WARNING"

    if extra_message:
        message += f" - {extra_message}"

    # Create error code by joining components
    code_parts = [
        error_type,
        error_main,
        error_category,
        error_module,
        error_detail,
    ]
    code = "_".join(code_parts)
    return f"[{code}] {message}"


def _get_log_path(module_type: str, execution_id: Optional[str] = None) -> str:
    """
    Get log path for the specified module.

    Args:
        module_type      : Module type
        execution_id     : Execution ID (optional)

    Returns:
        str              : Log path
    """
    if module_type in MODULE_CONFIG:
        log_path_template = MODULE_CONFIG[module_type]["log_path_template"]
        if execution_id:
            return log_path_template.format(execution_id=execution_id)
        return log_path_template.format(execution_id="default")

    return f"logs/{module_type}.log"


# ------------------------------------------------------------------------------
# Application Logger Class
# ------------------------------------------------------------------------------
class ApplicationLogger:
    """
    Application logging class.

    Provides logging to both standard output and S3.
    """

    def __init__(
        self,
        module_type: Optional[str] = None,
        execution_id: Optional[str] = None,
    ):
        """
        Initialize logger.

        Args:
            module_type      : Module type (optional)
            execution_id     : Execution ID (optional)
        """
        self.logger: Logger = logger
        self.module_type: Optional[str] = module_type
        self.execution_id: Optional[str] = execution_id

        # Record execution ID if provided
        if module_type and execution_id:
            self.set_execution_context(module_type, execution_id)

    def set_execution_context(
        self, module_type: str, execution_id: str
    ) -> None:
        """
        Set execution context.

        Args:
            module_type      : Module type
            execution_id     : Execution ID
        """
        self.module_type = module_type
        self.execution_id = execution_id
        self.info(
            f"Starting {module_type} processing. Execution ID: {execution_id}"
        )

    def _get_log_path(self) -> str:
        """
        Get log path based on current module type and execution ID.

        Returns:
            str              : Log path
        """
        if self.module_type:
            return _get_log_path(self.module_type, self.execution_id)
        return "logs/application.log"

    def log(self, level: int, message: str, log_to_s3: bool = True) -> None:
        """
        Log a general message.

        Args:
            level            : Log level
            message          : Log message
            log_to_s3        : Whether to log to S3 (default: True)
        """
        # Log to console
        self.logger.log(level, message)

        # Log to S3 if requested
        if log_to_s3:
            log_path = self._get_log_path()
            timestamp = from_utc_to_jst(datetime.now(timezone.utc).isoformat())
            log_message = f"[{timestamp}] {message}"

            # Ensure directory exists
            directory = os.path.dirname(log_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            try:
                storage_operator.put_log_to_s3(log_message, log_path)
            except Exception as error:
                # Log S3 storage failures to console
                self.logger.error(f"Failed to save log to S3: {str(error)}")

    def debug(self, message: str, log_to_s3: bool = True) -> None:
        """
        Log a debug message.

        Args:
            message          : Log message
            log_to_s3        : Whether to log to S3 (default: True)
        """
        self.log(logging.DEBUG, message, log_to_s3)

    def info(self, message: str, log_to_s3: bool = True) -> None:
        """
        Log an info message.

        Args:
            message          : Log message
            log_to_s3        : Whether to log to S3 (default: True)
        """
        self.log(logging.INFO, message, log_to_s3)

    def warning(self, message: str, log_to_s3: bool = True) -> None:
        """
        Log a warning message.

        Args:
            message          : Log message
            log_to_s3        : Whether to log to S3 (default: True)
        """
        self.log(logging.WARNING, message, log_to_s3)

    def error(self, message: str, log_to_s3: bool = True) -> None:
        """
        Log an error message.

        Args:
            message          : Log message
            log_to_s3        : Whether to log to S3 (default: True)
        """
        self.log(logging.ERROR, message, log_to_s3)

    def error_with_code(
        self,
        error_type: str,
        error_main: str,
        error_category: str,
        error_module: str,
        error_detail: str,
        extra_message: str = "",
    ) -> str:
        """
        Log a coded error message.

        Args:
            error_type       : Error type (ERR/WRN)
            error_main       : Error main (01-03)
            error_category   : Error category (1-2)
            error_module     : Error module (001-006)
            error_detail     : Error detail (001-013)
            extra_message    : Additional message (default: "")

        Returns:
            str              : Logged error message
        """
        error_message = _format_error_message(
            error_type,
            error_main,
            error_category,
            error_module,
            error_detail,
            extra_message,
        )

        self.error(error_message)
        return error_message

    def warning_with_code(
        self,
        error_type: str,
        error_main: str,
        error_category: str,
        error_module: str,
        error_detail: str,
        extra_message: str = "",
    ) -> str:
        """
        Log a coded warning message.

        Args:
            error_type       : Error type (ERR/WRN)
            error_main       : Error main (01-03)
            error_category   : Error category (1-2)
            error_module     : Error module (001-006)
            error_detail     : Error detail (001-013)
            extra_message    : Additional message (default: "")

        Returns:
            str              : Logged warning message
        """
        warning_message = _format_error_message(
            error_type,
            error_main,
            error_category,
            error_module,
            error_detail,
            extra_message,
        )

        self.warning(warning_message)
        return warning_message

    def get_error_main_for_module(self) -> str:
        """
        Get error main code based on current module type.

        Returns:
            str              : Error main code
        """
        if self.module_type and self.module_type in MODULE_CONFIG:
            return MODULE_CONFIG[self.module_type]["error_main"]
        return "99"


# ------------------------------------------------------------------------------
# Compatibility Functions
# ------------------------------------------------------------------------------
# Flag to track if warnings have been logged
has_warning: bool = False


def warning_logger(message: str) -> None:
    """
    Helper function to log warning messages.

    Args:
        message          : Log message
    """
    global has_warning
    has_warning = True
    logger.warning(message)


# Create singleton instance
application_logger = ApplicationLogger()

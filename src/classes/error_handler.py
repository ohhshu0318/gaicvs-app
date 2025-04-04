"""
Error handling module.

This module centrally manages error handling for the entire application.
Provides functionalities for error definition, logging, and exception handling.

Contents:
---------
Utility Functions:
    - is_safe_path            : Verify path safety
    - sanitize_for_logging    : Clean messages for logging

Error Classes:
    - CustomError             : Application-specific exception

Internal Helper Functions:
    - _format_error_message   : Format error codes into messages
    - _get_error_codes        : Retrieve error codes for modules
    - _get_log_path           : Generate log file paths

Core Logging Functions:
    - log_message             : Log to console/S3
    - log_error               : Log error with codes
    - log_and_raise_error     : Log and raise exception

Error Handling Wrappers:
    - error_handler           : Function decorator
    - error_context           : Context manager
    - safe_execute            : Execute with fallback

Legacy Support:
    - ErrorHandler            : Static methods for compatibility
"""

import functools
import inspect
import logging
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

from classes.storage_operation import storage_operator
from common.error_map import (CATEGORY_MAP, ERROR_MESSAGE_MAP, MODULE_MAP,
                              PROCESSING_MAP, TYPE_MAP)
from utils.jst_utils import from_utc_to_jst

# Type variable definition
T = TypeVar("T")
R = TypeVar("R")

# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------
logger = logging.getLogger("error_handler")
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    log_format = (
        "[%(asctime)s,%(msecs)d]"
        "{%(filename)s:%(lineno)s}"
        "%(levelname)s:%(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Module specific settings
MODULE_CONFIG = {
    "data_extraction": {
        "error_main": "01",
        "log_path_template": "logs/data_extraction_{execution_id}.log",
        "error_codes": {
            "db_query": {"category": "2", "module": "002", "detail": "007"},
            "s3_upload": {"category": "1", "module": "001", "detail": "002"},
            "get_last_run": {
                "category": "1",
                "module": "001",
                "detail": "001",
            },
        },
    },
    "summarization": {
        "error_main": "02",
        "log_path_template": "logs/summarization_{execution_id}.log",
        "error_codes": {
            "read_csv": {"category": "1", "module": "001", "detail": "001"},
            "generate_summaries": {
                "category": "1",
                "module": "004",
                "detail": "003",
            },
            "save_output": {"category": "1", "module": "001", "detail": "002"},
            "write_db": {"category": "1", "module": "002", "detail": "002"},
            "check_file": {"category": "1", "module": "001", "detail": "001"},
            "check_header": {
                "category": "1",
                "module": "001",
                "detail": "008",
            },
        },
    },
    "classification": {
        "error_main": "03",
        "log_path_template": "logs/classification_{execution_id}.log",
        "error_codes": {
            "get_prompt": {"category": "1", "module": "001", "detail": "001"},
            "load_input": {"category": "1", "module": "001", "detail": "001"},
            "search_similar": {
                "category": "1",
                "module": "005",
                "detail": "003",
            },
            "get_metadata": {
                "category": "1",
                "module": "002",
                "detail": "004",
            },
            "get_labels": {"category": "1", "module": "002", "detail": "004"},
            "save_db": {"category": "1", "module": "002", "detail": "002"},
            "check_file": {"category": "1", "module": "001", "detail": "001"},
            "check_header": {
                "category": "1",
                "module": "001",
                "detail": "008",
            },
        },
    },
}

# List of allowed base directories
ALLOWED_BASE_DIRS = [
    os.path.abspath("logs"),
]


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def is_safe_path(base_dirs: List[str], path: str) -> bool:
    """
    Check if the given path is safe.

    Args:
        base_dirs             : List of allowed base directories
        path                  : Path to verify
    Returns:
        bool                  : True if the path is safe, False otherwise
    """

    if not path:
        return False

    # Convert to absolute path and ensure path is normalized
    abs_path = os.path.abspath(os.path.normpath(path))

    # Check if the path is within any of the allowed base directories
    return any(abs_path.startswith(base_dir) for base_dir in base_dirs)


def sanitize_for_logging(message: Any) -> str:
    """
    Sanitize a message before logging it.

    Args:
        message               : Message to sanitize
    Returns:
        str                   : Sanitized message
    """

    # Convert to string if not already a string
    if not isinstance(message, str):
        message = str(message)

    # Replace newlines and control characters
    sanitized: str = re.sub(r"[\r\n\t]+", " ", message)

    # Replace other potentially dangerous characters
    sanitized = sanitized.replace("\\", "\\\\")

    return sanitized


# ------------------------------------------------------------------------------
# Error Classes
# ------------------------------------------------------------------------------
class CustomError(Exception):
    """
    Custom error class for application-specific error information.
    Error object containing application-specific error codes and messages.
    """

    def __init__(
        self,
        error_type: str,
        main: str,
        category: str,
        module: str,
        detail: str,
    ) -> None:
        """
        Initialize custom error.

        Args:
            error_type        : Error type (ERR/WRN)
            main              : Error main code (01-03)
            category          : Error category code (1-2)
            module            : Error module code (001-006)
            detail            : Error detail code (001-013)
        """
        self.type = error_type
        self.main = main
        self.category = category
        self.module = module
        self.detail = detail

        try:
            self.error_message = (
                f"{TYPE_MAP[error_type]} - {PROCESSING_MAP[main]} - "
                f"{CATEGORY_MAP[category]} - {MODULE_MAP[module]}: "
                f"{ERROR_MESSAGE_MAP[detail]}"
            )
        except KeyError:
            self.error_message = "UNKNOWN ERROR"

        super().__init__(self.error_message)

    def __str__(self) -> str:
        """
        String representation of the error.

        Returns:
            str               : Formatted error message with error code
        """
        code_parts = [
            self.type,
            self.main,
            self.category,
            self.module,
            self.detail,
        ]
        error_code = "_".join(code_parts)
        return f"[{error_code}] {self.error_message}"


# ------------------------------------------------------------------------------
# Internal Helper Functions
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
        error_type            : Error type (ERR/WRN)
        error_main            : Error main code (01-03)
        error_category        : Error category code (1-2)
        error_module          : Error module code (001-006)
        error_detail          : Error detail code (001-013)
        extra_message         : Additional message (default: "")
    Returns:
        str                   : Formatted error message
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
        # Sanitize the additional message
        extra_message = sanitize_for_logging(extra_message)
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


def _get_error_codes(module_type: str, operation: str) -> Dict[str, str]:
    """
    Get error codes for the specified module and operation.

    Args:
        module_type           : Module type
        operation             : Operation name
    Returns:
        Dict[str, str]        : Dictionary containing error code components
    """

    # Default error codes for unknown module types
    default_codes = {
        "main": "99",
        "category": "1",
        "module": "006",
        "detail": "001",
    }

    if module_type not in MODULE_CONFIG:
        return default_codes

    try:
        config = MODULE_CONFIG[module_type]
        error_main = str(config["error_main"])

        # Return operation-specific error codes if available
        if operation in config["error_codes"]:
            op_codes = config["error_codes"][operation]  # type: ignore
            return {
                "main": error_main,
                "category": str(op_codes["category"]),
                "module": str(op_codes["module"]),
                "detail": str(op_codes["detail"]),
            }

        # Default error codes for unknown operations within a known module
        return {
            "main": error_main,
            "category": "1",
            "module": "006",
            "detail": "001",
        }
    except (KeyError, TypeError):
        return default_codes


def _get_log_path(module_type: str, execution_id: Optional[str] = None) -> str:
    """
    Get log path for the specified module.

    Args:
        module_type           : Module type
        execution_id          : Execution ID (optional)
    Returns:
        str                   : Log file path
    """

    # For known module types, use the configured log path template
    if module_type in MODULE_CONFIG:
        try:
            # Explicitly cast to string to satisfy type checker
            config = MODULE_CONFIG[module_type]
            log_path_template: str = str(config["log_path_template"])

            # Handle execution_id
            if execution_id:
                # Sanitize execution_id for safe path construction
                safe_execution_id = re.sub(r"[^\w\-]", "_", str(execution_id))
                path = log_path_template.format(execution_id=safe_execution_id)
            else:
                path = log_path_template.format(execution_id="default")

            # Verify the generated path is safe
            if is_safe_path(ALLOWED_BASE_DIRS, path):
                return path
            else:
                logger.warning(
                    f"Unsafe log path detected: {sanitize_for_logging(path)}"
                )
                return "logs/error.log"  # Safe fallback path
        except (KeyError, ValueError, AttributeError):
            # Handle unexpected configuration format
            return f"logs/{sanitize_for_logging(module_type)}.log"

    # For unknown module types, create a generic log path
    return f"logs/{sanitize_for_logging(module_type)}.log"


# ------------------------------------------------------------------------------
# Core Logging Functions
# ------------------------------------------------------------------------------
def log_message(
    log_path: str,
    level: int,
    message: str,
    log_to_s3: bool = True,
    log_to_console: bool = True,
) -> None:
    """
    Log a message to console and/or S3.

    Args:
        log_path              : Log file path
        level                 : Log level (from logging module)
        message               : Log message
        log_to_s3             : Whether to log to S3 (default: True)
        log_to_console        : Whether to log to console (default: True)
    """

    # Sanitize the message before logging
    sanitized_message = sanitize_for_logging(message)

    # Log to console if requested
    if log_to_console:
        logger.log(level, sanitized_message)

    # Log to S3 if requested and path is provided
    if log_to_s3 and log_path:
        # Convert UTC timestamp to JST
        timestamp = from_utc_to_jst(datetime.now(timezone.utc).isoformat())
        log_entry = f"[{timestamp}] {sanitized_message}"

        # Verify the path is safe before writing
        if not is_safe_path(ALLOWED_BASE_DIRS, log_path):
            logger.warning(
                f"Unsafe log path detected: {sanitize_for_logging(log_path)}"
            )
            return

        # Try to save log to S3
        try:
            storage_operator.put_log_to_s3(log_entry, log_path)
        except Exception as error:
            # Log S3 storage failures to console
            error_msg = sanitize_for_logging(str(error))
            logger.error(f"Failed to save log to S3: {error_msg}")


def log_error(
    log_path: str,
    error_type: str,
    error_main: str,
    error_category: str,
    error_module: str,
    error_detail: str,
    extra_message: str = "",
    log_to_s3: bool = True,
) -> str:
    """
    Log an error message.

    Args:
        log_path              : Log file path
        error_type            : Error type (ERR/WRN)
        error_main            : Error main code (01-03)
        error_category        : Error category code (1-2)
        error_module          : Error module code (001-006)
        error_detail          : Error detail code (001-013)
        extra_message         : Additional message (default: "")
        log_to_s3             : Whether to log to S3 (default: True)
    Returns:
        str                   : Logged error message
    """
    # Sanitize the additional message
    sanitized_extra_message = sanitize_for_logging(extra_message)

    # Format the complete error message
    error_message = _format_error_message(
        error_type,
        error_main,
        error_category,
        error_module,
        error_detail,
        sanitized_extra_message,
    )

    # Determine log level based on error type
    level = logging.ERROR if error_type == "ERR" else logging.WARNING

    # Verify the path is safe before writing
    if not is_safe_path(ALLOWED_BASE_DIRS, log_path):
        safe_log_path = "logs/error.log"
        logger.warning(
            f"Unsafe log path detected. "
            f"Using default path: {log_path} -> {safe_log_path}"
        )
        log_path = safe_log_path

    # Log the error message
    log_message(log_path, level, error_message, log_to_s3)

    return error_message


def log_and_raise_error(
    log_path: str,
    error_type: str,
    error_main: str,
    error_category: str,
    error_module: str,
    error_detail: str,
    extra_message: str = "",
) -> None:
    """
    Log an error and raise an exception.

    Args:
        log_path              : Log file path
        error_type            : Error type (ERR/WRN)
        error_main            : Error main code (01-03)
        error_category        : Error category code (1-2)
        error_module          : Error module code (001-006)
        error_detail          : Error detail code (001-013)
        extra_message         : Additional message (default: "")
    Raises:
        CustomError           : Raises CustomError using provided codes
    """
    # Sanitize the additional message
    sanitized_extra_message = sanitize_for_logging(extra_message)

    # Log the error
    log_error(
        log_path,
        error_type,
        error_main,
        error_category,
        error_module,
        error_detail,
        sanitized_extra_message,
    )

    # If error_detail is already a CustomError instance, raise it directly
    if isinstance(error_detail, CustomError):
        raise error_detail
    else:
        # Otherwise, create and raise a new CustomError
        raise CustomError(
            error_type, error_main, error_category, error_module, error_detail
        )


# ------------------------------------------------------------------------------
# Error Handling Wrappers
# ------------------------------------------------------------------------------
def error_handler(
    module_type: str,
    operation: str,
    execution_id: Optional[str] = None,
    error_type: str = "ERR",
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Error handling decorator.

    Args:
        module_type           : Module type
        operation             : Operation name
        execution_id          : Execution ID (optional)
        error_type            : Error type (default: "ERR")

    Returns:
        Callable              : Decorator with error handling capabilities
    """
    # Sanitize execution_id to avoid path traversal
    if execution_id:
        execution_id = re.sub(r"[^\w\-]", "_", str(execution_id))

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            # Get error codes for this module and operation
            error_codes = _get_error_codes(module_type, operation)

            # Get log path for this module
            log_path = _get_log_path(module_type, execution_id)

            # Log the start of function execution
            logger.info(f"Starting execution of function {func.__name__}")

            try:
                return func(*args, **kwargs)
            except CustomError:
                # Pass through CustomError exceptions without modification
                raise
            except Exception as exception:
                # Get stack trace for error context
                stack = inspect.stack()
                caller = stack[1] if len(stack) > 1 else None
                file_name = caller.filename if caller else "unknown"
                line_no = caller.lineno if caller else 0

                # Format error information
                extra_info = (
                    f"File: {os.path.basename(file_name)}, "
                    f"Line: {line_no}, "
                    f"Error: {sanitize_for_logging(str(exception))}"
                )

                # Log the error and raise a CustomError
                log_and_raise_error(
                    log_path,
                    error_type,
                    error_codes["main"],
                    error_codes["category"],
                    error_codes["module"],
                    error_codes["detail"],
                    extra_info,
                )
                # This line never executes but satisfies type checker
                raise

        return wrapper

    return decorator


@contextmanager
def error_context(
    module_type: str,
    operation: str,
    execution_id: Optional[str] = None,
    error_type: str = "ERR",
) -> Generator[None, None, None]:
    """
    Error handling context manager.

    Args:
        module_type           : Module type
        operation             : Operation name
        execution_id          : Execution ID (optional)
        error_type            : Error type (default: "ERR")
    Yields:
        None
    Raises:
        CustomError           : On exception, logs and raises a CustomError
    """
    # Sanitize execution_id to avoid path traversal
    if execution_id:
        execution_id = re.sub(r"[^\w\-]", "_", str(execution_id))

    # Get error codes for this module and operation
    error_codes = _get_error_codes(module_type, operation)

    # Get log path for this module
    log_path = _get_log_path(module_type, execution_id)

    try:
        yield
    except CustomError:
        # Pass through CustomError exceptions without modification
        raise
    except Exception as exception:
        # Get stack trace for error context
        stack = inspect.stack()
        caller = stack[1] if len(stack) > 1 else None
        file_name = caller.filename if caller else "unknown"
        line_no = caller.lineno if caller else 0

        # Format error information
        extra_info = (
            f"File: {os.path.basename(file_name)}, "
            f"Line: {line_no}, "
            f"Error: {sanitize_for_logging(str(exception))}"
        )

        # Log the error and raise a CustomError
        log_and_raise_error(
            log_path,
            error_type,
            error_codes["main"],
            error_codes["category"],
            error_codes["module"],
            error_codes["detail"],
            extra_info,
        )
        # This line never executes but satisfies type checker
        raise


def safe_execute(
    func: Callable[..., T],
    module_type: str,
    operation: str,
    args: Optional[List[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    execution_id: Optional[str] = None,
    error_type: str = "ERR",
    default: Optional[T] = None,
) -> Optional[T]:
    """
    Safely execute a function and handle errors appropriately.

    Args:
        func                  : Function to execute
        module_type           : Module type
        operation             : Operation name
        args                  : Function positional arguments (default: None)
        kwargs                : Function keyword arguments (default: None)
        execution_id          : Execution ID (optional)
        error_type            : Error type (default: "ERR")
        default               : Return value on error (default: None)
    Returns:
        Optional[T]           : Function return value or default value on error
    """
    args = args or []
    kwargs = kwargs or {}

    # Sanitize execution_id to avoid path traversal
    if execution_id:
        execution_id = re.sub(r"[^\w\-]", "_", str(execution_id))

    # Get error codes for this module and operation
    error_codes = _get_error_codes(module_type, operation)

    # Get log path for this module
    log_path = _get_log_path(module_type, execution_id)

    try:
        return func(*args, **kwargs)
    except CustomError:
        # Pass through CustomError exceptions without modification
        raise
    except Exception as exception:
        # Get stack trace for error context
        stack = inspect.stack()
        caller = stack[1] if len(stack) > 1 else None
        file_name = caller.filename if caller else "unknown"
        line_no = caller.lineno if caller else 0

        # Format error information
        extra_info = (
            f"File: {os.path.basename(file_name)}, "
            f"Line: {line_no}, "
            f"Error: {sanitize_for_logging(str(exception))}"
        )

        # Log the error but don't raise (unlike error_handler/error_context)
        log_error(
            log_path,
            error_type,
            error_codes["main"],
            error_codes["category"],
            error_codes["module"],
            error_codes["detail"],
            extra_info,
        )

        # Return the default value instead of raising an exception
        return default


# ------------------------------------------------------------------------------
# Legacy Support
# ------------------------------------------------------------------------------
class ErrorHandler:
    """
    Error handling class for legacy code support.
    Provides static methods for handling error logs and exceptions.
    """

    @staticmethod
    def raise_error(
        error_type: str,
        main: str,
        category: str,
        module: str,
        detail: str,
        from_exception: Optional[Exception] = None,
    ) -> None:
        """
        Raise a CustomError exception.

        Args:
            error_type        : Error type (ERR/WRN)
            main              : Error main code (01-03)
            category          : Error category code (1-2)
            module            : Error module code (001-006)
            detail            : Error detail code (001-013)
            from_exception    : Original exception (default: None)
        """
        custom_error = CustomError(error_type, main, category, module, detail)
        if from_exception:
            raise custom_error from from_exception
        else:
            raise custom_error

    @staticmethod
    def log_error(
        error_type: str,
        main: str,
        category: str,
        module: str,
        detail: str,
        extra_message: str = "",
    ) -> None:
        """
        Log an error message.

        Args:
            error_type        : Error type (ERR/WRN)
            main              : Error main code (01-03)
            category          : Error category code (1-2)
            module            : Error module code (001-006)
            detail            : Error detail code (001-013)
            extra_message     : Additional message (default: "")
        """
        # Sanitize the additional message
        sanitized_extra_message = sanitize_for_logging(extra_message)

        error_message = _format_error_message(
            error_type, main, category, module, detail, sanitized_extra_message
        )
        logger.error(error_message)

    @staticmethod
    def log_warning(
        error_type: str,
        main: str,
        category: str,
        module: str,
        detail: str,
        extra_message: str = "",
    ) -> None:
        """
        Log a warning message.

        Args:
            error_type        : Error type (ERR/WRN)
            main              : Error main code (01-03)
            category          : Error category code (1-2)
            module            : Error module code (001-006)
            detail            : Error detail code (001-013)
            extra_message     : Additional message (default: "")
        """
        # Sanitize the additional message
        sanitized_extra_message = sanitize_for_logging(extra_message)

        warning_message = _format_error_message(
            error_type, main, category, module, detail, sanitized_extra_message
        )
        logger.warning(warning_message)

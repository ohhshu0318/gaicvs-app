"""エラーハンドリングモジュール。
Error handling module.

このモジュールは、アプリケーション全体のエラーハンドリングを一元的に管理します。
This module centrally manages error handling for the entire application.

以下の機能を提供します:
Provides the following functionalities:
- エラー定義と管理 (Error definition and management)
- エラーログ記録 (Error logging)
- 例外発生 (Exception raising)
- エラーハンドリングのためのデコレータとコンテキストマネージャー 
  (Decorators and context managers for error handling)
"""

import functools
import inspect
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from classes.storage_operation import storage_operator
from common.error_map import (
    CATEGORY_MAP,
    ERROR_MESSAGE_MAP,
    MODULE_MAP,
    PROCESSING_MAP,
    TYPE_MAP,
)
from common.utils import from_utc_to_jst

# 型変数定義 (Type variable definition)
T = TypeVar("T")
R = TypeVar("R")

# ロギング設定 (Logging configuration)
logger = logging.getLogger("error_handler")
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s,%(msecs)d]{%(filename)s:%(lineno)s}%(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# モジュール別設定 (Module specific settings)
MODULE_CONFIG = {
    "data_extraction": {
        "error_main": "01",
        "log_path_template": "logs/data_extraction_{execution_id}.log",
        "error_codes": {
            "db_query": {"category": "2", "module": "002", "detail": "007"},
            "s3_upload": {"category": "1", "module": "001", "detail": "002"},
            "get_last_run": {"category": "1", "module": "001", "detail": "001"},
        }
    },
    "summarization": {
        "error_main": "02",
        "log_path_template": "logs/summarization_{execution_id}.log",
        "error_codes": {
            "read_csv": {"category": "1", "module": "001", "detail": "001"},
            "generate_summaries": {"category": "1", "module": "004", "detail": "003"},
            "save_output": {"category": "1", "module": "001", "detail": "002"},
            "write_db": {"category": "1", "module": "002", "detail": "002"},
            "check_file": {"category": "1", "module": "001", "detail": "001"},
            "check_header": {"category": "1", "module": "001", "detail": "008"},
        }
    },
    "classification": {
        "error_main": "03",
        "log_path_template": "logs/classification_{execution_id}.log",
        "error_codes": {
            "get_prompt": {"category": "1", "module": "001", "detail": "001"},
            "load_input": {"category": "1", "module": "001", "detail": "001"},
            "search_similar": {"category": "1", "module": "005", "detail": "003"},
            "get_metadata": {"category": "1", "module": "002", "detail": "004"},
            "get_labels": {"category": "1", "module": "002", "detail": "004"},
            "save_db": {"category": "1", "module": "002", "detail": "002"},
            "check_file": {"category": "1", "module": "001", "detail": "001"},
            "check_header": {"category": "1", "module": "001", "detail": "008"},
        }
    }
}


class CustomError(Exception):
    """カスタムエラークラス。
    Custom error class.

    アプリケーション固有のエラー情報を含むエラーオブジェクト。
    Error object containing application-specific error information.
    """

    def __init__(
        self, error_type: str, main: str, category: str, module: str, detail: str
    ) -> None:
        """カスタムエラーを初期化。
        Initialize custom error.

        Args:
            error_type: エラータイプ (ERR/WRN) / Error type (ERR/WRN)
            main: エラーメイン (01-03) / Error main (01-03)
            category: エラーカテゴリ (1-2) / Error category (1-2)
            module: エラーモジュール (001-006) / Error module (001-006)
            detail: エラー詳細 (001-013) / Error detail (001-013)
        """
        self.type = error_type
        self.main = main
        self.category = category
        self.module = module
        self.detail = detail

        try:
            self.error_message = (
                f"{TYPE_MAP[error_type]} - {PROCESSING_MAP[main]} - "
                f"{CATEGORY_MAP[category]} - {MODULE_MAP[module]}: {ERROR_MESSAGE_MAP[detail]}"
            )
        except KeyError:
            self.error_message = "UNKNOWN ERROR"

        super().__init__(self.error_message)

    def __str__(self) -> str:
        """エラーの文字列表現。
        String representation of the error.

        Returns:
            フォーマット済みのエラーメッセージ / Formatted error message
        """
        return (
            f"[{self.type}_{self.main}_{self.category}_{self.module}_{self.detail}] "
            f"{self.error_message}"
        )


def _format_error_message(
    error_type: str,
    error_main: str,
    error_category: str,
    error_module: str,
    error_detail: str,
    extra_message: str = "",
) -> str:
    """エラーコードからエラーメッセージを生成する。
    Generate error message from error codes.

    Args:
        error_type: エラータイプ (ERR/WRN) / Error type (ERR/WRN)
        error_main: エラーメイン (01-03) / Error main (01-03)
        error_category: エラーカテゴリ (1-2) / Error category (1-2)
        error_module: エラーモジュール (001-006) / Error module (001-006)
        error_detail: エラー詳細 (001-013) / Error detail (001-013)
        extra_message: 追加メッセージ（デフォルト: ""）/ Additional message (default: "")

    Returns:
        フォーマット済みのエラーメッセージ / Formatted error message
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

    code = f"{error_type}_{error_main}_{error_category}_{error_module}_{error_detail}"
    return f"[{code}] {message}"


def _get_error_codes(module_type: str, operation: str) -> Dict[str, str]:
    """指定されたモジュールとオペレーションのエラーコードを取得する。
    Get error codes for the specified module and operation.

    Args:
        module_type: モジュールタイプ / Module type
        operation: オペレーション / Operation

    Returns:
        エラーコード情報 / Error code information
    """
    if module_type not in MODULE_CONFIG:
        return {
            "main": "99",
            "category": "1",
            "module": "006",
            "detail": "001",
        }

    config = MODULE_CONFIG[module_type]
    error_main = config["error_main"]
    
    if operation in config["error_codes"]:
        codes = config["error_codes"][operation]
        return {
            "main": error_main,
            "category": codes["category"],
            "module": codes["module"],
            "detail": codes["detail"],
        }
    
    return {
        "main": error_main,
        "category": "1",
        "module": "006",
        "detail": "001",
    }


def _get_log_path(module_type: str, execution_id: Optional[str] = None) -> str:
    """指定されたモジュールのログパスを取得する。
    Get log path for the specified module.

    Args:
        module_type: モジュールタイプ / Module type
        execution_id: 実行ID（オプション）/ Execution ID (optional)

    Returns:
        ログパス / Log path
    """
    if module_type in MODULE_CONFIG:
        log_path_template = MODULE_CONFIG[module_type]["log_path_template"]
        if execution_id:
            return log_path_template.format(execution_id=execution_id)
        return log_path_template.format(execution_id="default")
    
    return f"logs/{module_type}.log"


def log_message(
    log_path: str, 
    level: int, 
    message: str, 
    log_to_s3: bool = True,
    log_to_console: bool = True
) -> None:
    """ログメッセージを記録する。
    Log a message.

    Args:
        log_path: ログファイルのパス / Log file path
        level: ログレベル / Log level
        message: ログメッセージ / Log message
        log_to_s3: S3にログを記録するかどうか（デフォルト: True）/ Whether to log to S3 (default: True)
        log_to_console: コンソールにログを記録するかどうか（デフォルト: True）/ Whether to log to console (default: True)
    """
    if log_to_console:
        logger.log(level, message)
    
    if log_to_s3:
        timestamp = from_utc_to_jst(datetime.now(timezone.utc).isoformat())
        log_message = f"[{timestamp}] {message}"
        
        # S3ディレクトリが存在することを確認
        if log_path:
            directory = os.path.dirname(log_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # S3にログを保存
            try:
                storage_operator.put_log_to_s3(log_message, log_path)
            except Exception as e:
                # S3へのログ保存に失敗した場合はコンソールに出力
                logger.error(f"S3へのログ保存に失敗しました: {str(e)}")
                logger.error(f"Failed to save log to S3: {str(e)}")


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
    """エラーメッセージをログに記録する。
    Log an error message.

    Args:
        log_path: ログファイルのパス / Log file path
        error_type: エラータイプ (ERR/WRN) / Error type (ERR/WRN)
        error_main: エラーメイン (01-03) / Error main (01-03)
        error_category: エラーカテゴリ (1-2) / Error category (1-2)
        error_module: エラーモジュール (001-006) / Error module (001-006)
        error_detail: エラー詳細 (001-013) / Error detail (001-013)
        extra_message: 追加メッセージ（デフォルト: ""）/ Additional message (default: "")
        log_to_s3: S3にログを記録するかどうか（デフォルト: True）/ Whether to log to S3 (default: True)

    Returns:
        記録されたエラーメッセージ / Logged error message
    """
    error_message = _format_error_message(
        error_type, error_main, error_category, error_module, error_detail, extra_message
    )
    
    # ログレベルを決定（ERRはERROR、WRNはWARNING）
    level = logging.ERROR if error_type == "ERR" else logging.WARNING
    
    # ログメッセージを記録
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
    """エラーをログに記録し、例外を発生させる。
    Log an error and raise an exception.

    Args:
        log_path: ログファイルのパス / Log file path
        error_type: エラータイプ (ERR/WRN) / Error type (ERR/WRN)
        error_main: エラーメイン (01-03) / Error main (01-03)
        error_category: エラーカテゴリ (1-2) / Error category (1-2)
        error_module: エラーモジュール (001-006) / Error module (001-006)
        error_detail: エラー詳細 (001-013) / Error detail (001-013)
        extra_message: 追加メッセージ（デフォルト: ""）/ Additional message (default: "")
    """
    # エラーをログに記録
    log_error(
        log_path, error_type, error_main, error_category, error_module, error_detail, extra_message
    )
    
    # エラーを発生
    if not isinstance(error_detail, CustomError):
        raise CustomError(error_type, error_main, error_category, error_module, error_detail)


def error_handler(
    module_type: str,
    operation: str,
    execution_id: Optional[str] = None,
    error_type: str = "ERR",
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """エラーハンドリングを行うデコレータ。
    Error handling decorator.

    Args:
        module_type: モジュールタイプ / Module type
        operation: オペレーション / Operation
        execution_id: 実行ID（オプション）/ Execution ID (optional)
        error_type: エラータイプ（デフォルト: "ERR"）/ Error type (default: "ERR")

    Returns:
        デコレータ関数 / Decorator function
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            # エラーコードを取得
            error_codes = _get_error_codes(module_type, operation)
            
            # ログパスを取得
            log_path = _get_log_path(module_type, execution_id)
            
            # 関数名をログに出力
            logger.info(f"関数 {func.__name__} の実行を開始します")
            logger.info(f"Starting execution of function {func.__name__}")
            
            try:
                return func(*args, **kwargs)
            except CustomError:
                raise
            except Exception as e:
                # スタックトレースを取得
                stack = inspect.stack()
                caller = stack[1] if len(stack) > 1 else None
                file_name = caller.filename if caller else "unknown"
                line_no = caller.lineno if caller else 0
                
                # エラー情報を追加
                extra_info = f"File: {os.path.basename(file_name)}, Line: {line_no}, Error: {str(e)}"
                
                # エラーをログに記録し、例外を発生
                log_and_raise_error(
                    log_path,
                    error_type,
                    error_codes["main"],
                    error_codes["category"],
                    error_codes["module"],
                    error_codes["detail"],
                    extra_info,
                )
                # この行は実行されない（log_and_raise_errorで例外が発生する）が、
                # 型チェッカーを満足させるために必要
                # This line is never executed (exception is raised in log_and_raise_error),
                # but necessary to satisfy type checker
                raise
                
        return wrapper
    return decorator


@contextmanager
def error_context(
    module_type: str,
    operation: str,
    execution_id: Optional[str] = None,
    error_type: str = "ERR",
) -> None:
    """エラーハンドリングを行うコンテキストマネージャー。
    Error handling context manager.

    Args:
        module_type: モジュールタイプ / Module type
        operation: オペレーション / Operation
        execution_id: 実行ID（オプション）/ Execution ID (optional)
        error_type: エラータイプ（デフォルト: "ERR"）/ Error type (default: "ERR")
    """
    # エラーコードを取得
    error_codes = _get_error_codes(module_type, operation)
    
    # ログパスを取得
    log_path = _get_log_path(module_type, execution_id)
    
    try:
        yield
    except CustomError:
        raise
    except Exception as e:
        # スタックトレースを取得
        stack = inspect.stack()
        caller = stack[1] if len(stack) > 1 else None
        file_name = caller.filename if caller else "unknown"
        line_no = caller.lineno if caller else 0
        
        # エラー情報を追加
        extra_info = f"File: {os.path.basename(file_name)}, Line: {line_no}, Error: {str(e)}"
        
        # エラーをログに記録し、例外を発生
        log_and_raise_error(
            log_path,
            error_type,
            error_codes["main"],
            error_codes["category"],
            error_codes["module"],
            error_codes["detail"],
            extra_info,
        )


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
    """関数を安全に実行し、エラーを適切に処理する。
    Safely execute a function and handle errors appropriately.

    Args:
        func: 実行する関数 / Function to execute
        module_type: モジュールタイプ / Module type
        operation: オペレーション / Operation
        args: 関数の位置引数（デフォルト: None）/ Function positional arguments (default: None)
        kwargs: 関数のキーワード引数（デフォルト: None）/ Function keyword arguments (default: None)
        execution_id: 実行ID（オプション）/ Execution ID (optional)
        error_type: エラータイプ（デフォルト: "ERR"）/ Error type (default: "ERR")
        default: エラー発生時の戻り値（デフォルト: None）/ Return value on error (default: None)

    Returns:
        関数の戻り値またはエラー時のデフォルト値 / Function return value or default value on error
    """
    args = args or []
    kwargs = kwargs or {}
    
    # エラーコードを取得
    error_codes = _get_error_codes(module_type, operation)
    
    # ログパスを取得
    log_path = _get_log_path(module_type, execution_id)
    
    try:
        return func(*args, **kwargs)
    except CustomError:
        raise
    except Exception as e:
        # スタックトレースを取得
        stack = inspect.stack()
        caller = stack[1] if len(stack) > 1 else None
        file_name = caller.filename if caller else "unknown"
        line_no = caller.lineno if caller else 0
        
        # エラー情報を追加
        extra_info = f"File: {os.path.basename(file_name)}, Line: {line_no}, Error: {str(e)}"
        
        # エラーをログに記録
        log_error(
            log_path,
            error_type,
            error_codes["main"],
            error_codes["category"],
            error_codes["module"],
            error_codes["detail"],
            extra_info,
        )
        
        return default


# エラーハンドラークラス (互換性のため維持)
class ErrorHandler:
    """エラーハンドリングクラス。
    Error handling class.

    エラーログと例外を処理するためのスタティックメソッドを提供します。
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
        """エラーを発生させます。
        Raise an error.

        Args:
            error_type: エラータイプ (ERR/WRN) / Error type (ERR/WRN)
            main: エラーメイン (01-03) / Error main (01-03)
            category: エラーカテゴリ (1-2) / Error category (1-2)
            module: エラーモジュール (001-006) / Error module (001-006)
            detail: エラー詳細 (001-013) / Error detail (001-013)
            from_exception: 元の例外（デフォルト: None）/ Original exception (default: None)
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
        """エラーをログに記録します。
        Log an error.

        Args:
            error_type: エラータイプ (ERR/WRN) / Error type (ERR/WRN)
            main: エラーメイン (01-03) / Error main (01-03)
            category: エラーカテゴリ (1-2) / Error category (1-2)
            module: エラーモジュール (001-006) / Error module (001-006)
            detail: エラー詳細 (001-013) / Error detail (001-013)
            extra_message: 追加メッセージ（デフォルト: ""）/ Additional message (default: "")
        """
        error_message = _format_error_message(
            error_type, main, category, module, detail, extra_message
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
        """警告をログに記録します。
        Log a warning.

        Args:
            error_type: エラータイプ (ERR/WRN) / Error type (ERR/WRN)
            main: エラーメイン (01-03) / Error main (01-03)
            category: エラーカテゴリ (1-2) / Error category (1-2)
            module: エラーモジュール (001-006) / Error module (001-006)
            detail: エラー詳細 (001-013) / Error detail (001-013)
            extra_message: 追加メッセージ（デフォルト: ""）/ Additional message (default: "")
        """
        warning_message = _format_error_message(
            error_type, main, category, module, detail, extra_message
        )
        logger.warning(warning_message)

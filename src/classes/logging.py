"""ロギングモジュール。
Logging module.

このモジュールは、アプリケーション全体のロギングを一元的に管理します。
This module centrally manages logging for the entire application.

以下の機能を提供します:
Provides the following functionalities:
- 標準ロガーの設定 (Standard logger configuration)
- S3ロギング (S3 logging)
- エラーコード付きのログ記録 (Logging with error codes)
"""

import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Any

from classes.storage_operation import storage_operator
from common.error_map import (
    CATEGORY_MAP,
    ERROR_MESSAGE_MAP,
    MODULE_MAP,
    PROCESSING_MAP,
    TYPE_MAP,
)
from common.utils import from_utc_to_jst

# 既存のコード (Existing code)
logger = logging.getLogger("application")
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    logger.addHandler(handler)

formatter = logging.Formatter(
    fmt="[%(asctime)s,%(msecs)d]{%(filename)s:%(lineno)s}%(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
for handler in logger.handlers:
    handler.setFormatter(formatter)


# モジュール別設定 (Module specific settings)
MODULE_CONFIG = {
    "data_extraction": {
        "error_main": "01",
        "log_path_template": "logs/data_extraction_{execution_id}.log"
    },
    "summarization": {
        "error_main": "02",
        "log_path_template": "logs/summarization_{execution_id}.log"
    },
    "classification": {
        "error_main": "03",
        "log_path_template": "logs/classification_{execution_id}.log"
    }
}


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


class ApplicationLogger:
    """アプリケーションロギングクラス。
    Application logging class.

    標準出力とS3の両方へのロギングを提供します。
    Provides logging to both standard output and S3.
    """

    def __init__(self, module_type: Optional[str] = None, execution_id: Optional[str] = None):
        """ロガーを初期化する。
        Initialize logger.

        Args:
            module_type: モジュールタイプ（オプション）/ Module type (optional)
            execution_id: 実行ID（オプション）/ Execution ID (optional)
        """
        self.logger = logger
        self.module_type = module_type
        self.execution_id = execution_id
        
        # 実行IDがあれば記録
        if module_type and execution_id:
            self.set_execution_context(module_type, execution_id)
    
    def set_execution_context(self, module_type: str, execution_id: str) -> None:
        """実行コンテキストを設定する。
        Set execution context.

        Args:
            module_type: モジュールタイプ / Module type
            execution_id: 実行ID / Execution ID
        """
        self.module_type = module_type
        self.execution_id = execution_id
        self.info(f"{module_type} 処理を開始します。実行ID: {execution_id}")
        self.info(f"Starting {module_type} processing. Execution ID: {execution_id}")
    
    def _get_log_path(self) -> str:
        """現在のモジュールタイプと実行IDに基づいてログパスを取得する。
        Get log path based on current module type and execution ID.

        Returns:
            ログパス / Log path
        """
        if self.module_type:
            return _get_log_path(self.module_type, self.execution_id)
        return "logs/application.log"
    
    def log(self, level: int, message: str, log_to_s3: bool = True) -> None:
        """一般的なログメッセージを記録する。
        Log a general message.

        Args:
            level: ログレベル / Log level
            message: ログメッセージ / Log message
            log_to_s3: S3にログを記録するかどうか（デフォルト: True）/ Whether to log to S3 (default: True)
        """
        # コンソールに出力
        self.logger.log(level, message)
        
        # S3にも出力
        if log_to_s3:
            log_path = self._get_log_path()
            timestamp = from_utc_to_jst(datetime.now(timezone.utc).isoformat())
            log_message = f"[{timestamp}] {message}"
            
            # ディレクトリが存在することを確認
            directory = os.path.dirname(log_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            try:
                storage_operator.put_log_to_s3(log_message, log_path)
            except Exception as e:
                # S3へのログ保存に失敗した場合はコンソールに出力
                self.logger.error(f"S3へのログ保存に失敗しました: {str(e)}")
                self.logger.error(f"Failed to save log to S3: {str(e)}")
    
    def debug(self, message: str, log_to_s3: bool = True) -> None:
        """デバッグメッセージを記録する。
        Log a debug message.

        Args:
            message: ログメッセージ / Log message
            log_to_s3: S3にログを記録するかどうか（デフォルト: True）/ Whether to log to S3 (default: True)
        """
        self.log(logging.DEBUG, message, log_to_s3)
    
    def info(self, message: str, log_to_s3: bool = True) -> None:
        """情報メッセージを記録する。
        Log an info message.

        Args:
            message: ログメッセージ / Log message
            log_to_s3: S3にログを記録するかどうか（デフォルト: True）/ Whether to log to S3 (default: True)
        """
        self.log(logging.INFO, message, log_to_s3)
    
    def warning(self, message: str, log_to_s3: bool = True) -> None:
        """警告メッセージを記録する。
        Log a warning message.

        Args:
            message: ログメッセージ / Log message
            log_to_s3: S3にログを記録するかどうか（デフォルト: True）/ Whether to log to S3 (default: True)
        """
        self.log(logging.WARNING, message, log_to_s3)
    
    def error(self, message: str, log_to_s3: bool = True) -> None:
        """エラーメッセージを記録する。
        Log an error message.

        Args:
            message: ログメッセージ / Log message
            log_to_s3: S3にログを記録するかどうか（デフォルト: True）/ Whether to log to S3 (default: True)
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
        """コード化されたエラーメッセージを記録する。
        Log a coded error message.

        Args:
            error_type: エラータイプ (ERR/WRN) / Error type (ERR/WRN)
            error_main: エラーメイン (01-03) / Error main (01-03)
            error_category: エラーカテゴリ (1-2) / Error category (1-2)
            error_module: エラーモジュール (001-006) / Error module (001-006)
            error_detail: エラー詳細 (001-013) / Error detail (001-013)
            extra_message: 追加メッセージ（デフォルト: ""）/ Additional message (default: "")

        Returns:
            記録されたエラーメッセージ / Logged error message
        """
        error_message = _format_error_message(
            error_type, 
            error_main, 
            error_category, 
            error_module, 
            error_detail, 
            extra_message
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
        """コード化された警告メッセージを記録する。
        Log a coded warning message.

        Args:
            error_type: エラータイプ (ERR/WRN) / Error type (ERR/WRN)
            error_main: エラーメイン (01-03) / Error main (01-03)
            error_category: エラーカテゴリ (1-2) / Error category (1-2)
            error_module: エラーモジュール (001-006) / Error module (001-006)
            error_detail: エラー詳細 (001-013) / Error detail (001-013)
            extra_message: 追加メッセージ（デフォルト: ""）/ Additional message (default: "")

        Returns:
            記録された警告メッセージ / Logged warning message
        """
        warning_message = _format_error_message(
            error_type, 
            error_main, 
            error_category, 
            error_module, 
            error_detail, 
            extra_message
        )
        
        self.warning(warning_message)
        return warning_message
    
    def get_error_main_for_module(self) -> str:
        """現在のモジュールタイプに基づいてエラーメインコードを取得する。
        Get error main code based on current module type.

        Returns:
            エラーメインコード / Error main code
        """
        if self.module_type and self.module_type in MODULE_CONFIG:
            return MODULE_CONFIG[self.module_type]["error_main"]
        return "99"  # デフォルト / Default


# 下位互換性のための関数 (For backward compatibility)
def warning_logger(message: str) -> None:
    """警告メッセージをログに記録する補助関数。
    Helper function to log warning messages.

    Args:
        message: ログメッセージ / Log message
    """
    global has_warning
    has_warning = True
    logger.warning(message)


# シングルトンインスタンスの作成 (Create singleton instance)
application_logger = ApplicationLogger()

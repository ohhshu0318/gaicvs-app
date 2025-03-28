"""S3ユーティリティモジュール。
S3 Utility module.

このモジュールは、S3操作に関する共通ユーティリティを提供します。
This module provides common utilities for S3 operations.

- ファイル存在チェック (File existence check)
- ヘッダーチェック (Header check)
- endfile管理 (Endfile management)
"""

import os
from io import StringIO
from typing import Any, List, Optional, Tuple

import pandas as pd

from classes.error_handle import CustomError, error_context, log_and_raise_error
from classes.logging import s3_logger
from common.utils import from_utc_to_jst


class S3Utils:
    """S3操作に関する共通ユーティリティクラス。
    Common utility class for S3 operations.
    """

    def __init__(self, storage_operator: Any) -> None:
        """初期化。
        Initialization.

        Args:
            storage_operator: S3操作を行うオペレータ / Operator for S3 operations
        """
        self.storage_operator = storage_operator

    def check_file_exists(
        self,
        key: str,
        log_path: Optional[str] = None,
        error_type: str = "ERR",
        error_main: str = "01",
        error_category: str = "1",
        error_module: str = "001",
        error_detail: str = "001",
    ) -> bool:
        """S3上のファイルの存在チェックを行う。
        Check if a file exists on S3.

        Args:
            key: S3のオブジェクトキー / S3 object key
            log_path: ログファイルのパス / Log file path
            error_type: エラータイプ / Error type
            error_main: エラーメイン / Error main
            error_category: エラーカテゴリ / Error category
            error_module: エラーモジュール / Error module
            error_detail: エラー詳細 / Error detail

        Returns:
            ファイルが存在する場合はTrue / True if the file exists
        """
        try:
            file_exists = self.storage_operator.check_object_exists(key=key)
            s3_logger.info(f"ファイル存在チェック: {key}, 結果: {file_exists}")
            s3_logger.info(f"File existence check: {key}, result: {file_exists}")

            if not file_exists and log_path:
                log_and_raise_error(
                    log_path, error_type, error_main, error_category, error_module, error_detail
                )

            return file_exists
        except CustomError:
            raise
        except Exception as e:
            if log_path:
                log_and_raise_error(
                    log_path,
                    error_type,
                    error_main,
                    error_category,
                    error_module,
                    error_detail,
                    str(e),
                )
            return False

    def check_file_headers(
        self,
        key: str,
        expected_headers: List[str],
        log_path: Optional[str] = None,
        error_type: str = "ERR",
        error_main: str = "01",
        error_category: str = "1",
        error_module: str = "001",
        error_detail: str = "008",
    ) -> bool:
        """S3上のCSVファイルのヘッダーチェックを行う。
        Check headers of a CSV file on S3.

        Args:
            key: S3のオブジェクトキー / S3 object key
            expected_headers: 期待されるヘッダーリスト / List of expected headers
            log_path: ログファイルのパス / Log file path
            error_type: エラータイプ / Error type
            error_main: エラーメイン / Error main
            error_category: エラーカテゴリ / Error category
            error_module: エラーモジュール / Error module
            error_detail: エラー詳細 / Error detail

        Returns:
            ヘッダーが期待通りの場合はTrue / True if headers match expectations
        """
        try:
            csv_data = self.storage_operator.get_object(key=key)
            df = pd.read_csv(StringIO(csv_data))

            s3_logger.info(f"ファイルヘッダーチェック: {key}")
            s3_logger.info(f"File header check: {key}")
            s3_logger.info(f"実際のヘッダー: {df.columns.to_list()}")
            s3_logger.info(f"Actual headers: {df.columns.to_list()}")
            s3_logger.info(f"期待されるヘッダー: {expected_headers}")
            s3_logger.info(f"Expected headers: {expected_headers}")

            if not set(expected_headers).issubset(set(df.columns)):
                if log_path:
                    log_and_raise_error(
                        log_path,
                        error_type,
                        error_main,
                        error_category,
                        error_module,
                        error_detail,
                    )
                return False

            return True
        except CustomError:
            raise
        except Exception as e:
            if log_path:
                log_and_raise_error(
                    log_path,
                    error_type,
                    error_main,
                    error_category,
                    error_module,
                    error_detail,
                    str(e),
                )
            return False

    def get_last_run_date(
        self,
        endfile_prefix: str,
        log_path: Optional[str] = None,
        error_type: str = "ERR",
        error_main: str = "01",
        error_category: str = "1",
        error_module: str = "001",
        error_detail: str = "001",
    ) -> Tuple[str, str]:
        """前回実行日時の取得。
        Get the date of the last run.

        Args:
            endfile_prefix: endfileのプレフィックス / Prefix for endfile
            log_path: ログファイルのパス / Log file path
            error_type: エラータイプ / Error type
            error_main: エラーメイン / Error main
            error_category: エラーカテゴリ / Error category
            error_module: エラーモジュール / Error module
            error_detail: エラー詳細 / Error detail

        Returns:
            (前回実行日時, 前回endfileのパス) / (Last run date, path to last endfile)
        """
        with error_context(
            log_path, error_type, error_main, error_category, error_module, error_detail
        ):
            txt_files = list(
                filter(
                    lambda x: ".txt" in x,
                    self.storage_operator.list_objects(prefix=endfile_prefix),
                )
            )
            s3_logger.debug(f"検出されたendfile: {txt_files}")
            s3_logger.debug(f"Detected endfiles: {txt_files}")

            if not txt_files:
                if log_path:
                    log_and_raise_error(
                        log_path,
                        error_type,
                        error_main,
                        error_category,
                        error_module,
                        error_detail,
                    )
                raise ValueError("No endfile found")

            last_run_endfile = sorted(txt_files, reverse=True)[0]
            s3_logger.info(f"前回のendfileを取得しました: {last_run_endfile}")
            s3_logger.info(f"Retrieved last endfile: {last_run_endfile}")

            last_run_date = os.path.splitext(os.path.basename(last_run_endfile))[0]

            return last_run_date, last_run_endfile

    def manage_endfile(
        self,
        endfile_prefix: str,
        last_run_endfile: str,
        run_date: str,
        log_path: Optional[str] = None,
        error_type: str = "ERR",
        error_main: str = "01",
        error_category: str = "1",
        error_module: str = "001",
        error_detail: str = "002",
    ) -> str:
        """endfileの管理（バックアップと新規作成）。
        Manage endfile (backup and create new).

        Args:
            endfile_prefix: endfileのプレフィックス / Prefix for endfile
            last_run_endfile: 前回のendfileパス / Path to last endfile
            run_date: 実行日 / Run date
            log_path: ログファイルのパス / Log file path
            error_type: エラータイプ / Error type
            error_main: エラーメイン / Error main
            error_category: エラーカテゴリ / Error category
            error_module: エラーモジュール / Error module
            error_detail: エラー詳細 / Error detail

        Returns:
            新規作成したendfileのパス / Path to newly created endfile
        """
        with error_context(
            log_path, error_type, error_main, error_category, error_module, error_detail
        ):
            # 前回のendfileをバックアップ / Backup the previous endfile
            destination_key = str(last_run_endfile).replace(
                endfile_prefix, os.path.join(endfile_prefix, "backup/")
            )

            s3_logger.info(f"endfileを移動: {last_run_endfile} -> {destination_key}")
            s3_logger.info(f"Moving endfile: {last_run_endfile} -> {destination_key}")
            self.storage_operator.move_object(
                destination_key=destination_key, source_key=last_run_endfile
            )

            # 新しいendfileを作成 / Create new endfile
            endfile_key = os.path.join(
                endfile_prefix, f"{from_utc_to_jst(run_date).isoformat()}.txt"
            )

            s3_logger.info(f"新規endfileを作成: {endfile_key}")
            s3_logger.info(f"Creating new endfile: {endfile_key}")
            self.storage_operator.put_object(key=endfile_key, body="")

            return endfile_key

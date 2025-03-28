"""データ処理ユーティリティモジュール。
Data Processing Utility module.

このモジュールは、データ処理に関する共通ユーティリティを提供します。
This module provides common utilities for data processing.

- テキストデータのクリーニング (Text data cleaning)
- 正規表現を使用した抽出 (Extraction using regular expressions)
- テキストカラムの結合 (Text column combination)
- 固定値カラムの追加 (Fixed value column addition)
- 分類結果の抽出 (Classification result extraction)
"""

import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from classes.logging import s3_logger


class DataUtils:
    """データ処理に関する共通ユーティリティクラス。
    Common utility class for data processing.
    """

    @staticmethod
    def clean_text_data(
        df: pd.DataFrame, columns: List[str], null_replacement: str = "missing_value"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """テキストデータのクリーニングと欠損値の削除。
        Clean text data and remove rows with missing values.

        Args:
            df: 処理対象のDataFrame / DataFrame to process
            columns: クリーニング対象のカラム / Columns to clean
            null_replacement: 欠損値の代替テキスト / Replacement text for missing values

        Returns:
            (クリーニング後のDataFrame, 欠損値を含む行のDataFrame) / 
            (DataFrame after cleaning, DataFrame with rows containing missing values)
        """
        # 元のDataFrameをコピー / Copy the original DataFrame
        cleaned_df = df.copy()

        # 各カラムをクリーニング / Clean each column
        for column in columns:
            if column in cleaned_df.columns:
                cleaned_df[column] = (
                    cleaned_df[column]
                    .replace([None, "NaN", "null", ""], null_replacement)
                    .str.strip()
                )
                cleaned_df[column] = cleaned_df[column].replace("", null_replacement)

        # 欠損値を含む行を特定 / Identify rows with missing values
        missing_rows = pd.DataFrame()
        for column in columns:
            if column in cleaned_df.columns:
                missing_column_rows = cleaned_df[cleaned_df[column] == null_replacement]
                if not missing_column_rows.empty:
                    missing_rows = pd.concat([missing_rows, missing_column_rows])

        # 欠損値を含む行を削除 / Remove rows with missing values
        for column in columns:
            if column in cleaned_df.columns:
                cleaned_df = cleaned_df[cleaned_df[column] != null_replacement]

        s3_logger.info(
            f"テキストクリーニング: {len(df)} 行から {len(cleaned_df)} 行になりました。"
            f"{len(missing_rows)} 行は欠損値を含むため削除されました。"
        )
        s3_logger.info(
            f"Text cleaning: {len(df)} rows reduced to {len(cleaned_df)} rows. "
            f"{len(missing_rows)} rows were removed due to missing values."
        )

        return cleaned_df, missing_rows

    @staticmethod
    def extract_with_regex(
        text: str, pattern: str, group: int = 1, default: str = ""
    ) -> str:
        """正規表現を使用してテキストから情報を抽出。
        Extract information from text using regular expressions.

        Args:
            text: 処理対象のテキスト / Text to process
            pattern: 正規表現パターン / Regular expression pattern
            group: 抽出するグループ番号 (マッチ全体の場合は0) / Group number to extract (0 for entire match)
            default: マッチしなかった場合のデフォルト値 / Default value if no match

        Returns:
            抽出されたテキスト / Extracted text
        """
        match = re.search(pattern, str(text), re.DOTALL)
        if match:
            if group == 0:  # マッチ全体を返す / Return entire match
                return match.group(0).strip("\n").strip()
            else:  # 指定されたグループを返す / Return specified group
                try:
                    return match.group(group).strip("\n").strip()
                except IndexError:
                    return default
        return default

    @staticmethod
    def combine_text_columns(
        df: pd.DataFrame, columns: List[str], separator: str = " "
    ) -> pd.Series:
        """複数のテキストカラムを結合。
        Combine multiple text columns.

        Args:
            df: 処理対象のDataFrame / DataFrame to process
            columns: 結合対象のカラム / Columns to combine
            separator: 結合に使用する区切り文字 / Separator to use for combination

        Returns:
            結合されたテキストのSeries / Series of combined text
        """
        result = df[columns].apply(
            lambda x: separator.join(x.dropna().astype(str)), axis=1
        )
        s3_logger.info(f"テキストカラム結合: {columns} を {separator} で結合しました")
        s3_logger.info(f"Text column combination: Combined {columns} with separator '{separator}'")
        return result

    @staticmethod
    def add_fixed_columns(df: pd.DataFrame, fixed_values: Dict[str, Any]) -> pd.DataFrame:
        """固定値のカラムを追加。
        Add columns with fixed values.

        Args:
            df: 処理対象のDataFrame / DataFrame to process
            fixed_values: カラム名と固定値のディクショナリ / Dictionary of column names and fixed values

        Returns:
            カラムが追加されたDataFrame / DataFrame with added columns
        """
        result_df = df.copy()
        for column, value in fixed_values.items():
            result_df[column] = value

        s3_logger.info(f"固定値カラム追加: {list(fixed_values.keys())} を追加しました")
        s3_logger.info(f"Fixed value columns added: {list(fixed_values.keys())}")
        return result_df

    @staticmethod
    def extract_classification_results(response_text: str) -> Dict[str, str]:
        """APIレスポンスから分類結果を抽出。
        Extract classification results from API response.

        Args:
            response_text: APIレスポンステキスト / API response text

        Returns:
            分類結果の辞書 / Dictionary of classification results
        """
        classification = DataUtils.extract_with_regex(
            response_text, r"(?<=区分:)(.*?)(?=大分類:)", 1, ""
        )

        category = DataUtils.extract_with_regex(
            response_text, r"(?<=大分類:)(.*?)(?=中分類:)", 1, ""
        )

        subcategory = DataUtils.extract_with_regex(
            response_text, r"(?<=中分類:)(.*?)(?=小分類:)", 1, ""
        )

        subsubcategory = DataUtils.extract_with_regex(
            response_text, r"小分類:(.*?)(?:\n|$)", 1, ""
        )

        return {
            "classification": classification,
            "category": category,
            "subcategory": subcategory,
            "subsubcategory": subsubcategory,
        }

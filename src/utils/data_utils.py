"""
Data Processing Utility module.

This module provides common utilities for data processing tasks.

Contents:
---------
Classes:
    - DataUtils               : Common utility class for data processing
    - clean_text_data         : Clean text data and handle missing values
    - extract_with_regex      : Extract information using regular expressions
    - combine_text_columns    : Combine multiple text columns
    - add_fixed_columns       : Add columns with fixed values
    - parse_classification    : Extract classification from responses
"""

import re
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from classes.logging_handler import application_logger


class DataUtils:
    """
    Common utility class for data processing.
    """

    @staticmethod
    def clean_text_data(
        df: pd.DataFrame,
        columns: List[str],
        null_replacement: str = "missing_value",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean text data and remove rows with missing values.

        Args:
            df                : DataFrame to process
            columns           : Columns to clean
            null_replacement  : Replacement text for missing values
        Returns:
            Tuple containing:
                - cleaned_df  : DataFrame with valid rows
                - missing_df  : DataFrame with rows containing missing values
        """

        # Copy the original DataFrame
        cleaned_df = df.copy()

        # Clean each column
        for column in columns:
            if column in cleaned_df.columns:
                # Convert all types to string
                cleaned_df[column] = cleaned_df[column].astype(str)

                cleaned_df[column] = (
                    cleaned_df[column]
                    .replace(
                        [None, "NaN", "null", "", "None"], null_replacement
                    )
                    .str.strip()
                )
                cleaned_df[column] = cleaned_df[column].replace(
                    "", null_replacement
                )

        # Identify rows with missing values
        missing_rows = pd.DataFrame()
        for column in columns:
            if column in cleaned_df.columns:
                missing_column_rows = cleaned_df[
                    cleaned_df[column] == null_replacement
                ]
                if not missing_column_rows.empty:
                    missing_rows = pd.concat(
                        [missing_rows, missing_column_rows]
                    )

        # Remove rows with missing values
        for column in columns:
            if column in cleaned_df.columns:
                cleaned_df = cleaned_df[cleaned_df[column] != null_replacement]

        application_logger.info(
            f"Text cleaning: {len(df)} rows reduced to {len(cleaned_df)} rows."
            f"{len(missing_rows)} rows were removed due to missing values."
        )

        return cleaned_df, missing_rows

    @staticmethod
    def extract_with_regex(
        text: Union[str, int, float],
        pattern: str,
        group: int = 1,
        default: str = "",
    ) -> str:
        """
        Extract information from text using regular expressions.

        Args:
            text              : Text to process
            pattern           : Regular expression pattern
            group             : Group number to extract (0 for entire match)
            default           : Default value if no match is found
        Returns:
            str               : Extracted text
        """

        # Convert input to string
        text_str = str(text)

        match = re.search(pattern, text_str, re.DOTALL)
        if match:
            if group == 0:  # Return entire match
                return match.group(0).strip("\n").strip()
            else:  # Return specified group
                try:
                    return match.group(group).strip("\n").strip()
                except IndexError:
                    return default
        return default

    @staticmethod
    def combine_text_columns(
        df: pd.DataFrame, columns: List[str], separator: str = " "
    ) -> pd.Series:
        """
        Combine multiple text columns.

        Args:
            df                : DataFrame to process
            columns           : Columns to combine
            separator         : Separator to use for combination
        Returns:
            pd.Series         : Series with combined text
        """

        # Replace NaN with empty string and convert to string
        combined = df[columns].fillna("").astype(str)

        result = combined.apply(
            lambda x: separator.join(filter(bool, x)), axis=1
        )

        application_logger.info(
            f"Text column combination: "
            f"Combined {columns} with separator '{separator}'"
        )

        return result

    @staticmethod
    def add_fixed_columns(
        df: pd.DataFrame, fixed_values: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Add columns with fixed values.

        Args:
            df                : DataFrame to process
            fixed_values      : Dictionary of column names and fixed values
        Returns:
            pd.DataFrame      : DataFrame with added columns
        """

        result_df = df.copy()
        for column, value in fixed_values.items():
            # Create empty column for empty DataFrame
            if len(result_df) == 0:
                result_df[column] = pd.Series(
                    dtype=type(value) if value is not None else object
                )
            else:
                result_df[column] = value

        application_logger.info(
            f"Fixed value columns added: {list(fixed_values.keys())}"
        )
        return result_df

    @staticmethod
    def parse_classification(response_text: str) -> Dict[str, str]:
        """
        Extract classification results from API response.

        Args:
            response_text     : API response text
        Returns:
            Dict[str, str]    : Dictionary of classification results
        """

        # Convert response text to string
        response_str = str(response_text)

        def safe_extract(pattern: str) -> str:
            """Helper function for safe extraction"""
            match = re.search(pattern, response_str, re.DOTALL)
            return match.group(1).strip() if match else ""

        # Return extracted classification fields
        return {
            "classification": safe_extract(r"区分:\s*(.*?)(?=\n|$)"),
            "category": safe_extract(r"大分類:\s*(.*?)(?=\n|$)"),
            "subcategory": safe_extract(r"中分類:\s*(.*?)(?=\n|$)"),
            "subsubcategory": safe_extract(r"小分類:\s*(.*?)(?=\n|$)"),
        }

"""プロンプト生成ユーティリティモジュール。
Prompt Generation Utility module.

このモジュールは、プロンプト生成に関する共通ユーティリティを提供します。
This module provides common utilities for prompt generation.

- 分類タスク用のプロンプト生成 (Prompt generation for classification tasks)
- 要約タスク用のプロンプト生成 (Prompt generation for summarization tasks)
- APIレスポンスの結果抽出 (Result extraction from API responses)
"""

from typing import Any, Dict, List

import pandas as pd

from classes.logging import s3_logger


class PromptUtils:
    """プロンプト生成に関する共通ユーティリティクラス。
    Common utility class for prompt generation.
    """

    @staticmethod
    def generate_classification_prompt(
        prompt_template: str,
        similarity_results: Dict[int, Dict[str, Any]],
        references_metadata: Dict[str, Dict[str, str]],
        shot_number: int = 5,
    ) -> Dict[int, str]:
        """分類タスク用のプロンプトを生成。
        Generate prompts for classification tasks.

        Args:
            prompt_template: プロンプトのテンプレート / Prompt template
            similarity_results: 類似度検索結果 / Similarity search results
            references_metadata: 参照データのメタデータ / Metadata for reference data
            shot_number: プロンプトに含める例の数 / Number of examples to include in the prompt

        Returns:
            行番号をキーとするプロンプトのディクショナリ / Dictionary of prompts keyed by row number
        """
        prompts = {}
        for rn, data in similarity_results.items():
            references = data.get("results")
            if not references:
                prompts[rn] = ""
                data["candidate"] = {
                    "classification": "",
                    "category": "",
                    "subcategory": "",
                    "subsubcategory": "",
                }
                s3_logger.warning(f"行番号 {rn} の類似テキストが見つかりませんでした")
                s3_logger.warning(f"No similar texts found for row number {rn}")
                continue

            references = sorted(
                references, key=lambda x: x["score"], reverse=True
            )[:shot_number]
            reference_text = ""
            classification_candidates = []
            category_candidates = []
            subcategory_candidates = []
            subsubcategory_candidates = []

            for ref in references:
                classification = references_metadata.get(ref["voice_no"], {}).get(
                    "classification", ""
                )
                category = references_metadata.get(ref["voice_no"], {}).get("category", "")
                subcategory = references_metadata.get(ref["voice_no"], {}).get(
                    "subcategory", ""
                )
                subsubcategory = references_metadata.get(ref["voice_no"], {}).get(
                    "subsubcategory", ""
                )
                cleaned_text = (
                    str(ref["text"])
                    .replace("\n", " ")
                    .replace("\r\n", " ")
                    .replace("\r", " ")
                )
                reference_text += (
                    "文章: {}\n区分: {}\n大分類: {}\n中分類: {}\n小分類: {}\n\n".format(
                        cleaned_text,
                        classification,
                        category,
                        subcategory,
                        subsubcategory,
                    )
                )
                classification_candidates.append(classification)
                category_candidates.append(category)
                subcategory_candidates.append(subcategory)
                subsubcategory_candidates.append(subsubcategory)

            prompt = prompt_template.format(
                input_text=(
                    str(data["text"])
                    .replace("\n", " ")
                    .replace("\r\n", " ")
                    .replace("\r", " ")
                ),
                reference_text=reference_text,
                classification=",".join(set(filter(None, classification_candidates))),
                category=",".join(set(filter(None, category_candidates))),
                subcategory=",".join(set(filter(None, subcategory_candidates))),
                subsubcategory=",".join(set(filter(None, subsubcategory_candidates))),
            )
            data["candidate"] = {
                "classification": classification_candidates,
                "category": category_candidates,
                "subcategory": subcategory_candidates,
                "subsubcategory": subsubcategory_candidates,
            }
            prompts[rn] = prompt

        s3_logger.info(f"{len(prompts)} 件のプロンプトを生成しました")
        s3_logger.info(f"Generated {len(prompts)} prompts")
        return prompts

    @staticmethod
    def generate_summarization_prompt(
        df: pd.DataFrame, prompt_template: str, text_column: str
    ) -> pd.DataFrame:
        """要約タスク用のプロンプトを生成。
        Generate prompts for summarization tasks.

        Args:
            df: データフレーム / DataFrame
            prompt_template: プロンプトのテンプレート / Prompt template
            text_column: 要約対象のテキストカラム / Text column to summarize

        Returns:
            プロンプトカラムが追加されたデータフレーム / DataFrame with added prompt column
        """
        df_copy = df.copy()
        df_copy["genai_summary_prompt"] = df_copy.apply(
            lambda row: prompt_template.replace("{text}", row[text_column]),
            axis=1,
        )

        s3_logger.info(f"{len(df_copy)} 件の要約プロンプトを生成しました")
        s3_logger.info(f"Generated {len(df_copy)} summarization prompts")
        return df_copy

    @staticmethod
    def extract_api_responses(
        responses: List[Dict[str, Any]],
        error_code_success: str = "",
        error_code_timeout: str = "",
        error_code_failed: str = "",
    ) -> List[Dict[str, Any]]:
        """APIレスポンスから結果とエラー情報を抽出。
        Extract results and error information from API responses.

        Args:
            responses: APIレスポンスのリスト / List of API responses
            error_code_success: 成功時のエラーコード / Error code for success
            error_code_timeout: タイムアウト時のエラーコード / Error code for timeout
            error_code_failed: 失敗時のエラーコード / Error code for failure

        Returns:
            抽出された結果とエラー情報のリスト / List of extracted results and error information
        """
        results = []

        for response in responses:
            status_code = response.get("status_code")

            if status_code == 200:
                results.append(
                    {
                        "data": response.get("data", ""),
                        "error_message": error_code_success,
                    }
                )
            elif status_code == 500:
                results.append(
                    {
                        "data": "",
                        "error_message": error_code_timeout,
                    }
                )
            else:
                results.append(
                    {
                        "data": "",
                        "error_message": error_code_failed,
                    }
                )

        s3_logger.info(f"{len(results)} 件のAPIレスポンスを処理しました")
        s3_logger.info(f"Processed {len(results)} API responses")
        return results

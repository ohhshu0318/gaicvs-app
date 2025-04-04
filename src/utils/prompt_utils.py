
"""
Prompt Generation Utility module.

This module provides common utilities for prompt generation for AI models.

Contents:
---------
Classes:
    - PromptUtils             : Common utility class for prompt generation
    - create_class_prompt     : Generate prompts for classification tasks
    - create_summary_prompt   : Generate prompts for summarization tasks
    - extract_api_responses   : Extract results from API responses
"""

from typing import Any, Dict, List

import pandas as pd
from classes.logging_handler import application_logger


class PromptUtils:
    """
    Common utility class for prompt generation.
    """

    @staticmethod
    def create_class_prompt(
        prompt_template: str,
        sim_results: Dict[int, Dict[str, Any]],
        ref_metadata: Dict[str, Dict[str, str]],
        shot_number: int = 5,
    ) -> Dict[int, str]:
        """
        Generate prompts for classification tasks.

        Args:
            prompt_template   : Prompt template
            sim_results       : Similarity search results
            ref_metadata      : Metadata for reference data
            shot_number       : Number of examples to include in the prompt
        Returns:
            Dict[int, str]    : Dictionary of prompts keyed by row number
        """

        prompts = {}
        for rn, data in sim_results.items():
            references = data.get("results")
            if not references:
                prompts[rn] = ""
                data["candidate"] = {
                    "classification": "",
                    "category": "",
                    "subcategory": "",
                    "subsubcategory": "",
                }
                application_logger.warning(
                    f"No similar texts found for row number {rn}"
                )
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
                classification = ref_metadata.get(ref["voice_no"], {}).get(
                    "classification", ""
                )
                category = ref_metadata.get(ref["voice_no"], {}).get(
                    "category", ""
                )
                subcategory = ref_metadata.get(ref["voice_no"], {}).get(
                    "subcategory", ""
                )
                subsubcategory = ref_metadata.get(ref["voice_no"], {}).get(
                    "subsubcategory", ""
                )
                cleaned_text = (
                    str(ref["text"])
                    .replace("\n", " ")
                    .replace("\r\n", " ")
                    .replace("\r", " ")
                )
                reference_text += (
                    "文章: {}\n"
                    "区分: {}\n"
                    "大分類: {}\n"
                    "中分類: {}\n"
                    "小分類: {}\n\n"
                ).format(
                    cleaned_text,
                    classification,
                    category,
                    subcategory,
                    subsubcategory,
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
                classification=",".join(
                    set(filter(None, classification_candidates))
                ),
                category=",".join(set(filter(None, category_candidates))),
                subcategory=",".join(
                    set(filter(None, subcategory_candidates))
                ),
                subsubcategory=",".join(
                    set(filter(None, subsubcategory_candidates))
                ),
            )
            data["candidate"] = {
                "classification": classification_candidates,
                "category": category_candidates,
                "subcategory": subcategory_candidates,
                "subsubcategory": subsubcategory_candidates,
            }
            prompts[rn] = prompt

        application_logger.info(f"Generated {len(prompts)} prompts")
        return prompts

    @staticmethod
    def create_summary_prompt(
        df: pd.DataFrame, prompt_template: str, text_column: str
    ) -> pd.DataFrame:
        """
        Generate prompts for summarization tasks.

        Args:
            df                : DataFrame to process
            prompt_template   : Prompt template
            text_column       : Text column to summarize
        Returns:
            pd.DataFrame      : DataFrame with added prompt column
        """

        df_copy = df.copy()
        df_copy["genai_summary_prompt"] = df_copy.apply(
            lambda row: prompt_template.replace("{text}", row[text_column]),
            axis=1,
        )

        application_logger.info(
            f"Generated {len(df_copy)} summarization prompts"
        )
        return df_copy

    @staticmethod
    def extract_api_responses(
        responses: List[Dict[str, Any]],
        error_code_success: str = "",
        error_code_timeout: str = "",
        error_code_failed: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Extract results and error information from API responses.

        Args:
            responses         : List of API responses
            error_code_success: Error code for success
            error_code_timeout: Error code for timeout
            error_code_failed : Error code for failure
        Returns:
            List              : Processed API responses with data and error
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

        application_logger.info(f"Processed {len(results)} API responses")
        return results

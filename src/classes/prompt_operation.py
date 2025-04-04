import logging

from classes.storage_operation import storage_operator_prompt
from common.const import PROMPTS_BUCKET_NAME


class s3_prompt_manager:
    def __init__(self, bucket_name):
        self.storage = storage_operator_prompt
        self.bucket_name = bucket_name

    def get_prompt(self, prompt_key):
        try:
            response = self.storage.get_object(prompt_key)
            logging.info(f"プロンプト取得した:{self.bucket_name}{prompt_key}")
            return response
        except Exception as e:
            logging.error(f"プロンプト取得失敗:{self.bucket_name}{prompt_key}{str(e)}")
            raise RuntimeError(f"プロンプト取得失敗 '{prompt_key}': {str(e)}") from e


s3_prompt = s3_prompt_manager(PROMPTS_BUCKET_NAME)

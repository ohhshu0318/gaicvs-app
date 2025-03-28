from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

from classes.logging import logger


class BedrockOperation:
    def __init__(
        self,
    ) -> None:
        self.s3_client = self.get_bedrock()

    def get_bedrock(self) -> boto3.client:
        return boto3.client(
            "bedrock-agent-runtime", region_name="ap-northeast-1"
        )

    def retrieve(
        self, knowledge_base_id: str, data_source_id: str, text: str
    ) -> list[dict]:
        try:
            results = []
            response = self.s3_client.retrieve(
                knowledgeBaseId=knowledge_base_id,
                retrievalQuery={"text": text},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {
                        "numberOfResults": 5,
                        "overrideSearchType": "SEMANTIC",
                        "filter": {
                            "equals": {
                                "key": "x-amz-bedrock-kb-data-source-id",
                                "value": data_source_id,
                            }
                        },
                    }
                },
            )
            results.extend(response["retrievalResults"])

            while response.get("nextToken"):
                response = self.s3_client.retrieve(
                    knowledgeBaseId=knowledge_base_id,
                    nextToken=response.get("nextToken"),
                    retrievalQuery={"text": text},
                    retrievalConfiguration={
                        "vectorSearchConfiguration": {
                            "numberOfResults": 5,
                            "overrideSearchType": "SEMANTIC",
                            "filter": {
                                "equals": {
                                    "key": "x-amz-bedrock-kb-data-source-id",
                                    "value": data_source_id,
                                }
                            },
                        }
                    },
                )
                results.extend(response["retrievalResults"])
            return results
        except Exception:
            logger.exception(
                f"Bedrockから類似文章を取得した際に、エラーが発生した:\n  {text}"
            )
            return []

    def batch_retrieve(
        self, knowledge_base_id: str, data_source_id: str, texts: list[str]
    ) -> list:
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    self.retrieve, knowledge_base_id, data_source_id, text
                )
                for text in texts
            ]

            for future in as_completed(futures):
                results.append(future.result())

        return results


bedrock_operator = BedrockOperation()

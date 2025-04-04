import json
from logging import getLogger

import boto3

# Logging configuration
logger = getLogger(__name__)


class SecretsOperation():
    def __init__(self, secret_id) -> None:
        self.secret_id = secret_id
        logger.debug(f"Secretsoperation input value:{secret_id}")
        self.sm_client = boto3.client("secretsmanager")
        try:
            self.secrets = self.sm_client.get_secret_value(
                SecretId=secret_id
                )["SecretString"]
            logger.debug(f"self.secrets:{self.secrets}")
        except Exception as e:
            raise RuntimeError(
                f"Secrets Managerからシークレットの取得に失敗しました: {str(e)}"
            )

    def get(self, key, default=None):
        deserialized_secrets: dict = json.loads(self.secrets)
        return deserialized_secrets.get(key, default)

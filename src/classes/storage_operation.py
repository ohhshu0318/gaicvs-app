import csv
from io import StringIO
from logging import getLogger

import boto3
import botocore
import botocore.exceptions
import pandas as pd
from common.const import PROMPTS_BUCKET_NAME, RESULTS_BUCKET_NAME

# Logging configuration
logger = getLogger(__name__)


class StorageOperation:
    def __init__(self, bucket) -> None:
        self.bucket = bucket
        self.s3_client = self.get_s3_client()

    def get_s3_client(self):
        return boto3.client('s3')

    def get_object(self, key):
        return self.s3_client.get_object(Bucket=self.bucket,
                                         Key=key)['Body'].read().decode('utf8')

    def list_objects(self, prefix: str, delimiter: str = '/') -> list:
        results = []
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket,
            Delimiter=delimiter,
            Prefix=prefix
        )
        if 'Contents' in response:
            results.extend([content["Key"]
                            for content in response["Contents"]])

        while response.get('IsTruncated'):
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Delimiter=delimiter,
                Prefix=prefix,
                ContinuationToken=response['NextContinuationToken']
            )
            if 'Contents' in response:
                results.extend([content["Key"]
                                for content in response["Contents"]])

        return results

    def put_object(self, key, body: bytes):
        return self.s3_client.put_object(Bucket=self.bucket,
                                         Key=key, Body=body)

    def put_df_to_s3(self, df: pd.DataFrame, key) -> None:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False, quoting=csv.QUOTE_ALL,
                  encoding='utf8', lineterminator='\n')
        csv_buffer.seek(0)
        self.put_object(key=key, body=csv_buffer.getvalue().encode('utf-8'))

    def put_log_to_s3(self, log_message: str, key: str) -> None:
        log_buffer = StringIO()
        log_buffer.write(log_message)
        log_buffer.seek(0)
        self.put_object(key=key, body=log_buffer.getvalue().encode('utf-8'))

    def delete_objects(self, keys: list, quiet: bool = False) -> None:
        delete_keys = [{'Key': key} for key in keys]
        self.s3_client.delete_objects(Bucket=self.bucket,
                                      Delete={'Objects': delete_keys,
                                              'Quiet': quiet})

    def move_object(self, destination_key, source_key):
        logger.info(f'オブジェクトを移動する。\n From: {source_key}\n To: {destination_key}')
        self.s3_client.copy_object(Bucket=self.bucket, Key=destination_key,
                                   CopySource={'Bucket': self.bucket, 'Key': source_key})
        self.s3_client.delete_object(Bucket=self.bucket, Key=source_key)

    def check_object_exists(self, key: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise e


storage_operator = StorageOperation(RESULTS_BUCKET_NAME)
storage_operator_prompt = StorageOperation(PROMPTS_BUCKET_NAME)

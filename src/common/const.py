import os

# S3
RESULTS_BUCKET_NAME = os.environ.get("RESULTS_BUCKET_NAME")
PROMPTS_BUCKET_NAME = os.environ.get("PROMPTS_BUCKET_NAME")
EMBEDDING_BUCKET_NAME = os.environ.get("EMBEDDING_BUCKET_NAME")
S3_INPUT_PATH = os.environ.get("S3_INPUT_PATH")
S3_LOG_PATH = os.environ.get("S3_LOG_PATH")
S3_ENDFILE_PATH = os.environ.get("S3_ENDFILE_PATH")
S3_SUMMARIZATION_RESULTS_PATH = os.environ.get("S3_SUMMARIZATION_PATH")
S3_CLASSIFICATION_RESULTS_PATH = os.environ.get("S3_CLASSIFICATION_PATH")
SUMMARIZATION_PROMPT_PATH = os.environ.get("SUMMARIZATION_PROMPT_PATH")
CLASSIFICATION_PROMPT_PATH = os.environ.get("CLASSIFICATION_PROMPT_PATH")

# CVS DB config
CVS_DB_HOST = os.environ.get("CVS_DB_HOST")
CVS_DB_PORT = os.environ.get("CVS_DB_PORT", "5432")
CVS_DB_NAME = os.environ.get("CVS_DB_NAME")

# Bedrock
BEDROCK_KNOWLEDGE_BASE_ID = os.environ.get("BEDROCK_KNOWLEDGE_BASE_ID")
BEDROCK_DATA_SOURCE_ID = os.environ.get("BEDROCK_DATA_SOURCE_ID")

# Florence
FLORENCE_URL = os.environ.get("FLORENCE_URL")
ACCGROUP = os.environ.get("ACCGROUP")
COSTCENTER = os.environ.get("COSTCENTER")

# Secrets Manager
CVS_DB_USER = os.environ.get("CVS_DB_USER")
CVS_DB_PASSWORD = os.environ.get("CVS_DB_PASSWORD")
CMDBID = os.environ.get("CMDBID")
APITOKEN = os.environ.get("APITOKEN")

RETRY_NUM = 1

# Output file header
SUMMARIZATION_RESULTS_CSV_HEADER = [
    "kjoinfo_tsuban",
    "genai_summary_prompt",
    "genai_summary_text",
    "genai_summary_run4md",
    "error_message",
]

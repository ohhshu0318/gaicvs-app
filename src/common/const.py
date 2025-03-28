import os

# S3
RESULTS_BUCKET_NAME = os.environ.get("RESULTS_BUCKET_NAME")
PROMPTS_BUCKET_NAME = os.environ.get("PROMPTS_BUCKET_NAME")
EMBEDDING_BUCKET_NAME = os.environ.get("EMBEDDING_BUCKET_NAME")
SUMMARIZATION_PROMPT_PATH = "summarization/summarization_prompt.txt"
CLASSIFICATION_PROMPT_PATH = "classification/classification_prompt.txt"
S3_INPUT_PATH = "landing/type=@type/dt=@run_date/@execution_id.csv"
S3_SUMMARIZATION_RESULTS_PATH = "summarization/dt=@run_date/@execution_id.csv"
S3_CLASSIFICATION_RESULTS_PATH = (
    "classification/dt=@run_date/@execution_id.csv"
)
S3_LOG_PATH = "logs/type=@type/dt=@run_date/@execution_id.log"
S3_ENDFILE_PATH = "endfile/type=@type/"

# Secrets Manager
CVS_DB_USER = os.environ.get("CVS_DB_USER")
CVS_DB_PASSWORD = os.environ.get("CVS_DB_PASSWORD")
CMDBID = os.environ.get("CMDBID")
APITOKEN = os.environ.get("APITOKEN")

# CVS_SECRETS_ID = os.environ.get("CVS_SECRETS_ID")
# FLORENCE_SECRETS_ID = os.environ.get("FLORENCE_SECRETS_ID")

# CVS DB config
CVS_DB_NAME = os.environ.get("CVS_DB_NAME")
CVS_DB_HOST = os.environ.get("CVS_DB_HOST")
CVS_DB_PORT = os.environ.get("CVS_DB_PORT", "5432")

# Florence
FLORENCE_URL = os.environ.get("FLORENCE_URL")
ACCGROUP = os.environ.get("ACCGROUP")
COSTCENTER = os.environ.get("COSTCENTER")

# Bedrock
BEDROCK_KNOWLEDGE_BASE_ID = os.environ.get("BEDROCK_KNOWLEDGE_BASE_ID")
BEDROCK_DATA_SOURCE_ID = os.environ.get("BEDROCK_DATA_SOURCE_ID")

RETRY_NUM = 1

# Output file header
SUMMARIZATION_RESULST_CSV_HEADER = [
    "kjoinfo_tsuban",
    "genai_summary_prompt",
    "genai_summary_text",
    "genai_summary_run4md",
]

"""分類タスク実行モジュール。
Classification task execution module.

このモジュールは、生命保険会社の苦情要望データの分類処理を行います。
This module performs classification processing of life insurance company complaint data.

処理の流れ (Processing flow):
1. 入力CSVの取得と検証 (Get and validate input CSV)
2. Bedrockによる類似テキスト検索 (Search similar texts using Bedrock)
3. DBからの分類情報取得 (Get classification information from DB)
4. プロンプト生成とFlorence APIへのリクエスト (Generate prompt and request to Florence API)
5. 分類結果の抽出と検証 (Extract and validate classification results)
6. DB書き込みとS3への結果保存 (Write to DB and save results to S3)
"""

import argparse
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Set, Tuple, Optional

import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.exc import OperationalError

from classes.bedrock_operation import bedrock_operator
from classes.db_operation import db_operator
from classes.error_handle import CustomError, error_context, error_handler, safe_execute
from classes.florence_operation import florence_operator
from classes.logging import s3_logger
from classes.prompt_operation import s3_prompt
from classes.storage_operation import storage_operator
from common.const import (
    BEDROCK_DATA_SOURCE_ID,
    BEDROCK_KNOWLEDGE_BASE_ID,
    CLASSIFICATION_PROMPT_PATH,
    S3_CLASSIFICATION_RESULTS_PATH,
    S3_ENDFILE_PATH,
    S3_INPUT_PATH,
    S3_LOG_PATH,
)
from common.models import Case, GenaiClassificationResults, LabelMaster
from common.utils import from_utc_to_jst
from utils.data_utils import DataUtils
from utils.prompt_utils import PromptUtils
from utils.s3_utils import S3Utils


def init_args() -> argparse.Namespace:
    """コマンドライン引数を初期化する。
    Initialize command line arguments.

    Returns:
        解析されたコマンドライン引数 / Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="分類タスクを実行する")

    parser.add_argument(
        "--run_date",
        type=str,
        dest="run_date",
        required=True,
        help="タスク実行日 / Task execution date",
    )
    parser.add_argument(
        "--execution_id",
        type=str,
        dest="execution_id",
        required=True,
        help="Step Function execution ID",
    )
    return parser.parse_args()


@error_handler(
    "logs/classification.log", "ERR", "03", "1", "001", "001"
)
def get_prompt_template(prompt_path: str) -> str:
    """プロンプトテンプレートをS3から取得する。
    Get prompt template from S3.

    Args:
        prompt_path: プロンプトファイルのパス / Path to prompt file

    Returns:
        プロンプトテンプレート / Prompt template
    """
    s3_logger.info(f"分類プロンプトを取得: {prompt_path}")
    s3_logger.info(f"Retrieving classification prompt: {prompt_path}")
    return s3_prompt.get_prompt(prompt_path)


@error_handler(
    "logs/classification.log", "ERR", "03", "1", "001", "001"
)
def load_input_data(s3_key: str) -> pd.DataFrame:
    """入力CSVファイルをロードして検証する。
    Load and validate input CSV file.

    Args:
        s3_key: S3のキー / S3 key

    Returns:
        ロードしたデータフレーム / Loaded DataFrame
    """
    s3_logger.info(f"入力CSVファイルを取得: {s3_key}")
    s3_logger.info(f"Retrieving input CSV file: {s3_key}")
    
    csv_files = storage_operator.get_object(s3_key)
    df = pd.read_csv(csv_files)

    # データ件数のバリデーション / Validate data count
    if len(df) == 0:
        s3_logger.error("入力CSVファイルにデータがありません")
        s3_logger.error("Input CSV file contains no data")
        raise ValueError("Input CSV file is empty")

    s3_logger.info(f"入力されたDataFrameの形: {df.shape}")
    s3_logger.info(f"Input DataFrame shape: {df.shape}")
    s3_logger.info(f"入力されたDataFrameのヘッダー: {df.columns.to_list()}")
    s3_logger.info(f"Input DataFrame headers: {df.columns.to_list()}")

    # ヘッダーのバリデーション / Validate headers
    required_headers = ["kjoinfo_tsuban", "mosd_naiyou", "taioushousai", "kujo_partybiko"]
    if not set(required_headers).issubset(df.columns):
        s3_logger.error(f"必要なヘッダーがありません: {required_headers}")
        s3_logger.error(f"Required headers missing: {required_headers}")
        raise ValueError(f"Required headers missing: {required_headers}")

    return df


def search_similar_texts(
    texts: List[str], voice_nos: List[str], row_numbers: List[int], log_path: str
) -> Tuple[Dict[int, Dict[str, Any]], Set[str]]:
    """Bedrockを使用して類似テキストを検索する。
    Search for similar texts using Bedrock.

    Args:
        texts: 検索対象テキストのリスト / List of texts to search
        voice_nos: 苦情通番のリスト / List of complaint voice numbers
        row_numbers: 行番号のリスト / List of row numbers
        log_path: ログファイルのパス / Log file path

    Returns:
        (類似テキスト結果の辞書, 参照通番のセット) / 
        (Dictionary of similar text results, Set of reference voice numbers)
    """
    s3_logger.info("Bedrockから類似テキストを取得")
    s3_logger.info("Retrieving similar texts from Bedrock")
    
    with error_context(log_path, "ERR", "03", "1", "005", "003"):
        bedrock_results = bedrock_operator.batch_retrieve(
            knowledge_base_id=BEDROCK_KNOWLEDGE_BASE_ID,
            data_source_id=BEDROCK_DATA_SOURCE_ID,
            texts=texts,
        )
    
    # 結果の検証 / Validate results
    df_error_messages = {}
    for idx, result in enumerate(bedrock_results):
        if not result:
            warning = "WRN_03_005_1_012 Bedrock検索結果がブランク或いは０件"
            df_error_messages[idx] = warning
            s3_logger.warning(f"行 {idx+1}: {warning}")
            s3_logger.warning(f"Row {idx+1}: Bedrock search result is blank or empty")

    s3_logger.info("Bedrockからの類似テキストの取得に成功しました")
    s3_logger.info("Successfully retrieved similar texts from Bedrock")
    
    # 類似テキスト結果の整形 / Format similar text results
    similarity_results = {}
    references_voice_no = set()
    
    for rn, voice_no, text, results in zip(
        row_numbers, voice_nos, texts, bedrock_results
    ):
        tmp = []
        for obj in results:
            ref_voice_no = os.path.splitext(
                os.path.basename(obj["location"]["s3Location"]["uri"])
            )[0]
            tmp.append(
                {
                    "voice_no": ref_voice_no,
                    "text": str(obj["content"]["text"]),
                    "score": obj["score"],
                }
            )
            references_voice_no.add(ref_voice_no)
        
        similarity_results[rn] = {
            "voice_no": voice_no,
            "text": text,
            "results": tmp,
            "error_message": df_error_messages.get(rn-1, ""),
        }
    
    return similarity_results, references_voice_no


@error_handler(
    "logs/classification.log", "ERR", "03", "1", "002", "004"
)
def get_reference_metadata(
    references_voice_no: Set[str]
) -> Dict[str, Dict[str, str]]:
    """DBから参照データのメタデータを取得する。
    Get metadata for reference data from DB.

    Args:
        references_voice_no: 参照通番のセット / Set of reference voice numbers

    Returns:
        メタデータの辞書 / Dictionary of metadata
    """
    s3_logger.info("CVSデータベースから類似テキストのラベルを取得")
    s3_logger.info("Retrieving labels for similar texts from CVS database")
    
    # メタデータの初期化 / Initialize metadata
    references_metadata = {}
    
    # 分類情報を取得 / Get classification information
    statement = (
        select(
            Case.voice_no,
            LabelMaster.classification_name,
            LabelMaster.classification_code,
        )
        .join(
            LabelMaster,
            Case.classification_code == LabelMaster.classification_code,
            isouter=True,
        )
        .where(Case.voice_no.in_(list(references_voice_no)))
        .distinct()
    )
    
    s3_logger.info(f"実行するSQL: {str(statement)}")
    s3_logger.info(f"Executing SQL: {str(statement)}")
    
    for row in db_operator.execute_statement("statement", statement):
        references_metadata.setdefault(
            str(row.voice_no).strip(), {}
        ).update(
            {
                "classification": str(row.classification_name).strip(),
                "classification_code": str(row.classification_code).strip(),
            }
        )
    
    # 大中小分類を取得 / Get category information
    statement = (
        select(
            Case.voice_no,
            LabelMaster.category_name,
            LabelMaster.subcategory_name,
            LabelMaster.subsubcategory_name,
            LabelMaster.category_code,
            LabelMaster.subcategory_code,
            LabelMaster.subsubcategory_code,
        )
        .join(
            LabelMaster,
            and_(
                Case.category_code == LabelMaster.category_code,
                Case.subcategory_code == LabelMaster.subcategory_code,
                Case.subsubcategory_code == LabelMaster.subsubcategory_code,
            ),
            isouter=True,
        )
        .where(Case.voice_no.in_(list(references_voice_no)))
        .distinct()
    )
    
    s3_logger.info(f"実行するSQL: {str(statement)}")
    s3_logger.info(f"Executing SQL: {str(statement)}")
    
    for row in db_operator.execute_statement("statement", statement):
        references_metadata.setdefault(
            str(row.voice_no).strip(), {}
        ).update(
            {
                "category": str(row.category_name).strip(),
                "subcategory": str(row.subcategory_name).strip(),
                "subsubcategory": str(row.subsubcategory_name).strip(),
                "category_code": str(row.category_code).strip(),
                "subcategory_code": str(row.subcategory_code).strip(),
                "subsubcategory_code": str(row.subsubcategory_code).strip(),
            }
        )
    
    s3_logger.info(f"{len(references_metadata)} 件のメタデータを取得しました")
    s3_logger.info(f"Retrieved metadata for {len(references_metadata)} records")
    
    return references_metadata


def process_florence_responses(
    prompts: Dict[int, str], 
    chat_results: Dict[int, Dict[str, Any]], 
    similarity_results: Dict[int, Dict[str, Any]],
    log_path: str
) -> Tuple[Dict[int, str], Set[int]]:
    """Florence APIのレスポンスを処理する。
    Process Florence API responses.

    Args:
        prompts: プロンプトの辞書 / Dictionary of prompts
        chat_results: チャット結果の辞書 / Dictionary of chat results
        similarity_results: 類似度検索結果の辞書 / Dictionary of similarity search results
        log_path: ログファイルのパス / Log file path

    Returns:
        (修正したプロンプトの辞書, トークン超過インデックスのセット) / 
        (Dictionary of modified prompts, Set of token exceeded indices)
    """
    # Token制限のハンドリング / Handle token limit
    s3_logger.info("ステータスコード422のレスポンスがあるかどうかを確認")
    s3_logger.info("Checking for status code 422 responses")
    
    token_exceed_results_index = set()
    
    for rn, response in chat_results.items():
        status_code = response.get("status_code")
        text = str(response.get("text", ""))
        context_error = "model's maximum context length is" in text
        
        if status_code == 422 and context_error:
            token_exceed_results_index.add(rn)
            voice_no = similarity_results.get(rn, {}).get("voice_no", "Unknown")
            s3_logger.error(
                f"5-shotプロンプトがChatGPT Token制限を超過: 苦情通番 {voice_no}"
            )
            s3_logger.error(
                f"5-shot prompt exceeded ChatGPT token limit: Complaint ID {voice_no}"
            )
        elif status_code == 500:
            similarity_results[rn]["error_message"] = (
                similarity_results[rn].get("error_message", "") +
                "; WRN_03_1_004_006 分類処理システムエラーFlorenceAPI Timeout"
            ).strip("; ")
            s3_logger.warning(
                f"行番号 {rn}: Florence API タイムアウト"
            )
            s3_logger.warning(
                f"Row {rn}: Florence API timeout"
            )
    
    if token_exceed_results_index:
        s3_logger.info(
            f"{len(token_exceed_results_index)} 件のレスポンスが422で返されました"
        )
        s3_logger.info(
            f"{len(token_exceed_results_index)} responses returned with status code 422"
        )
        
        # 3-SHOT プロンプト生成 / Generate 3-shot prompts
        s3_logger.info("5-Shotで失敗したレコードを3-shotで再実行")
        s3_logger.info("Re-running records that failed with 5-shot using 3-shot")
        
        # 共通ユーティリティを使用 / Use common utility
        similarity_results_slice = {
            rn: similarity_results[rn] for rn in token_exceed_results_index
        }
    
    return prompts, token_exceed_results_index


@error_handler(
    "logs/classification.log", "ERR", "03", "1", "002", "004"
)
def get_label_mappings(log_path: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]], List[Tuple[str, str, str]]]:
    """DBから分類マッピング情報を取得する。
    Get classification mapping information from DB.

    Args:
        log_path: ログファイルのパス / Log file path

    Returns:
        (区分マッピング, カテゴリマッピング, カテゴリ組み合わせ) / 
        (Classification mapping, Category mapping, Category combinations)
    """
    # 区分コードと区分名を取得 / Get classification codes and names
    s3_logger.info("CVSデータベースからすべての区分コードと区分名を取得")
    s3_logger.info("Retrieving all classification codes and names from CVS database")
    
    classification_mapping = {}
    statement = select(
        LabelMaster.classification_code, LabelMaster.classification_name
    ).distinct()
    
    s3_logger.info(f"実行するSQL: {str(statement)}")
    s3_logger.info(f"Executing SQL: {str(statement)}")
    
    for row in db_operator.execute_statement("statement", statement):
        classification_mapping[str(row.classification_name).strip()] = str(
            row.classification_code
        ).strip()
    
    # 分類コードと分類名を取得 / Get category codes and names
    s3_logger.info("CVSデータベースからすべての分類コードと分類名を取得")
    s3_logger.info("Retrieving all category codes and names from CVS database")
    
    category_mapping = {}
    category_combination = []
    statement = select(
        LabelMaster.category_code,
        LabelMaster.category_name,
        LabelMaster.subcategory_code,
        LabelMaster.subcategory_name,
        LabelMaster.subsubcategory_code,
        LabelMaster.subsubcategory_name,
    ).distinct()
    
    s3_logger.info(f"実行するSQL: {str(statement)}")
    s3_logger.info(f"Executing SQL: {str(statement)}")
    
    for row in db_operator.execute_statement("statement", statement):
        category_combination.append(
            (
                str(row.category_name).strip(),
                str(row.subcategory_name).strip(),
                str(row.subsubcategory_name).strip(),
            )
        )
        category_mapping[
            str(row.category_name).strip()
            + str(row.subcategory_name).strip()
            + str(row.subsubcategory_name).strip()
        ] = {
            "category_code": str(row.category_code).strip(),
            "subcategory_code": str(row.subcategory_code).strip(),
            "subsubcategory_code": str(row.subsubcategory_code).strip(),
        }
    
    s3_logger.info(f"{len(classification_mapping)} 件の区分マッピングを取得")
    s3_logger.info(f"Retrieved {len(classification_mapping)} classification mappings")
    s3_logger.info(f"{len(category_mapping)} 件のカテゴリマッピングを取得")
    s3_logger.info(f"Retrieved {len(category_mapping)} category mappings")
    
    return classification_mapping, category_mapping, category_combination


@error_handler(
    "logs/classification.log", "ERR", "03", "1", "002", "002"
)
def save_results_to_db(df: pd.DataFrame, log_path: str) -> None:
    """分類結果をDBに保存する。
    Save classification results to DB.

    Args:
        df: 保存するデータフレーム / DataFrame to save
        log_path: ログファイルのパス / Log file path
    """
    s3_logger.info("データをCVSデータベースにエクスポート")
    s3_logger.info("Exporting data to CVS database")
    
    records = df[
        [
            "kjoinfo_tsuban",
            "genai_mshd_daibunric",
            "genai_mosd_chbnrui_c",
            "genai_mosd_shbnrui_c",
            "genai_mosd_saibnruic",
            "genai_msd_daibnrmei",
            "genai_mosdchbnrui",
            "genai_mosd_shbnrui",
            "genai_msdsaibnrmei",
            "tourokutstnp",
            "trksha_shzkc",
            "trksha_shzkmei",
            "tourokushacode",
            "tourokushamei",
            "finalupd_tstnp",
            "finalupopshzkc",
            "finluposhzkmei",
            "final_upd_op_c",
            "saishuksnsmei",
            "sakujo_flg",
            "sakujo_tstnp",
            "del_op_shzk_c",
            "del_op_shzkmei",
            "sakujosha_code",
            "sakujoshamei",
        ]
    ].to_dict("records")
    
    s3_logger.info(f"{len(records)} 件のレコードをDBに保存")
    s3_logger.info(f"Saving {len(records)} records to DB")
    
    db_operator.do_insert(
        table_object=GenaiClassificationResults,
        records=records,
        index_elements=["kjoinfo_tsuban"],
    )
    
    s3_logger.info("DBへの保存完了")
    s3_logger.info("Completed saving to DB")


def main() -> None:
    """メイン関数。
    Main function.
    """
    # 初期化 / Initialization
    args = init_args()
    execution_id = args.execution_id
    run_date = args.run_date
    run_date_jst = from_utc_to_jst(run_date).strftime("%Y%m%d")
    current_timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    
    # ログパスの生成 / Generate log path
    log_path = (
        S3_LOG_PATH.replace("@run_date", run_date_jst)
        .replace("@type", "classification")
        .replace("@execution_id", execution_id)
    )
    
    # 共通ユーティリティのインスタンス化 / Instantiate common utilities
    s3_utils = S3Utils(storage_operator)
    
    # 入力ファイルのパスを生成 / Generate input file path
    input_s3_key = (
        S3_INPUT_PATH.replace("@run_date", run_date_jst)
        .replace("@execution_id", execution_id)
        .replace("@type", "classification")
    )
    
    # 結果ファイルのパスを生成 / Generate result file path
    result_s3_key = S3_CLASSIFICATION_RESULTS_PATH.replace(
        "@run_date", run_date_jst
    ).replace("@execution_id", execution_id)
    
    # S3から分類プロンプトを取得 / Get classification prompt from S3
    classification_prompt = get_prompt_template(CLASSIFICATION_PROMPT_PATH)
    
    # S3から入力CSVファイルを取得 / Get input CSV file from S3
    df = load_input_data(input_s3_key)
    
    # データクリーニング / Data cleaning
    df, missing_data_rows = DataUtils.clean_text_data(df, ["mosd_naiyou"])
    if not missing_data_rows.empty:
        s3_logger.info(f"{len(missing_data_rows)} 行のデータに欠損値があります")
        s3_logger.info(f"{len(missing_data_rows)} rows have missing values")
    
    # テキストカラムの結合 / Combine text columns
    df["combined_text"] = DataUtils.combine_text_columns(
        df, ["mosd_naiyou", "taioushousai", "kujo_partybiko"]
    )
    df.replace({"combined_text": "nan"}, "", inplace=True, regex=True)
    
    # エラーメッセージ列を初期化 / Initialize error message column
    df["error_message"] = ""
    
    # テキスト、通番、行番号のリストを作成 / Create lists of text, voice number, row number
    texts = df["combined_text"].to_list()
    voices_no = df["kjoinfo_tsuban"].to_list()
    df["row_number"] = df.index + 1
    row_numbers = df["row_number"].to_list()
    
    # Bedrockから類似テキストを取得 / Get similar texts from Bedrock
    similarity_results, references_voice_no = search_similar_texts(
        texts, voices_no, row_numbers, log_path
    )
    
    # DBから類似テキストの区分を取得 / Get classification of similar texts from DB
    references_metadata = get_reference_metadata(references_voice_no)
    
    # プロンプト生成 / Generate prompts
    s3_logger.info("各レコードの分類プロンプトを生成")
    s3_logger.info("Generating classification prompts for each record")
    
    prompts = PromptUtils.generate_classification_prompt(
        classification_prompt,
        similarity_results,
        references_metadata,
        shot_number=5
    )
    
    # Florenceにリクエストを送信 / Send request to Florence API
    s3_logger.info("Florence APIにリクエストを送信")
    s3_logger.info("Sending request to Florence API")
    
    with error_context(log_path, "ERR", "03", "1", "004", "003"):
        chat_results = {}
        api_responses = florence_operator.chat(
            [prompt for rn, prompt in prompts.items() if prompt]
        )
        
        for rn, response in zip(
            [rn for rn, prompt in prompts.items() if prompt], 
            api_responses
        ):
            chat_results[rn] = response
    
    # Florence APIのレスポンスを処理 / Process Florence API responses
    prompts, token_exceed_results_index = process_florence_responses(
        prompts, chat_results, similarity_results, log_path
    )
    
    # 3-SHOTプロンプト生成と再リクエスト / Generate 3-shot prompts and re-request
    if token_exceed_results_index:
        similarity_results_slice = {
            rn: similarity_results[rn] for rn in token_exceed_results_index
        }
        
        s3_logger.info("3-SHOTプロンプトを生成")
        s3_logger.info("Generating 3-SHOT prompts")
        
        prompts_3shot = PromptUtils.generate_classification_prompt(
            classification_prompt,
            similarity_results_slice,
            references_metadata,
            shot_number=3
        )
        
        s3_logger.info("Florence APIにリクエストを再送信")
        s3_logger.info("Re-sending request to Florence API")
        
        with error_context(log_path, "ERR", "03", "1", "004", "003"):
            api_responses = florence_operator.chat(
                [prompt for rn, prompt in prompts_3shot.items() if prompt]
            )
            
            for rn, response in zip(
                [rn for rn, prompt in prompts_3shot.items() if prompt], 
                api_responses
            ):
                chat_results[rn] = response
                status_code = response.get("status_code")
                text = str(response.get("text", ""))
                context_error = "model's maximum context length is" in text
                
                if status_code == 422 and context_error:
                    df.loc[df["row_number"] == rn, "error_message"] = (
                        df.loc[df["row_number"] == rn, "error_message"].str.cat([
                            "WRN_03_1_004_005 分類処理システムエラーFlorenceトーケン数は制限値よりオーバーする"
                        ], sep="; ").str.strip("; ")
                    )
        
        # プロンプトを更新 / Update prompts
        for rn, prompt in prompts_3shot.items():
            prompts[rn] = prompt
    
    # プロンプトと応答をDataFrameに追加 / Add prompts and responses to DataFrame
    df["genai_classification_prompt"] = df["row_number"].apply(
        lambda x: (
            prompts.get(x, "")
            if prompts.get(x) and isinstance(prompts.get(x), str)
            else ""
        )
    )
    
    df["genai_classification_response"] = df["row_number"].apply(
        lambda x: (
            chat_results.get(x, {}).get("data", "")
            if chat_results.get(x, {}).get("data")
            and isinstance(chat_results.get(x, {}).get("data"), str)
            else ""
        )
    )
    
    df["genai_classify_run4md"] = current_timestamp
    
    # レスポンスから分類結果を抽出 / Extract classification results from response
    s3_logger.info("GPTのレスポンスからラベルを抽出")
    s3_logger.info("Extracting labels from GPT responses")
    
    # DataUtilsを使用して正規表現抽出 / Use DataUtils for regex extraction
    df["genai_msd_daibnrmei"] = df["genai_classification_response"].apply(
        lambda x: DataUtils.extract_with_regex(x, r"(?<=区分:)(.*?)(?=大分類:)", 1, "")
    )
    
    df["genai_mosdchbnrui"] = df["genai_classification_response"].apply(
        lambda x: DataUtils.extract_with_regex(x, r"(?<=大分類:)(.*?)(?=中分類:)", 1, "")
    )
    
    df["genai_mosd_shbnrui"] = df["genai_classification_response"].apply(
        lambda x: DataUtils.extract_with_regex(x, r"(?<=中分類:)(.*?)(?=小分類:)", 1, "")
    )
    
    df["genai_msdsaibnrmei"] = df["genai_classification_response"].apply(
        lambda x: DataUtils.extract_with_regex(x, r"小分類:(.*?)(?:\n|$)", 1, "")
    )
    
    # 区分コードと区分名を取得 / Get classification codes and names
    classification_mapping, category_mapping, category_combination = get_label_mappings(log_path)
    
    # 候補区分、候補分類以外の回答を調整 / Adjust responses outside candidate classification
    for col, obj in zip(
        [
            "genai_msd_daibnrmei",
            "genai_mosdchbnrui",
            "genai_mosd_shbnrui",
            "genai_msdsaibnrmei",
        ],
        ["classification", "category", "subcategory", "subsubcategory"],
    ):
        df[col] = df.apply(
            lambda row: (
                row[col]
                if row[col] in similarity_results.get(row["row_number"], {}).get("candidate", {}).get(obj, [])
                else "-"
            ),
            axis=1,
        )
    
    # エラーメッセージの設定 / Set error messages
    for idx, row in df.iterrows():
        if "-" in [
            row["genai_msd_daibnrmei"],
            row["genai_mosdchbnrui"],
            row["genai_mosd_shbnrui"],
            row["genai_msdsaibnrmei"],
        ]:
            warning_message = "WRN_03_1_004_013 指定する範囲外の分類結果がある"
            df.at[idx, "error_message"] = (
                (df.at[idx, "error_message"] + "; " + warning_message).strip("; ")
                if df.at[idx, "error_message"]
                else warning_message
            )
    
    # 大中小分類の組み合わせバリデーション / Validate category combinations
    s3_logger.info("大中小分類の組み合わせを検証")
    s3_logger.info("Validating category combinations")
    
    df["valid"] = df.apply(
        lambda row: (
            row["genai_mosdchbnrui"],
            row["genai_mosd_shbnrui"],
            row["genai_msdsaibnrmei"],
        ) in category_combination,
        axis=1,
    )
    
    s3_logger.info(f'有効なレコード数: {len(df[df["valid"] == True])}')
    s3_logger.info(f'Number of valid records: {len(df[df["valid"] == True])}')
    s3_logger.info(f'無効なレコード数: {len(df[df["valid"] == False])}')
    s3_logger.info(f'Number of invalid records: {len(df[df["valid"] == False])}')
    
    # 無効なレコードにエラーフラグを設定 / Set error flag for invalid records
    warning_message = "WRN_03_2_005_012 有効の分類情報が無し"
    df["error_message"] = df.apply(
        lambda row: (
            (row["error_message"] + "; " + warning_message).strip("; ")
            if not row["valid"] and row["error_message"]
            else warning_message if not row["valid"] else row["error_message"]
        ),
        axis=1,
    )
    
    # 無効な分類を"-"に変更 / Change invalid classifications to "-"
    for col in [
        "genai_mosdchbnrui",
        "genai_mosd_shbnrui",
        "genai_msdsaibnrmei",
    ]:
        df[col] = df.apply(
            lambda row: "-" if not row["valid"] else row[col], axis=1
        )
    
    # 区分コード、大中小分類コードをマッピング / Map classification and category codes
    df["category_combined"] = df.apply(
        lambda row: row["genai_mosdchbnrui"] + row["genai_mosd_shbnrui"] + row["genai_msdsaibnrmei"],
        axis=1,
    )
    
    df["genai_mshd_daibunric"] = df["genai_msd_daibnrmei"].apply(
        lambda x: classification_mapping.get(x, "")
    )
    
    df["genai_mosd_chbnrui_c"] = df["category_combined"].apply(
        lambda x: category_mapping.get(x, {}).get("category_code", "")
    )
    
    df["genai_mosd_shbnrui_c"] = df["category_combined"].apply(
        lambda x: category_mapping.get(x, {}).get("subcategory_code", "")
    )
    
    df["genai_mosd_saibnruic"] = df["category_combined"].apply(
        lambda x: category_mapping.get(x, {}).get("subsubcategory_code", "")
    )
    
    # 固定項目追加 / Add fixed items
    fixed_values = {
        "tourokutstnp": current_timestamp,
        "trksha_shzkc": "gen-ai",
        "trksha_shzkmei": "gen-ai",
        "tourokushacode": "gen-ai",
        "tourokushamei": "gen-ai",
        "finalupd_tstnp": current_timestamp,
        "finalupopshzkc": "gen-ai",
        "finluposhzkmei": "gen-ai",
        "final_upd_op_c": "gen-ai",
        "saishuksnsmei": "gen-ai",
        "sakujo_flg": "0",
        "sakujo_tstnp": None,
        "del_op_shzk_c": None,
        "del_op_shzkmei": None,
        "sakujosha_code": None,
        "sakujoshamei": None,
    }
    
    df = DataUtils.add_fixed_columns(df, fixed_values)
    
    # S3に出力 / Output to S3
    s3_logger.info(f"{result_s3_key} にデータをエクスポート")
    s3_logger.info(f"Exporting data to {result_s3_key}")
    
    storage_operator.put_df_to_s3(df=df, key=result_s3_key)
    
    # CVS DBにINSERT / INSERT to CVS DB
    save_results_to_db(df, log_path)
    
    # アウトプットファイル存在チェック / Check output file existence
    s3_logger.info("アウトプットCSVファイル存在チェックを行う")
    s3_logger.info("Checking output CSV file existence")
    
    s3_utils.check_file_exists(
        key=result_s3_key,
        log_path=log_path,
        error_type="ERR",
        error_main="03",
        error_category="1",
        error_module="001",
        error_detail="001"
    )
    
    # アウトプットファイルヘッダーチェック / Check output file headers
    s3_logger.info("アウトプットCSVファイルのヘッダーチェックを行う")
    s3_logger.info("Checking output CSV file headers")
    
    s3_utils.check_file_headers(
        key=result_s3_key,
        expected_headers=df.columns.tolist(),
        log_path=log_path,
        error_type="ERR",
        error_main="03", 
        error_category="1",
        error_module="001",
        error_detail="008"
    )
    
    # 前回実行日時の取得 / Get last run date
    endfile_s3_key_prefix = S3_ENDFILE_PATH.replace("@type", "classification")
    
    last_run_date, last_run_endfile = s3_utils.get_last_run_date(
        endfile_prefix=endfile_s3_key_prefix,
        log_path=log_path,
        error_type="ERR",
        error_main="03",
        error_category="1",
        error_module="001",
        error_detail="001"
    )
    
    # endfileの管理（バックアップと新規作成）/ Manage endfile (backup and create new)
    s3_utils.manage_endfile(
        endfile_prefix=endfile_s3_key_prefix,
        last_run_endfile=last_run_endfile,
        run_date=args.run_date,
        log_path=log_path,
        error_type="ERR",
        error_main="03",
        error_category="1", 
        error_module="001",
        error_detail="002"
    )
    
    s3_logger.info("分類処理が完了しました")
    s3_logger.info("Classification processing completed")


if __name__ == "__main__":
    main()

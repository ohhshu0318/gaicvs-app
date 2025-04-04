import argparse
import asyncio
import os
import re
from datetime import datetime
from io import StringIO
from zoneinfo import ZoneInfo

import pandas as pd
from classes.bedrock_operation import bedrock_operator
from classes.db_operation import db_operator
from classes.error_handler import CustomError, log_and_raise_error
from classes.florence_operation import florence_operator
from classes.logging_handler import application_logger as logger
from classes.prompt_operation import s3_prompt
from classes.storage_operation import storage_operator
from common.const import (BEDROCK_DATA_SOURCE_ID, BEDROCK_KNOWLEDGE_BASE_ID,
                          CLASSIFICATION_PROMPT_PATH,
                          S3_CLASSIFICATION_RESULTS_PATH, S3_ENDFILE_PATH,
                          S3_INPUT_PATH, S3_LOG_PATH)
from common.models import Case, GenaiClassificationResults, LabelMaster
from sqlalchemy import and_, select
from sqlalchemy.exc import OperationalError
from utils.jst_utils import from_utc_to_jst


def init_args():
    parser = argparse.ArgumentParser(
        description='分類タスクを実行する')

    parser.add_argument('--run_date', type=str, dest='run_date', required=True,
                        help='タスク実行日')
    parser.add_argument('--execution_id', type=str,
                        dest='execution_id', required=True,
                        help='Step Function execution ID')
    return parser.parse_args()


def generate_prompt(
    prompt_template: str,
    similarity_results: dict,
    references_metadata: dict,
    shot_number: int = 5,
):
    prompts = {}
    for rn, data in similarity_results.items():
        references = data.get('results')
        if not references:
            prompts[rn] = ''
            data['candidate'] = {
                'classification': [],
                'category': [],
                'subcategory': [],
                'subsubcategory': []
            }
            continue
        references = sorted(references,
                            key=lambda x: x['score'],
                            reverse=True)[:shot_number]
        reference_text = ''
        classification_candidates = []
        category_candidates = []
        subcategory_candidates = []
        subsubcategory_candidates = []
        for ref in references:
            classification = references_metadata.get(
                ref['voice_no'], {}).get('classification', '')
            category = references_metadata.get(ref['voice_no']).get('category')
            subcategory = references_metadata.get(
                ref['voice_no']).get('subcategory')
            subsubcategory = references_metadata.get(
                ref['voice_no']).get('subsubcategory')
            cleaned_text = (
                str(ref['text'])
                .replace('\n', ' ')
                .replace('\r\n', ' ')
                .replace('\r', ' ')
                )
            reference_text += (
                '文章: {}\n区分: {}\n大分類: {}\n中分類: {}\n小分類: {}\n\n'.format(
                    cleaned_text,
                    classification,
                    category,
                    subcategory,
                    subsubcategory
                    )
            )
            classification_candidates.append(classification)
            category_candidates.append(category)
            subcategory_candidates.append(subcategory)
            subsubcategory_candidates.append(subsubcategory)
        prompt = prompt_template.format(
            input_text=(str(data['text']).replace('\n', ' ')
                        .replace('\r\n', ' ').replace('\r', ' ')),
            reference_text=reference_text,
            classification=','.join(
                set(filter(None, classification_candidates))),
            category=','.join(set(filter(None, category_candidates))),
            subcategory=','.join(set(filter(None, subcategory_candidates))),
            subsubcategory=','.join(set(filter(None, subsubcategory_candidates)))
        )
        data['candidate'] = {
            'classification': classification_candidates,
            'category': category_candidates,
            'subcategory': subcategory_candidates,
            'subsubcategory': subsubcategory_candidates
        }
        prompts[rn] = prompt
    return prompts


def validate_classification_combinations(
    df: pd.DataFrame,
    valid_classification: list[str],
    valid_category: list[tuple],
    valid_subcategory: list[tuple],
    valid_subsubcategory: list[tuple]
) -> pd.DataFrame:
    df['genai_msd_daibnrmei'] = df['genai_msd_daibnrmei'] \
        .apply(lambda x: '-' if x not in valid_classification else x)
    df['genai_mosdchbnrui'] = df[['genai_msd_daibnrmei', 'genai_mosdchbnrui']] \
        .apply(lambda x: '-' if (x.genai_msd_daibnrmei, x.genai_mosdchbnrui) not in valid_category else x.genai_mosdchbnrui, axis=1)
    df['genai_mosd_shbnrui'] = df[['genai_msd_daibnrmei', 'genai_mosdchbnrui', 'genai_mosd_shbnrui']] \
        .apply(lambda x: '-' if (x.genai_msd_daibnrmei, x.genai_mosdchbnrui, x.genai_mosd_shbnrui) not in valid_subcategory else x.genai_mosd_shbnrui, axis=1)
    df['genai_msdsaibnrmei'] = df[['genai_msd_daibnrmei', 'genai_mosdchbnrui', 'genai_mosd_shbnrui', 'genai_msdsaibnrmei']] \
        .apply(lambda x: '-' if (x.genai_msd_daibnrmei, x.genai_mosdchbnrui, x.genai_mosd_shbnrui, x.genai_msdsaibnrmei) not in valid_subsubcategory else x.genai_msdsaibnrmei, axis=1)
    return df


def main():
    args = init_args()
    execution_id = args.execution_id
    run_date = args.run_date
    run_date = from_utc_to_jst(run_date).strftime('%Y%m%d')
    current_timestamp = datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()
    log_path = S3_LOG_PATH.replace('@run_date', run_date).replace('@type', 'classification').replace('@execution_id', execution_id)

    # 実行コンテキストの設定（新機能）
    logger.set_execution_context("classification", execution_id)

    input_s3_key = S3_INPUT_PATH.replace('@run_date', run_date)\
        .replace('@execution_id', execution_id).replace('@type', "classification")
    result_s3_key = S3_CLASSIFICATION_RESULTS_PATH.replace('@run_date', run_date)\
        .replace('@execution_id', execution_id)
    # S3 Landingからデータ取得
    # 存在しない場合、エラーになる
    logger.info(f"CSVファイルを取得する: {result_s3_key}")
    try:
        classification_prompt = s3_prompt.get_prompt(CLASSIFICATION_PROMPT_PATH)
    except Exception as e:
        log_and_raise_error(log_path, "ERR", "03", "1", "001", "001", str(e))

    try:
        csv_files = storage_operator.get_object(input_s3_key)
        df = pd.read_csv(StringIO(csv_files))

        # データ件数のバリデーション
        # 件数が０件の場合は警告ログを出して正常終了
        if len(df) == 0:
            logger.warning("データが0件のため、処理をスキップして正常終了します")

            # 今回stepfunctionの実行時間を保持するendfileを作成する
            endfile_s3_key_prefix = S3_ENDFILE_PATH.replace('@type', 'classification')
            endfile_s3_key = os.path.join(endfile_s3_key_prefix, f'{from_utc_to_jst(args.run_date).isoformat()}.txt')
            try:
                storage_operator.put_object(key=endfile_s3_key, body='')
                logger.info(f"空のendfileを作成しました: {endfile_s3_key}")
            except Exception as e:
                logger.error(f"endfileの作成に失敗しました: {str(e)}")

            return  # 処理を終了
    except CustomError:
        raise
    except Exception as e:
        log_and_raise_error(log_path, "ERR", "03", "1", "001", "001", str(e))

    logger.info(f'入力されたDataFrameの形: {df.shape}')

    # ヘッダーのバリデーション
    # ヘッダー欠損の場合、エラーになる
    logger.info(f'入力されたDataFrameのヘッダー: {df.columns.to_list()}')
    if not {'kjoinfo_tsuban', 'mosd_naiyou', 'taioushousai', 'kujo_partybiko'}.issubset(df.columns):
        log_and_raise_error(log_path, "ERR", "03", "1", "001", "008")

    df['mosd_naiyou'] = df['mosd_naiyou'].replace([None, "NaN", "null", ""], "missing_value").str.strip()
    df['mosd_naiyou'] = df['mosd_naiyou'].replace("", "missing_value")
    missing_data_rows = df[df['mosd_naiyou'] == "missing_value"]
    if not missing_data_rows.empty:
        df = df[df['mosd_naiyou'] != "missing_value"]
        logger.info("欠落値や空の値がある行を削除しました")

    # Bedrockから類似テキストを取得
    logger.info('Bedrockから類似テキストを取得する')
    df['error_message'] = ''
    df['combined_text'] = (df['mosd_naiyou'].astype(str) + '\n' + df['taioushousai'].astype(str) + '\n' + df['kujo_partybiko'].astype(str))
    df.replace({'combined_text': 'nan'}, '', inplace=True, regex=True)
    texts = df['combined_text'].to_list()
    voices_no = df['kjoinfo_tsuban'].to_list()
    df['row_number'] = df.index + 1
    row_numbers = df['row_number'].to_list()
    bedrock_results = bedrock_operator.batch_retrieve(
        knowledge_base_id=BEDROCK_KNOWLEDGE_BASE_ID,
        data_source_id=BEDROCK_DATA_SOURCE_ID,
        texts=texts
    )
    for idx, result in enumerate(bedrock_results):
        if not result:
            df.at[idx, 'error_message'] = "WRN_03_1_005_011 分類処理システムエラーBedrock検索結果がブランク或いは０件"

    logger.info('Bedrockからの類似テキストの取得に成功しました')
    similarity_results = {}  # 類似テキストの結果
    references_voice_no = set()  # 類似テキストの参照リスト
    for rn, voice_no, text, results in zip(row_numbers, voices_no,
                                           texts, bedrock_results):
        tmp = []
        for obj in results:
            ref_voice_no = os.path.splitext(
                os.path.basename(obj['location']['s3Location']['uri']))[0]
            tmp.append({
                'voice_no': ref_voice_no,
                'text': str(obj['content']['text']),
                'score': obj['score']
            })
            references_voice_no.add(ref_voice_no)
        similarity_results[rn] = {
            'voice_no': voice_no, 'text': text, 'results': tmp
        }

    # CVS DBから類似テキストの大中小細分類を取得する
    logger.info('CVSデータベースから類似テキストのラベルを取得する')
    references_metadata = {}  # メタデータの結果
    statement = select(
        Case.voice_no,
        LabelMaster.classification_name,
        LabelMaster.category_name,
        LabelMaster.subcategory_name,
        LabelMaster.subsubcategory_name,
        LabelMaster.classification_code,
        LabelMaster.category_code,
        LabelMaster.subcategory_code,
        LabelMaster.subsubcategory_code
    ) \
        .join(
            LabelMaster,
            and_(
                Case.classification_code == LabelMaster.classification_code,
                Case.category_code == LabelMaster.category_code,
                Case.subcategory_code == LabelMaster.subcategory_code,
                Case.subsubcategory_code == LabelMaster.subsubcategory_code
            ),
            isouter=True
        ) \
        .where(Case.voice_no.in_(list(references_voice_no))) \
        .distinct()
    logger.info(f'実行するSQL: {str(statement)}')
    try:
        for row in db_operator.execute_statement('statement', statement):
            references_metadata.setdefault(str(row.voice_no).strip(), {}).update({
                'classification': str(row.classification_name).strip(),
                'category':  str(row.category_name).strip(),
                'subcategory':  str(row.subcategory_name).strip(),
                'subsubcategory':  str(row.subsubcategory_name).strip(),
                'classification_code': str(row.classification_code).strip(),
                'category_code':  str(row.category_code).strip(),
                'subcategory_code':  str(row.subcategory_code).strip(),
                'subsubcategory_code':  str(row.subsubcategory_code).strip()
            })
    except OperationalError as e:
        log_and_raise_error(log_path, "ERR", "03", "1", "002", "004", str(e))
    except Exception as e:
        log_and_raise_error(log_path, "ERR", "03", "1", "002", "002", str(e))
    logger.debug(bedrock_results[:5])
    logger.debug(references_voice_no)
    logger.debug(references_metadata)
    logger.debug(similarity_results)

    # 5-SHOT プロンプト生成
    logger.info('各レコードの分類プロンプトを生成する')
    prompts = generate_prompt(classification_prompt,
                              similarity_results,
                              references_metadata,
                              shot_number=5)
    logger.debug(prompts)
    logger.debug(similarity_results)

    # Florenceにリクエストを送信
    logger.info('Florence APIにリクエストを送信する')
    loop = asyncio.get_event_loop()
    try:
        chat_results = {}
        api_responses = loop.run_until_complete(florence_operator.chat([prompt for rn, prompt in prompts.items()], 'gpt-4o'))
        for rn, response in zip(prompts, api_responses):
            chat_results[rn] = response
    except Exception as e:
        log_and_raise_error(log_path, "ERR", "03", "1", "004", "003", str(e))

    # Token制限のハンドリング
    logger.info('ステータスコード422のレスポンスがあるかどうかを確認する')
    token_exceed_results_index = set()
    failed_rows = set()  # 処理に失敗した行番号を格納

    for rn, response in chat_results.items():
        status_code = response.get('status_code')
        text = str(response.get('text'))
        context_error = "model's maximum context length is" in text
        if status_code == 422 and context_error:
            token_exceed_results_index.add(rn)
            logger.error(f'5-shotプロンプトがChatGPT Token制限を超過した。\n 苦情通番：{similarity_results.get(rn, {}).get("voice_no")}')
        if status_code == 500:
            df.at[rn - 1, 'error_message'] = "WRN_03_1_004_006 分類処理システムエラーFlorenceAPI Timeout"

    if token_exceed_results_index:
        logger.info(f'{len(token_exceed_results_index)} 件のレスポンスが422で返されました')
        # 3-SHOT プロンプト生成
        logger.info('5-Shotで失敗したレコードを3-shotで再実行する')
        similarity_results_slice = {
            rn: similarity_results[rn] for rn in token_exceed_results_index
        }
        prompts_3shot = generate_prompt(classification_prompt,
                                        similarity_results_slice,
                                        references_metadata,
                                        shot_number=3)
        logger.debug(prompts_3shot)
        # Florenceにリクエストを再送信
        api_responses = loop.run_until_complete(florence_operator.chat([prompt for rn, prompt in prompts_3shot.items()], 'gpt-4o'))
        for rn, response in zip(prompts_3shot, api_responses):
            chat_results[rn] = response
            status_code = response.get('status_code')
            context_error = "model's maximum context length is" in str(response.get('text', ''))
            if status_code == 422 and context_error:
                # トークンエラーが続く行は処理をスキップする
                failed_rows.add(rn - 1)  # DataFrameのインデックスに変換
                df.at[rn - 1, 'error_message'] = "WRN_03_1_004_005 分類処理システムエラーFlorenceトーケン数は制限値よりオーバーする"
                logger.warning(f"トークン制限エラーが続くため、インデックス {rn-1} の行（苦情通番: {similarity_results.get(rn, {}).get('voice_no')}）の処理をスキップします")
        for rn, prompt in prompts_3shot.items():
            prompts[rn] = prompt
    loop.close()
    # エラー行を除外した処理を続行
    df['genai_classification_prompt'] = df['row_number'].apply(lambda x: prompts.get(x) if prompts.get(x) and isinstance(prompts.get(x), str) else '')
    df['genai_classification_response'] = df['row_number'].apply(lambda x: chat_results.get(x, {}).get('data') if chat_results.get(x, {}).get('data') and isinstance(chat_results.get(x, {}).get('data'), str) else '')
    df['genai_classify_run4md'] = current_timestamp

    # 続行不可能なエラー行がある場合はフラグを立てる
    df['process_skip'] = df.index.isin(failed_rows)
    logger.info(f"トークン制限エラーなどでスキップされる行数: {len(failed_rows)}")

    # エラー行を含めて結果を記録するが、後の処理ではスキップする

    # レスポンスから分類結果を抽出（処理スキップ行は除外）
    logger.info('GPTのレスポンスからラベルを抽出する')
    df['genai_msd_daibnrmei'] = df.apply(
        lambda row: (re.search(
            r"(?<=区分:)(.*?)(?=大分類:)",
            str(row["genai_classification_response"]),
            re.DOTALL
        ).group().strip("\n").strip() if re.search(
            r"(?<=区分:)(.*)(?=大分類:)",
            str(row["genai_classification_response"]),
            re.DOTALL
        ) else "") if not row.get('process_skip', False) else "",
        axis=1
    )
    df['genai_mosdchbnrui'] = df.apply(
        lambda row: (re.search(
            r"(?<=大分類:)(.*?)(?=中分類:)",
            str(row["genai_classification_response"]),
            re.DOTALL
        ).group().strip("\n").strip() if re.search(
            r"(?<=大分類:)(.*)(?=中分類:)",
            str(row["genai_classification_response"]),
            re.DOTALL
        ) else "") if not row.get('process_skip', False) else "",
        axis=1
    )
    df['genai_mosd_shbnrui'] = df.apply(
        lambda row: (re.search(
            r"(?<=中分類:)(.*?)(?=小分類:)",
            str(row["genai_classification_response"]),
            re.DOTALL
        ).group().strip("\n").strip() if re.search(
            r"(?<=中分類:)(.*)(?=小分類:)",
            str(row["genai_classification_response"]),
            re.DOTALL
        ) else "") if not row.get('process_skip', False) else "",
        axis=1
    )
    df['genai_msdsaibnrmei'] = df.apply(
        lambda row: (re.search(
            r"小分類:(.*?)(?:\n|$)",
            str(row["genai_classification_response"]),
            re.DOTALL
        ).group(1).strip("\n").strip() if re.search(
            r"小分類:(.*)",
            str(row["genai_classification_response"]),
            re.DOTALL
        ) else "") if not row.get('process_skip', False) else "",
        axis=1
    )

    # 分類コードと分類名を取得する
    logger.info('CVSデータベースからすべての分類コードと分類名を取得する')
    category_mapping = []
    valid_classification, valid_category, valid_subcategory, valid_subsubcategory = [], [], [], []
    statement = select(
        LabelMaster.classification_code,
        LabelMaster.classification_name,
        LabelMaster.category_code,
        LabelMaster.category_name,
        LabelMaster.subcategory_code,
        LabelMaster.subcategory_name,
        LabelMaster.subsubcategory_code,
        LabelMaster.subsubcategory_name).distinct()
    logger.info(f'実行するSQL: {str(statement)}')
    try:
        for row in db_operator.execute_statement('statement', statement):
            valid_classification.append(
                str(row.classification_name).strip()
            )
            valid_category.append((
                str(row.classification_name).strip(),
                str(row.category_name).strip()
            ))
            valid_subcategory.append((
                str(row.classification_name).strip(),
                str(row.category_name).strip(),
                str(row.subcategory_name).strip()
            ))
            valid_subsubcategory.append((
                str(row.classification_name).strip(),
                str(row.category_name).strip(),
                str(row.subcategory_name).strip(),
                str(row.subsubcategory_name).strip()
            ))
            category_mapping.append({
                "genai_mshd_daibunric": str(row.classification_code).strip(),
                "genai_mosd_chbnrui_c": str(row.category_code).strip(),
                "genai_mosd_shbnrui_c": str(row.subcategory_code).strip(),
                "genai_mosd_saibnruic": str(row.subsubcategory_code).strip(),
                "genai_msd_daibnrmei": str(row.classification_name).strip(),
                "genai_mosdchbnrui": str(row.category_name).strip(),
                "genai_mosd_shbnrui": str(row.subcategory_name).strip(),
                "genai_msdsaibnrmei": str(row.subsubcategory_name).strip()
            })
    except OperationalError as e:
        log_and_raise_error(log_path, "ERR", "03", "1", "002", "004", str(e))
    except Exception as e:
        log_and_raise_error(log_path, "ERR", "03", "1", "002", "001", str(e))

    logger.debug(category_mapping)

    # 候補区分、候補分類以外の回答を調整する
    for col, obj in zip(['genai_msd_daibnrmei', 'genai_mosdchbnrui', 'genai_mosd_shbnrui', 'genai_msdsaibnrmei'],
                        ['classification', 'category', 'subcategory', 'subsubcategory']):
        df[col] = df[[col, 'row_number', 'process_skip']].apply(
            lambda x: getattr(x, col) if not x.process_skip and getattr(x, col) in similarity_results[x.row_number]['candidate'][obj] else '-',
            axis=1
        )
        df['error_message'] = df[[col, 'row_number', 'error_message', 'process_skip']].apply(
            lambda x: str(str(x.error_message) + '; WRN_03_1_004_013 分類処理システムエラーFlorence指定する範囲外の分類結果がある').strip('; ')
            if not x.process_skip and similarity_results[x.row_number]['candidate'][obj] and getattr(x, col) not in similarity_results[x.row_number]['candidate'][obj] else str(x.error_message),
            axis=1
        )

    # 大中小細分類の組み合わせバリデーション
    logger.info('大中小分類の組み合わせを検証する')
    df['error_message'] = df[['genai_msd_daibnrmei', 'genai_mosdchbnrui', 'genai_mosd_shbnrui', 'genai_msdsaibnrmei', 'error_message']] \
        .apply(lambda x: "WRN_03_1_004_012 分類処理システムエラーFlorenceありえない大中小細分類結果がある"
               if (x.genai_msd_daibnrmei, x.genai_mosdchbnrui, x.genai_mosd_shbnrui, x.genai_msdsaibnrmei) not in valid_subsubcategory
               and '-' not in (x.genai_msd_daibnrmei, x.genai_mosdchbnrui, x.genai_mosd_shbnrui, x.genai_msdsaibnrmei)
               and '' not in (x.genai_msd_daibnrmei, x.genai_mosdchbnrui, x.genai_mosd_shbnrui, x.genai_msdsaibnrmei)
               else x.error_message,
               axis=1)
    df = validate_classification_combinations(
        df, valid_classification, valid_category, valid_subcategory, valid_subsubcategory
    )
    df['valid'] = df[['genai_msd_daibnrmei', 'genai_mosdchbnrui', 'genai_mosd_shbnrui', 'genai_msdsaibnrmei']] \
        .apply(lambda x: False if (x.genai_msd_daibnrmei, x.genai_mosdchbnrui, x.genai_mosd_shbnrui, x.genai_msdsaibnrmei) == ('-', '-', '-', '-') else True, axis=1)
    logger.info(f'有効なレコード数: {len(df[df["valid"] == True])}')
    logger.info(f'無効なレコード数: {len(df[df["valid"] == False])}')

    # 区分コード、大中小分類コードをマッピングする
    df_category_mapping = pd.DataFrame(category_mapping)
    df = df.merge(
        df_category_mapping[['genai_msd_daibnrmei', 'genai_mshd_daibunric']].drop_duplicates(),
        how='left',
        on=['genai_msd_daibnrmei']
    )
    df = df.merge(
        df_category_mapping[['genai_msd_daibnrmei', 'genai_mosdchbnrui', 'genai_mosd_chbnrui_c']].drop_duplicates(),
        how='left',
        on=['genai_msd_daibnrmei', 'genai_mosdchbnrui']
    )
    df = df.merge(
        df_category_mapping[['genai_msd_daibnrmei', 'genai_mosdchbnrui', 'genai_mosd_shbnrui', 'genai_mosd_shbnrui_c']].drop_duplicates(),
        how='left',
        on=['genai_msd_daibnrmei', 'genai_mosdchbnrui', 'genai_mosd_shbnrui']
    )
    df = df.merge(
        df_category_mapping[['genai_msd_daibnrmei', 'genai_mosdchbnrui', 'genai_mosd_shbnrui', 'genai_msdsaibnrmei', 'genai_mosd_saibnruic']].drop_duplicates(),
        how='left',
        on=['genai_msd_daibnrmei', 'genai_mosdchbnrui', 'genai_mosd_shbnrui', 'genai_msdsaibnrmei']
    )
    df['genai_mshd_daibunric'] = df['genai_mshd_daibunric'].fillna("")
    df['genai_mosd_chbnrui_c'] = df['genai_mosd_chbnrui_c'].fillna("")
    df['genai_mosd_shbnrui_c'] = df['genai_mosd_shbnrui_c'].fillna("")
    df['genai_mosd_saibnruic'] = df['genai_mosd_saibnruic'].fillna("")
    df['genai_mshd_daibunric'] = df['genai_mshd_daibunric'].apply(lambda x: x if x else "")
    df['genai_mosd_chbnrui_c'] = df['genai_mosd_chbnrui_c'].apply(lambda x: x if x else "")
    df['genai_mosd_shbnrui_c'] = df['genai_mosd_shbnrui_c'].apply(lambda x: x if x else "")
    df['genai_mosd_saibnruic'] = df['genai_mosd_saibnruic'].apply(lambda x: x if x else "")

    # 固定項目追加
    df['tourokutstnp'] = current_timestamp
    df['trksha_shzkc'] = 'gen-ai'
    df['trksha_shzkmei'] = 'gen-ai'
    df['tourokushacode'] = 'gen-ai'
    df['tourokushamei'] = 'gen-ai'
    df['finalupd_tstnp'] = current_timestamp
    df['finalupopshzkc'] = 'gen-ai'
    df['finluposhzkmei'] = 'gen-ai'
    df['final_upd_op_c'] = 'gen-ai'
    df['saishuksnsmei'] = 'gen-ai'
    df['sakujo_flg'] = '0'
    df['sakujo_tstnp'] = None
    df['del_op_shzk_c'] = None
    df['del_op_shzkmei'] = None
    df['sakujosha_code'] = None
    df['sakujoshamei'] = None

    # process_skipフラグが立っている行を最終出力から除外
    df_output = df[~df['process_skip']]

    # 有効な結果があるか確認
    if len(df_output) == 0:
        logger.warning("処理可能なデータがありません。トークン制限エラーなどにより全ての行がスキップされました。")
        # 空のデータでも処理は続行し、空の結果ファイルを生成

    # S3に出力
    logger.info(f'{result_s3_key}にデータをエクスポートする')
    storage_operator.put_df_to_s3(df=df_output, key=result_s3_key)

    # スキップした行の情報をログに残す
    if len(df) != len(df_output):
        skipped_records = df[df['process_skip']]['kjoinfo_tsuban'].tolist()
        logger.info(f"スキップされた苦情通番: {skipped_records}")

    # CVS DBにINSERT（有効な結果のみ）
    logger.info('データをCVSデータベースにエクスポートする')
    if len(df_output) > 0:
        records = df_output[[
            'kjoinfo_tsuban', 'genai_mshd_daibunric', 'genai_mosd_chbnrui_c',
            'genai_mosd_shbnrui_c', 'genai_mosd_saibnruic', 'genai_msd_daibnrmei',
            'genai_mosdchbnrui', 'genai_mosd_shbnrui', 'genai_msdsaibnrmei',
            'tourokutstnp', 'trksha_shzkc', 'trksha_shzkmei',
            'tourokushacode', 'tourokushamei', 'finalupd_tstnp',
            'finalupopshzkc', 'finluposhzkmei', 'final_upd_op_c',
            'saishuksnsmei', 'sakujo_flg', 'sakujo_tstnp',
            'del_op_shzk_c', 'del_op_shzkmei', 'sakujosha_code',
            'sakujoshamei'
        ]].to_dict('records')
    else:
        records = []

    logger.debug(df.head().to_dict('records'))
    try:
        # レコードが存在する場合のみDBに挿入
        if records:
            db_operator.do_insert(
                table_object=GenaiClassificationResults,
                records=records,
                index_elements=['kjoinfo_tsuban']
            )
            logger.info(f"{len(records)}件のレコードをDBに挿入しました")
        else:
            logger.warning("DBに挿入するレコードがありません")
    except OperationalError as e:
        log_and_raise_error(log_path, "ERR", "03", "1", "002", "004", str(e))
    except Exception as e:
        log_and_raise_error(log_path, "ERR", "03", "1", "002", "002", str(e))

    # アウトプットファイル存在チェック
    # 存在しない場合、エラーになる
    logger.info('アウトプットCSVファイル存在チェックを行う')
    try:
        file_exists = storage_operator.check_object_exists(key=result_s3_key)
        if not file_exists:
            log_and_raise_error(log_path, "ERR", "03", "1", "001", "001")
    except CustomError:
        raise
    except Exception as e:
        log_and_raise_error(log_path, "ERR", "03", "1", "001", "001", str(e))

    # アウトプットファイルヘッダーチェック
    # ヘッダー欠損の場合、エラーになる
    df_tmp = pd.read_csv(StringIO(storage_operator.get_object(key=result_s3_key)))
    logger.info(f'アウトプットのヘッダー: {df_tmp.columns.to_list()}')
    if set(df_tmp.columns) < set(df.columns):
        log_and_raise_error(log_path, "ERR", "03", "1", "001", "008")

    # 前回stepfunctionの実行時間を取得する
    try:
        endfile_s3_key_prefix = S3_ENDFILE_PATH.replace('@type', 'classification')
        txt_files = list(filter(lambda x: '.txt' in x, storage_operator.list_objects(prefix=endfile_s3_key_prefix)))
        logger.debug(txt_files)
        if not txt_files:
            log_and_raise_error(log_path, "ERR", "03", "1", "001", "001")
    except CustomError:
        raise
    except Exception as e:
        log_and_raise_error(log_path, "ERR", "03", "1", "001", "001", str(e))

    last_run_endfile = sorted(txt_files, reverse=True)[0]
    logger.info(f'前回のendfileを取得する: {last_run_endfile}')

    # 前回stepfunction生成したendfileを移動する
    try:
        _ = storage_operator.move_object(
            destination_key=str(last_run_endfile).replace(
                endfile_s3_key_prefix, os.path.join(endfile_s3_key_prefix, 'backup/')),
            source_key=last_run_endfile
        )

        # 今回stepfunctionの実行時間を保持するendfileを作成する
        endfile_s3_key = os.path.join(endfile_s3_key_prefix, f'{from_utc_to_jst(args.run_date).isoformat()}.txt')
        storage_operator.put_object(key=endfile_s3_key, body='')
    except Exception as e:
        log_and_raise_error(log_path, "ERR", "03", "1", "001", "002", str(e))


if __name__ == '__main__':
    main()

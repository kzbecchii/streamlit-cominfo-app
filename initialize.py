"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from docx import Document
from langchain_community.document_loaders import WebBaseLoader
import csv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import constants as ct
from constants import RETRIEVER_DOCUMENT_COUNT, CHUNK_SIZE, CHUNK_OVERLAP


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように）
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    # ロガーを読み込むことで、後続の処理中に発生したエラーなどがログファイルに記録される
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにRetrieverが作成済みの場合、後続の処理を中断
    if "retriever" in st.session_state:
        return
    
    # RAGの参照先となるデータソースの読み込み
    docs_all = load_data_sources()

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    # 埋め込みモデルの用意
    embeddings = OpenAIEmbeddings()
    
    # チャンク分割用のオブジェクトを作成
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n"
    )

    # チャンク分割を実施
    splitted_docs = text_splitter.split_documents(docs_all)

    # ベクターストアの作成
    db = Chroma.from_documents(splitted_docs, embedding=embeddings)

    # ベクターストアを検索するRetrieverの作成
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": RETRIEVER_DOCUMENT_COUNT})


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    web_docs_all = []
    # ファイルとは別に、指定のWebページ内のデータも読み込み
    # 読み込み対象のWebページ一覧に対して処理
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        # 指定のWebページを読み込み
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        # for文の外のリストに読み込んだデータソースを追加
        web_docs_all.extend(web_docs)
    # 通常読み込みのデータソースにWebページのデータを追加
    docs_all.extend(web_docs_all)

    return docs_all


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ファイルの拡張子を取得
    file_extension = os.path.splitext(path)[1]
    # ファイル名（拡張子を含む）を取得
    file_name = os.path.basename(path)

    # 想定していたファイル形式の場合のみ読み込む
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()
        # CSV の場合、CSVLoader は通常行ごとに Document を返すため、
        # 社員名簿のような一覧CSVは1つの大きな Document に統合し、
        # 各行を「列名: 値」の形式で整形して検索に有利なテキストを作成する。
        if file_extension == '.csv':
            try:
                # CSV を開いて、ヘッダーと各行を取得して
                # 各行ごとに Document を作成する（検索時に複数行を返せるようにする）
                from langchain.schema import Document as LangDocument
                row_docs = []
                with open(path, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    headers = next(reader, None)
                    if headers:
                                    # CSV の全行を1つの大きなドキュメントに統合して検索精度を高める
                                    from collections import defaultdict
                                    import re
                                    merged_rows = []
                                    dept_map = defaultdict(list)
                                    for row in reader:
                                        row_dict = {h.strip(): v.strip() for h, v in zip(headers, row)}
                                        dept = row_dict.get('部署') or row_dict.get('部門') or ''
                                        name = row_dict.get('氏名（フルネーム）') or row_dict.get('氏名') or ''
                                        role = row_dict.get('役職') or row_dict.get('職位') or ''
                                        # 各行を「所属部署はXで、氏名はY、役職はZ。その他: 列名: 値、...」のような自然文にする
                                        other_pairs = [f"{h.strip()}: {v.strip()}" for h, v in zip(headers, row) if h.strip() not in ['部署', '氏名（フルネーム）', '氏名', '役職', '職位']]
                                        prefix_parts = []
                                        if dept:
                                            prefix_parts.append(f"所属部署は{dept}")
                                        if name:
                                            prefix_parts.append(f"氏名は{name}")
                                        if role:
                                            prefix_parts.append(f"役職は{role}")
                                        if other_pairs:
                                            other_text = '、'.join(other_pairs)
                                            line_text = '、'.join(prefix_parts + [f"その他: {other_text}"])
                                        else:
                                            line_text = '、'.join(prefix_parts)
                                        # 1) merged_rows に追加
                                        merged_rows.append(line_text)
                                        # 2) 部署ごとの集計にも追加
                                        if dept:
                                            dept_map[dept].append(line_text)
                                    # すべての行を改行で結合して1つの Document にする
                                    if merged_rows:
                                        content = '\n'.join(merged_rows)
                                        md = {"source": path}
                                        docs = [LangDocument(page_content=content, metadata=md)]
                                    # ▼ 追加: 部署ごとのまとめドキュメント（要約行つき）を作成
                                    for d, lines_list in dept_map.items():
                                        # 氏名だけを抽出して1行要約を作る
                                        names = []
                                        for ln in lines_list:
                                            m = re.search(r"氏名は(.+?)(、|$)", ln)
                                            if m:
                                                names.append(m.group(1))
                                        header = f"{d}の従業員一覧（{len(names)}名）: " + "、".join(names)

                                        dept_doc = LangDocument(
                                            page_content = header + "\n" + "\n".join(lines_list),
                                            metadata = {"source": path, "dept": d, "doc_type": "dept_summary"}
                                        )
                                        docs.append(dept_doc)
                # 統合ドキュメントを優先して使用する（row_docs による上書きは行わない）
            except Exception:
                # 失敗した場合は loader の返す docs を使う
                pass

        # --- ページ番号の正規化: PyMuPDFLoader 等は0始まりで page メタデータを返す場合があるため、
        #     ここで page が存在する場合は 1 加算して常に 1 始まりで保存する。
        for d in docs:
            if hasattr(d, 'metadata') and isinstance(d.metadata, dict) and 'page' in d.metadata:
                try:
                    d.metadata['page'] = int(d.metadata['page']) + 1
                except Exception:
                    # 数値変換できない場合はそのままにする
                    pass

        docs_all.extend(docs)


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s
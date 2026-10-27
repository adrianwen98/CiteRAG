import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings

from file_utils import (
    SUPPORTED_SUFFIXES,
    DocumentNoTextError,
    DocumentParseError,
)
from langchain_file_utils import load_and_split_documents

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
RETRIEVER_K = 2
SCORE_THRESHOLD = 0.5


class RetrievalError(Exception):
    pass


class ModelServiceError(Exception):
    pass


class VectorDeleteError(Exception):
    pass


class DocumentEncodingError(Exception):
    pass


class DocumentReadError(Exception):
    pass


class DocumentTextError(Exception):
    """The file is readable but has no usable text, or cannot be parsed."""


class VectorIndexError(Exception):
    pass


embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)


def build_chunk_ids(
        file_path: Path,
        file_hash: str,
        chunk_count: int,
) -> list[str]:
    """Chunk IDs must be unique per file, not just per content.

    Two files with identical bytes but different names would otherwise share
    IDs: indexing the second one overwrites the first one's metadata, and
    deleting either removes both. A stable hash of the file name keeps IDs
    deterministic while separating identical-content files.
    """
    file_key = hashlib.sha256(
        file_path.name.encode("utf-8")
    ).hexdigest()[:16]

    index_version = (
        f"{file_hash}_{file_key}_{CHUNK_SIZE}_{CHUNK_OVERLAP}"
    )

    return [
        f"{index_version}_{index}"
        for index in range(chunk_count)
    ]


llm = ChatDeepSeek(
    model="deepseek-v4-pro",
    temperature=0,
    timeout=10.0,
    max_retries=0
)


BASE_DIR = Path(__file__).resolve().parent

CHROMA_DB_DIR = BASE_DIR / "chroma_db"

vector_store = Chroma(
    collection_name="langchain_rag_documents_v2",
    embedding_function=embedding_model,
    persist_directory=str(CHROMA_DB_DIR),
    collection_configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)



def format_documents(documents: list[Document]) -> str:
    context_parts = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "未知来源")

        source_name = Path(source).name

        page = document.metadata.get("page")

        source_label = (
            f"{source_name} 第{page}页"
            if document.metadata.get("file_type") == ".pdf" and page
            else source_name
        )

        save_str = (
            f"【片段{index}|来源：{source_label}】\n"
            f"{document.page_content}"
        )

        context_parts.append(save_str)

    context_part = "\n\n".join(context_parts)

    return context_part


def index_file_langchain(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        return "文件不存在"

    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return "暂时只支持 .txt、.md、文本型 .pdf 和 .docx 文件"

    try:
        split_documents = load_and_split_documents(
            file_path=file_path,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    except UnicodeDecodeError as exc:

        raise DocumentEncodingError(
            "文件编码不是UTF-8,暂时无法读取"
        ) from exc


    except OSError as exc:

        raise DocumentReadError(
            "文件读取失败"
        ) from exc

    except (DocumentNoTextError, DocumentParseError) as exc:

        raise DocumentTextError(str(exc)) from exc

    if not split_documents:
        return "文件内容为空"

    file_hash = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    chunk_ids = build_chunk_ids(
        path,
        file_hash,
        len(split_documents),
    )

    result = vector_store.get(
        include=["metadatas"]
    )

    ids = result.get("ids") or []

    metadatas = result.get("metadatas") or []

    old_ids = []

    for record_id, metadata in zip(ids, metadatas):

        source = metadata.get(
            "source",
            ""
        )

        source_name = Path(source).name

        if source_name == path.name:
            old_ids.append(record_id)

    if set(old_ids) == set(chunk_ids):
        return "文件内容未变化，无需重复索引"

    # 先写新版本、成功后再删旧版本。原实现先删后写，一旦写入失败，
    # 旧索引已经丢失，文件也无法再被检索到。
    existing_ids = set(ids)
    added_ids = [
        chunk_id
        for chunk_id in chunk_ids
        if chunk_id not in existing_ids
    ]

    def discard_added_chunks() -> None:
        if not added_ids:
            return
        try:
            vector_store.delete(ids=added_ids)
        except Exception:
            # 清理失败不能掩盖原始写入错误；旧索引仍在。
            pass

    try:
        vector_store.add_documents(
            split_documents,
            ids=chunk_ids
        )

    except Exception as exc:
        discard_added_chunks()

        raise VectorIndexError(
            "文件索引失败"
        ) from exc

    stale_ids = [
        old_id
        for old_id in old_ids
        if old_id not in set(chunk_ids)
    ]

    if stale_ids:
        try:
            vector_store.delete(ids=stale_ids)
        except Exception as exc:
            discard_added_chunks()

            raise VectorIndexError(
                "文件索引失败"
            ) from exc

    return (
        f"文件索引成功，并写入{len(split_documents)}个片段"
    )


def get_vector_count() -> int:
    result = vector_store.get()

    return len(result["ids"])


def list_indexed_files() -> list[str]:
    result = vector_store.get(
        include=["metadatas"]
    )

    metadatas = result.get("metadatas") or []

    file_names = {
        Path(metadata.get("source", "未知来源")).name
        for metadata in metadatas
    }

    return sorted(file_names)


def delete_indexed_file(file_name: str) -> str:
    result = vector_store.get(
        include=["metadatas"]
    )

    ids = result.get("ids") or []

    metadatas = result.get("metadatas") or []

    delete_ids = []

    for record_id, metadata in zip(ids, metadatas):

        source = metadata.get(
            "source",
            ""
        )

        source_name = Path(source).name

        if source_name == file_name:
            delete_ids.append(record_id)

    if not delete_ids:
        return "知识库中不存在该文件"

    try:
        vector_store.delete(
            ids=delete_ids
        )


    except Exception as exc:

        raise VectorDeleteError(
            "向量删除失败"
        ) from exc

    return (
        f"文件删除成功，并删除{len(delete_ids)}个片段"
    )


def retrieve_relevant_documents(
        question: str,
        score_threshold: float = SCORE_THRESHOLD,
) -> list[Document]:

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": RETRIEVER_K,
            "score_threshold": score_threshold
        }
    )

    retriever_results = retriever.invoke(question)

    return retriever_results


def answer_from_knowledge_base(
        question: str
) -> str:

    from retrieval_pipeline import retrieve_final_documents

    if question.strip() == "":

        return "问题不能为空"


    try:

        retriever_results = retrieve_final_documents(
            question
        )


    except Exception as exc:

        raise RetrievalError(
            "向量检索失败"
        ) from exc

    if not retriever_results:

        return "没有找到与问题相关的资料"

    context = format_documents(
        retriever_results
    )


    try:

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个严谨的文件问答助手，只能根据参考资料回答。"
                    "如果资料不足，请明确说明无法回答。"
                    "回答中的结论必须标注完整的片段标签，"
                    "例如【片段1｜来源：tests.txt】或"
                    "【片段1｜来源：manual.pdf 第2页】。"
                ),
                (
                    "human",
                    "参考资料: \n{context}\n\n用户问题:\n{question}"
                )
            ]
        )

        prompt_llm_chain = (
                prompt_template
                | llm
                | StrOutputParser()
        )

        response = prompt_llm_chain.invoke(
            {
                "context": context,
                "question": question
             }
        )

    except Exception as exc:

        print(type(exc).__name__, repr(exc))

        raise ModelServiceError(
            "模型服务调用失败"
        ) from exc

    return str(response)

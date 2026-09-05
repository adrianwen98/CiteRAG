import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from langchain_rag_service import vector_store


def retrieve_bm25_documents(
        question: str,
        k: int = 2
) -> list[Document]:

    data = vector_store.get(
        include=["documents", "metadatas"]
    )

    texts = data["documents"]
    metadatas = data["metadatas"]
    ids = data["ids"]

    words_documents = []

    for text in texts:
        words = list(jieba.cut(text))
        words_documents.append(words)

    bm25 = BM25Okapi(words_documents)

    question_word = list(jieba.cut(question))

    scores = bm25.get_scores(question_word)

    score_store = []

    for index, score in enumerate(scores):
        if score > 0:
            score_store.append((index, score))

    score_store.sort(
        key=lambda item: item[1],
        reverse=True
    )

    top_results = score_store[:k]

    bm25_documents = []

    for doc_index, score in top_results:
        doc_id = ids[doc_index]
        doc_page = texts[doc_index]
        doc_meta = metadatas[doc_index]

        document = Document(
            id=doc_id,
            page_content=doc_page,
            metadata=doc_meta
        )

        bm25_documents.append(document)

    return bm25_documents
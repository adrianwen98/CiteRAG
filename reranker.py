from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder("BAAI/bge-reranker-base")


def rerank_documents(question: str, documents: list[Document]):
    pairs = []

    for document in documents:
        page = document.page_content
        pairs.append((question, page))

    result = reranker_model.predict(pairs)

    scored_candidates = []

    for index, score in enumerate(result):
        candidate = documents[index]
        scored_candidates.append((candidate, score))

    scored_candidates.sort(
        key=lambda item: item[1],
        reverse=True
    )

    final_documents = []

    for candidate_2, score_2 in scored_candidates:
        final_documents.append(candidate_2)

    return final_documents
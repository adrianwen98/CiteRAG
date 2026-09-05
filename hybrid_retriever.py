from langchain_rag_service import retrieve_relevant_documents
from bm25_retriever import retrieve_bm25_documents


def retrieve_hybrid_documents(question: str):
    dense_documents = retrieve_relevant_documents(question)
    bm25_documents = retrieve_bm25_documents(question)

    rrf_scores = {}

    for rank, document in enumerate(dense_documents, start=1):
        document_id = document.id
        rank_result = 1 / (60 + rank)

        rrf_scores[document_id] = rrf_scores.get(document_id, 0) + rank_result


    for rank, document in enumerate(bm25_documents, start=1):
        document_id = document.id
        rank_result = 1 / (60 + rank)

        rrf_scores[document_id] = rrf_scores.get(document_id, 0) + rank_result


    sorted_rrf = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    documents_store = {}

    for document in dense_documents + bm25_documents:
        document_id = document.id

        documents_store[document_id] = document

    hybrid_documents = []

    for documid, rrfsco in sorted_rrf:
        document = documents_store[documid]

        hybrid_documents.append(document)

    return hybrid_documents

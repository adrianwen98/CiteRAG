from langchain_core.documents import Document

from hybrid_retriever import retrieve_hybrid_documents
from reranker import rerank_documents

def retrieve_final_documents(question: str) -> list[Document]:
    hybrid_result = retrieve_hybrid_documents(question)
    rerank_result = rerank_documents(question, hybrid_result)

    return rerank_result
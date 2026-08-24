from langchain_rag_service import retrieve_relevant_documents
from langchain_core.documents import Document
from langchain_file_utils import load_and_split_documents


result_40: list[Document] = load_and_split_documents(
    file_path="../data/test.txt",
    chunk_size=40,
    chunk_overlap=8
)

print("chunk_size 40 的片段总数：", len(result_40))
print("第一个片段正文：", result_40[0].page_content)
print("第二个片段正文：", result_40[1].page_content)


result_80: list[Document] = load_and_split_documents(
    file_path="../data/test.txt",
    chunk_size=80,
    chunk_overlap=8
)

print("chunk_size 80 的片段总数：", len(result_80))
print("第一个片段正文：", result_80[0].page_content)
print("第二个片段正文：", result_80[1].page_content)


search_results = retrieve_relevant_documents(
    "企业智能知识库系统是如何工作的？"
)


print("\n检索结果：")

for doc in search_results:
    print("----------------")
    print(doc.page_content)
    print(doc.metadata)
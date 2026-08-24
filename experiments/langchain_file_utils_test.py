from langchain_file_utils import load_and_split_documents


split_documents = load_and_split_documents(
    "../data/test.txt",
    chunk_size=40,
    chunk_overlap=8
)


print("切分后的片段数量：", len(split_documents))

print("\n第一个片段内容：")
print(split_documents[0].page_content)

print("\n第一个片段metadata：")
print(split_documents[0].metadata)

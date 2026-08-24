from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from file_utils import read_file_content


content = read_file_content("../data/test.txt")


documents = [
    Document(
        page_content=content,
        metadata={
            "source": "test.txt",
            "topic": "企业知识库",
            "file_type": "txt"
        }
    )
]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=40,
    chunk_overlap=8
)

split_documents = text_splitter.split_documents(documents)


for document in split_documents:
    print()
    print(document.page_content)
    print(document.metadata)


print(len(split_documents))
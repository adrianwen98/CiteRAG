from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from file_utils import read_file_content

def load_and_split_documents(
        file_path: str,
        chunk_size: int,
        chunk_overlap: int,
) -> list[Document]:


    content = read_file_content(file_path)

    file_type = Path(file_path).suffix.lower()

    documents = [
        Document(
            page_content=content,
            metadata={
                "source": file_path,
                "file_type": file_type
            }
        )
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_documents = text_splitter.split_documents(documents)

    return split_documents

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from file_utils import read_document_pages


def load_and_split_documents(
        file_path: str,
        chunk_size: int,
        chunk_overlap: int,
) -> list[Document]:

    file_type = Path(file_path).suffix.lower()

    documents = [
        Document(
            page_content=content,
            metadata={
                "source": file_path,
                "file_type": file_type,
                **({"page": page_number} if page_number is not None else {}),
            },
        )
        for page_number, content in read_document_pages(file_path)
        if content.strip()
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return text_splitter.split_documents(documents)

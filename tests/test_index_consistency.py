from pathlib import Path

import pytest
from langchain_core.documents import Document

import bm25_retriever
import langchain_rag_service
from langchain_rag_service import VectorIndexError, build_chunk_ids


class FakeVectorStore:
    """记录调用顺序，用于验证“先写后删”的更新顺序。"""

    def __init__(self, ids=None, metadatas=None, fail_add=False, fail_delete=False):
        self.ids = list(ids or [])
        self.metadatas = list(metadatas or [])
        self.calls = []
        self.deleted_ids = []
        self.fail_add = fail_add
        self.fail_delete = fail_delete

    def get(self, include=None):
        self.calls.append(("get",))
        return {"ids": list(self.ids), "metadatas": list(self.metadatas)}

    def add_documents(self, documents, ids):
        self.calls.append(("add", list(ids)))
        if self.fail_add:
            raise RuntimeError("embedding failed")
        self.ids.extend(ids)
        self.metadatas.extend(document.metadata for document in documents)

    def delete(self, ids):
        self.calls.append(("delete", list(ids)))
        if self.fail_delete:
            raise RuntimeError("delete failed")
        self.deleted_ids.extend(ids)


def _prepare_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "doc.txt"
    file_path.write_text("新的内容", encoding="utf-8")
    return file_path


def _prepare_store(file_path: Path, fail_add=False, fail_delete=False):
    return FakeVectorStore(
        ids=["legacy_0"],
        metadatas=[{"source": str(file_path)}],
        fail_add=fail_add,
        fail_delete=fail_delete,
    )


def test_chunk_ids_include_file_identity():
    ids_a = build_chunk_ids(Path("A.txt"), "same-content-hash", 2)
    ids_b = build_chunk_ids(Path("B.txt"), "same-content-hash", 2)

    assert set(ids_a).isdisjoint(ids_b)
    assert ids_a == build_chunk_ids(Path("A.txt"), "same-content-hash", 2)


def test_reindex_writes_new_chunks_before_deleting_old(tmp_path, monkeypatch):
    file_path = _prepare_file(tmp_path)
    documents = [
        Document(page_content="新的内容", metadata={"source": str(file_path)})
    ]
    store = _prepare_store(file_path)

    monkeypatch.setattr(
        langchain_rag_service,
        "load_and_split_documents",
        lambda **kwargs: documents,
    )
    monkeypatch.setattr(langchain_rag_service, "vector_store", store)

    message = langchain_rag_service.index_file_langchain(str(file_path))

    assert "索引成功" in message
    assert [call[0] for call in store.calls] == ["get", "add", "delete"]
    assert store.deleted_ids == ["legacy_0"]


def test_failed_reindex_keeps_old_vectors(tmp_path, monkeypatch):
    file_path = _prepare_file(tmp_path)
    documents = [
        Document(page_content="新的内容", metadata={"source": str(file_path)})
    ]
    store = _prepare_store(file_path, fail_add=True)

    monkeypatch.setattr(
        langchain_rag_service,
        "load_and_split_documents",
        lambda **kwargs: documents,
    )
    monkeypatch.setattr(langchain_rag_service, "vector_store", store)

    with pytest.raises(VectorIndexError):
        langchain_rag_service.index_file_langchain(str(file_path))

    assert "legacy_0" not in store.deleted_ids


def test_failed_cleanup_after_add_keeps_old_vectors(tmp_path, monkeypatch):
    file_path = _prepare_file(tmp_path)
    documents = [
        Document(page_content="新的内容", metadata={"source": str(file_path)})
    ]
    store = _prepare_store(file_path, fail_delete=True)

    monkeypatch.setattr(
        langchain_rag_service,
        "load_and_split_documents",
        lambda **kwargs: documents,
    )
    monkeypatch.setattr(langchain_rag_service, "vector_store", store)

    with pytest.raises(VectorIndexError):
        langchain_rag_service.index_file_langchain(str(file_path))

    assert "legacy_0" not in store.deleted_ids


def test_bm25_returns_empty_for_empty_knowledge_base(monkeypatch):
    class EmptyStore:
        def get(self, include=None):
            return {"ids": [], "documents": [], "metadatas": []}

    monkeypatch.setattr(bm25_retriever, "vector_store", EmptyStore())

    assert bm25_retriever.retrieve_bm25_documents("任意问题") == []

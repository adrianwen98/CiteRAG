from fastapi.testclient import TestClient
from unittest.mock import patch
from pathlib import Path
from api_main import app
from langchain_rag_service import ModelServiceError

client = TestClient(app)

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_stats():
    response = client.get("/stats")

    assert response.status_code == 200
    data = response.json()

    assert "vector_count" in data
    assert isinstance(data["vector_count"], int)
    assert data["vector_count"] >= 0

def test_documents():
    response = client.get("/documents")

    assert response.status_code == 200

    data = response.json()

    assert "files" in data

    assert isinstance(data["files"], list)



def test_documents_keyword():

    client.post(
        "/documents/index",
        json={
            "file_path": "data/test_2.txt"
        }
    )

    response = client.get(
        "/documents",
        params={"keyword": "test_2"}
    )

    assert response.status_code == 200

    data = response.json()

    assert "files" in data

    assert isinstance(data["files"], list)

    assert "test_2.txt" in data["files"]

def test_documents_keyword_too_long():
    too_long_keyword = "a" * 51
    response = client.get(
        "/documents",
        params={"keyword": too_long_keyword}
    )
    assert response.status_code == 422



def test_delete_document_not_found():
    response = client.delete(
        "/documents/not_exist_123456.txt"
    )

    data = response.json()

    assert response.status_code == 404

    assert "detail" in data

    assert data["detail"] == "文件不存在"

def test_upload_document_conflict():

    # 测试前清理，保证环境一致
    client.delete("/documents/tests.txt")

    files = {
        "file": (
            "tests.txt",
            b"tests content",
            "text/plain"
        )
    }

    response1 = client.post(
        "/documents/upload",
        files=files
    )

    assert response1.status_code == 200

    response2 = client.post(
        "/documents/upload",
        files=files
    )

    data = response2.json()

    assert response2.status_code == 409

    assert "detail" in data

    assert data["detail"] == "同名文件已存在"

    # 测试后清理
    client.delete("/documents/tests.txt")



def test_ask_success():
    with patch(
        "api_main.answer_from_knowledge_base",
        return_value = "测试答案"
    ):
        response = client.post(
            "ask",
            json={"question": "测试问题"}
        )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data

    assert data["answer"] == "测试答案"

def test_ask_model_error():
    with patch(
        "api_main.answer_from_knowledge_base",
        side_effect=ModelServiceError("模型服务调用失败")
    ):
        response = client.post(
            "/ask",
            json={"question": "测试问题"}
        )

    assert response.status_code == 502

    data = response.json()

    assert data["detail"] == "模型服务调用失败"



def test_index_document_success():
    with patch(
        "api_main.index_file_langchain",
        return_value="文件索引成功，并写入1个片段"
    ):
        response = client.post(
            "/documents/index",
            json={
                "file_path": "data/test.txt"
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert "message" in data

    assert data["message"] == "文件索引成功，并写入1个片段"
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query,
    Path as PathParam
)
from pydantic import BaseModel, Field
from pathlib import Path
from langchain_rag_service import (
    get_vector_count,
    list_indexed_files,
    answer_from_knowledge_base,
    delete_indexed_file,
    index_file_langchain,
    RetrievalError,
    ModelServiceError,
    VectorDeleteError,
    DocumentReadError,
    DocumentEncodingError,
    VectorIndexError
)

# 创建FastAPI应用对象
app = FastAPI(
    title="RAG Knowledge Base API"
)

DATA_DIR = Path(__file__).resolve().parent / "data"

MAX_FILE_SIZE = 2 * 1024 * 1024

class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=500,
        description="要向知识库提出的问题"
    )

class AskResponse(BaseModel):
    answer: str

class VectorCountResponse(BaseModel):
    vector_count: int

class IndexRequest(BaseModel):
    file_path: str

class IndexResponse(BaseModel):
    message: str

class DeleteResponse(BaseModel):
    message: str
    local_file_deleted: bool

class DocumentResponse(BaseModel):
    files: list[str]

class UploadResponse(BaseModel):
    file_name: str
    message: str


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={
        400:{
            "description": "问题不能为空"
        },
        500:{
            "description": "向量检索失败"
        },
        502:{
            "description": "模型服务调用失败"
        }
    }
)

def ask_question(request: AskRequest):
    clean_question = request.question.strip()

    if not clean_question:
        raise HTTPException(
            status_code=400,
            detail="问题不能为空"
        )

    try:
        answer = answer_from_knowledge_base(clean_question)

    except RetrievalError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

    except ModelServiceError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc)
        )

    return {"answer": answer}

@app.get(
    "/stats",
    response_model=VectorCountResponse
)
def get_stats():
    vector_count = get_vector_count()
    return {
        "vector_count": vector_count
    }



@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }

@app.get(
    "/documents",
    response_model=DocumentResponse
)
def get_documents(
        keyword: str | None = Query(
            default=None,
            min_length=1,
            max_length=50,
            description="按照文件名关键词筛选"
        )
):
    files = list_indexed_files()

    if keyword is not None:
        clean_keyword = keyword.lower()

        files = [
            file_name
            for file_name in files
            if clean_keyword in file_name.lower()
        ]

    return {
        "files": files
    }

@app.delete(
    "/documents/{file_name}",
    response_model=DeleteResponse,
    responses={
        404: {
            "description":"要删除的文件不存在"
        },
        500:{
            "description":"向量删除失败"
        }
    }
)
def delete_document(
        file_name: str = PathParam(
            min_length=1,
            max_length=255,
            description="要删除的知识库文件名"
        )
):
    safe_file_name = Path(file_name).name

    local_file_path = DATA_DIR / safe_file_name

    local_file_exists = local_file_path.exists()

    try:
        vector_result = delete_indexed_file(safe_file_name)

    except VectorDeleteError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

    vector_file_found = (
        vector_result != "知识库中不存在该文件"
    )

    if not vector_file_found and not local_file_exists:
        raise HTTPException(
            status_code=404,
            detail="文件不存在"
        )

    local_file_deleted = False

    if local_file_exists:
        local_file_path.unlink()
        local_file_deleted = True

    return {
        "message": vector_result,
        "local_file_deleted": local_file_deleted
    }

@app.post(
    "/documents/index",
    response_model=IndexResponse,
    responses={
        404:{
            "description": "传入的文件不存在"
        },
        400:{
            "description": "文件格式不支持"
        },
        403:{
            "description": "禁止访问data文件夹以外的文件"
        }
    }
)
def index_document(request: IndexRequest) -> dict[str, str]:
    data_dir_resolved = DATA_DIR.resolve()
    file_path = Path(request.file_path).resolve()

    if not file_path.is_relative_to(data_dir_resolved):
        raise HTTPException(
            status_code=403,
            detail="只能索引data中的文件"
        )

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="传入的文件不存在"
        )

    if file_path.suffix.lower() != ".txt":
        raise HTTPException(
            status_code=400,
            detail="目前只支持索引 .txt 文件"
        )

    try:
        result = index_file_langchain(
            file_path=str(file_path)
        )

    except DocumentEncodingError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except DocumentReadError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

    except VectorIndexError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

    if result == "文件内容为空":
        raise HTTPException(
            status_code=400,
            detail=result
        )

    return {
        "message": result
    }

@app.post(
    "/documents/upload",
    response_model=UploadResponse,
    responses={
        400:{
            "description": "文件格式错误或文件内容为空"
        },
        409:{
            "description": "同名文件已存在"
        },
        413:{
            "description": "上传文件超过2MB"
        }
    }
)
async def upload_document(file:UploadFile = File(...)):
    file_name = Path(file.filename or "").name

    if Path(file_name).suffix.lower() != ".txt":
        raise HTTPException(status_code=400,detail="目前只支持上传.txt文件")

    save_path = DATA_DIR / file_name

    if save_path.exists():
        raise HTTPException(
            status_code=409,
            detail="同名文件已存在"
        )

    file_content = await file.read()

    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="上传文件不能超过2MB"
        )

    if not file_content.strip():
        raise HTTPException(status_code=400,detail="不能上传空文件")

    save_path.write_bytes(file_content)

    try:
        result = index_file_langchain(file_path=str(save_path))

    except DocumentEncodingError as exc:
        if save_path.exists():
            save_path.unlink()

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except DocumentReadError as exc:
        if save_path.exists():
            save_path.unlink()

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except VectorIndexError as exc:
        if save_path.exists():
            save_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

    return {
        "file_name": file_name,
        "message": result
    }
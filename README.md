# 基于 LangChain + Chroma + FastAPI 的 RAG 知识库问答系统

## 项目简介

本项目实现了一个基于检索增强生成（Retrieval-Augmented Generation, RAG）的知识库问答系统。

系统支持用户上传本地 txt 文档，通过文本切分、Embedding 向量化、向量检索，将相关知识片段与用户问题结合后交由大语言模型生成回答。

项目采用 FastAPI 提供 RESTful API 服务，LangChain 负责 RAG 流程编排，Chroma 作为向量数据库，实现了从文档索引、语义检索到大语言模型生成回答的完整流程。

同时支持文档上传、删除、重复索引检测、知识库查询以及 API 自动化测试。


## 技术栈

- Python
- FastAPI
- LangChain
- ChromaDB
- HuggingFace Embedding
- DeepSeek API
- Pytest


## 项目结构

```text
RAG_Write
│
├── api_main.py                  # FastAPI接口层
├── langchain_rag_service.py     # RAG业务逻辑层
├── langchain_file_utils.py      # 文档加载与切分
├── file_utils.py                # 文件读取工具
│
├── data/                        # 示例知识库文档
│
├── experiments/                 # LangChain实验代码
│
├── tests/                       # API自动化测试
│   └── test_api.py
│
├── requirements.txt
└── README.md
```


## 系统架构

本项目采用分层架构设计，将 API 接口层、RAG 业务逻辑层、文档处理层以及向量数据库进行解耦。

整体架构如下：

```text
用户请求
    ↓
FastAPI API 层（api_main.py）
    ↓
RAG 业务层（langchain_rag_service.py）
    ↓
├── 文件读取（file_utils.py）
├── 文档加载与切分（langchain_file_utils.py）
├── 向量生成（HuggingFace Embedding）
├── 向量存储与检索（ChromaDB）
└── 大语言模型生成（DeepSeek）
```


## RAG 工作流程

系统主要包含 Indexing、Retrieval、Generation 三个阶段。


### 1. Indexing（索引阶段）

用户上传文档后：

1. 读取文件内容
2. 使用 RecursiveCharacterTextSplitter 对文本进行切分
3. 将文本片段转换为 Document 对象
4. 使用 HuggingFace Embedding 模型生成向量
5. 将向量及 metadata 存入 Chroma 向量数据库


流程：

```text
文档
↓
文本切分
↓
Document
↓
Embedding
↓
Chroma Vector Store
```


### 2. Retrieval（检索阶段）

用户提出问题后：

1. 对用户问题进行向量化
2. 在 Chroma 中进行相似度搜索
3. 返回相关文档片段


流程：

```text
用户问题
↓
Query Embedding
↓
Chroma Similarity Search
↓
相关文本片段
```


### 3. Generation（生成阶段）

系统将：

- 用户问题
- 检索到的知识片段

共同发送给大语言模型，由模型生成最终回答。


流程：

```text
Question + Retrieved Context
↓
Prompt Template
↓
DeepSeek LLM
↓
Answer
```


## 核心功能


### 1. 文档索引

支持将本地 txt 文档加入知识库：

- 文档读取
- 文本切分
- 向量化
- Chroma 持久化存储


### 2. 知识库检索

支持：

- 文件列表查询
- 关键词筛选
- 向量相似度检索


### 3. RAG 问答

用户输入问题后：

- 自动检索相关知识片段
- 基于检索结果生成回答
- 返回来源片段信息


### 4. 文件管理

支持：

- 上传文件
- 删除文件
- 重复文件检测
- 索引状态管理


### 5. 接口测试

使用 Pytest 对 API 进行自动化测试，覆盖：

- 正常请求
- 参数校验失败
- 文件不存在
- 文件冲突
- 模型调用异常


# API 接口

项目通过 FastAPI 提供 RESTful API 服务。


## 1. 健康检查

GET `/health`

用于检查服务运行状态。

返回示例：

```json
{
  "status": "ok"
}
```


---

## 2. 查看向量库状态

GET `/stats`

返回当前 Chroma 向量数据库中的向量数量。


返回示例：

```json
{
  "vector_count": 36
}
```

> vector_count 会根据当前知识库中的文档数量动态变化。


---

## 3. 查询已索引文件

GET `/documents`

用于查看当前知识库中已经索引的文件。

支持关键词筛选：

示例：

```text
GET /documents?keyword=test
```


返回示例：

```json
{
  "files": [
    "test.txt",
    "test_2.txt"
  ]
}
```


---

## 4. 文档索引

POST `/documents/index`

将指定 txt 文件加入知识库。


请求：

```json
{
  "file_path": "data/test.txt"
}
```


处理流程：

```text
文件读取
↓
文本切分
↓
Embedding向量生成
↓
Chroma向量存储
```


---

## 5. 文件上传

POST `/documents/upload`

上传新的 txt 文件，并自动完成索引。


支持：

- 文件类型校验
- 文件大小限制
- 同名文件检测
- 自动建立向量索引


---

## 6. 删除文件

DELETE `/documents/{file_name}`

删除指定文件，同时删除 Chroma 中对应的向量数据。


---

## 7. 知识库问答

POST `/ask`

根据知识库内容回答用户问题。


请求：

```json
{
  "question": "企业智能知识库系统是如何工作的？"
}
```


返回：

```json
{
  "answer": "企业智能知识库系统通过文档管理、向量检索和大语言模型实现智能问答。"
}
```


# 项目启动


## 1. 安装依赖

```bash
pip install -r requirements.txt
```


## 2. 启动服务

```bash
uvicorn api_main:app --host 0.0.0.0 --port 8002
```


启动成功后访问：

```text
http://127.0.0.1:8002/docs
```


通过 Swagger UI 查看和测试接口。


# 自动化测试

项目使用 Pytest 对 FastAPI 接口进行自动化测试。


运行：

```bash
pytest tests/test_api.py -v
```


当前测试覆盖：

- 健康检查接口
- 向量库状态查询
- 文件列表查询
- Query 参数校验
- 文件不存在处理
- 文件冲突处理
- 模型服务异常处理
- 文档索引接口


测试结果：

```text
10 passed
```


# 项目亮点

- 基于 LangChain 构建完整 RAG 流程，实现文档索引、向量检索和大语言模型生成。

- 使用 FastAPI 将 RAG 能力服务化，提供文档管理和知识库问答接口。

- 使用 Chroma 作为向量数据库，通过 metadata 保存文档来源，实现回答溯源。

- 使用 HuggingFace Embedding 进行文本向量化，并支持基于语义的相似度检索。

- 使用 Pytest 对 API 接口进行自动化测试，覆盖正常流程、参数异常、资源不存在以及模型服务异常等场景。


# 项目效果展示


## Swagger API 文档

![Swagger API](docs/images/swagger.jpg)


## 文档索引

![Document Index](docs/images/index.jpg)


## RAG 问答效果

![RAG Answer](docs/images/rag_answer.jpg)
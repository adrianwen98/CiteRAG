from langchain_huggingface import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


question = "企业知识库系统如何实现智能问答？"


query_embedding = embedding_model.embed_query(question)


print(type(query_embedding))
print(len(query_embedding))
print(query_embedding[:5])
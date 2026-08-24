from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
from langchain_file_utils import load_and_split_documents


llm = ChatDeepSeek(
    model="deepseek-v4-pro",
    temperature=0,
    timeout=10.0,
    max_retries=0
)


question = "企业智能知识库系统是如何工作的？"


documents = load_and_split_documents(
    file_path="../data/test.txt",
    chunk_size=300,
    chunk_overlap=50
)


def format_documents(documents: list[Document]) -> str:

    context_parts = []

    for index, document in enumerate(documents, start=1):
        save_str = f"【片段{index}】\n{document.page_content}"
        context_parts.append(save_str)

    return "\n\n".join(context_parts)


context = format_documents(documents)


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个严谨的文件问答助手，只能根据参考资料回答。"
            "如果资料不足，请明确说明无法回答。"
            "回答中的结论必须标注对应的片段编号"
        ),
        (
            "human",
            "参考资料:\n{context}\n\n用户问题:\n{question}"
        )
    ]
)


prompt_llm_chain = prompt_template | llm

response = prompt_llm_chain.invoke(
    {
        "context": context,
        "question": question
    }
)

print(response.content)
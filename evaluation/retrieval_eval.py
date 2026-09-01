from pathlib import Path

from langchain_rag_service import retrieve_relevant_documents, vector_store

eval_cases = [
    {
        "question": "新员工入职第一天需要在人事系统中填写哪些信息？",
        "expected_source": "test.txt"
    },
    {
        "question": "员工创建基础账号后还可以申请哪些系统权限？",
        "expected_source": "test.txt"
    },
    {
        "question": "企业文件管理员需要如何维护已有文档？",
        "expected_source": "test.txt"
    },
    {
        "question": "向量数据库在知识库系统中负责保存什么？",
        "expected_source": "test_2.txt"
    },
    {
        "question": "为什么文本片段需要转换成向量？",
        "expected_source": "test_2.txt"
    },
    {
        "question": "RAG 系统检索到相关内容以后会怎样处理？",
        "expected_source": "test_2.txt"
    },
    {
        "question": "公司的年终奖发几个月？",
        "expected_source": None
    },
    {
        "question": "员工每年有多少天带薪休假？",
        "expected_source": None
    },
    {
        "question": "公司的食堂几点关门？",
        "expected_source": None
    }
]

thresholds = [0.50, 0.45, 0.40, 0.35]

for threshold in thresholds:
    hit_count = 0

    for index, case in enumerate(eval_cases, start=1):

        question = case["question"]

        documents = retrieve_relevant_documents(question, threshold)

        expected_source = case["expected_source"]

        sources = []

        for document in documents:

            source = document.metadata["source"]

            file_name = Path(source).name.lower()

            sources.append(file_name)

        if expected_source is None:
            hit = len(sources) == 0

        else:
            hit = expected_source.lower() in sources

        print(f"[{index}] 问题：{question}")
        print(f"预期来源：{expected_source}")
        print(f"实际来源：{sources}")

        if hit:
            print("结果：PASS")
        else:
            print("结果：FAIL")

        if not hit:
            score_result = vector_store.similarity_search_with_relevance_scores(question, k=2)

            for document, score in score_result:
                document_result = Path(document.metadata["source"]).name
                print(f"候选来源：{document_result}, 相关性分数：{score}")

        if hit:
            hit_count += 1

    hit_rate = hit_count / len(eval_cases)

    print(f"命中数：{hit_count} / {len(eval_cases)}")
    print(f"Hit Rate@2：{hit_rate:.2%}")
    print(f"Threshold: {threshold}")




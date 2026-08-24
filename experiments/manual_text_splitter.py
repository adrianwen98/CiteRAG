def split_text(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
    chunk_content = []

    if chunk_size <= 0:
        raise ValueError("\n\n您当前的chunk_size不符合运行条件，chunk_size 必须大于 0")

    elif chunk_overlap >= chunk_size or chunk_overlap < 0:
        raise ValueError("\n\nchunk_overlap 必须大于等于 0，并且小于 chunk_size")

    start_position = 0

    while start_position < len(text):
        terminate_position = start_position + chunk_size

        chunk = text[start_position:terminate_position]

        chunk_content.append(chunk)

        if terminate_position >= len(text):
            break

        start_position += chunk_size - chunk_overlap

    return chunk_content

if __name__ == "__main__":

    text = "这是一个文本切分测试示例，用于验证chunk_size和chunk_overlap参数对文本切分结果的影响。"

    chunks = split_text(
        text=text,
        chunk_size=10,
        chunk_overlap=3
    )

    print("切分后的片段数量：", len(chunks))

    for index, chunk in enumerate(chunks):
        print(f"\n--- chunk {index} ---")
        print(chunk)
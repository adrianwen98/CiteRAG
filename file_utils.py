def read_file_content(file_path: str) -> str:

    with open(
        file_path,
        "r",
        encoding="UTF-8"
    ) as f:
        content = f.read()

    return content


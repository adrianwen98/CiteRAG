from pathlib import Path

import pymupdf


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


class UnsupportedDocumentError(ValueError):
    """The file extension is not supported."""


class DocumentNoTextError(ValueError):
    """A PDF has no extractable text layer (for example, a scanned PDF)."""


class DocumentParseError(ValueError):
    """The document is encrypted, corrupted, or otherwise unreadable."""


def read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="UTF-8") as file:
        return file.read()


def read_document_pages(file_path: str) -> list[tuple[int | None, str]]:
    """Return (page_number, text) pairs. Page numbers are 1-based for PDFs."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return [(None, read_text_file(file_path))]

    if suffix != ".pdf":
        raise UnsupportedDocumentError(
            "暂时只支持 .txt、.md 和文本型 .pdf 文件"
        )

    try:
        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise DocumentParseError("PDF 已加密，暂时无法读取")
            pages = []
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text")
                if text.strip():
                    pages.append((page_number, text))
            if not pages:
                raise DocumentNoTextError(
                    "PDF 没有可提取的文字层，可能是扫描件；暂不做 OCR，请上传文本型 PDF"
                )
            return pages
    except (DocumentNoTextError, DocumentParseError):
        raise
    except Exception as exc:
        raise DocumentParseError("PDF 读取失败，文件可能损坏") from exc

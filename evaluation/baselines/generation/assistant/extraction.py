from __future__ import annotations

import io
import unicodedata
from importlib.metadata import version
from pathlib import Path

from .schemas import Page, Paper


def normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in unicodedata.normalize("NFC", text).replace("\r\n", "\n").splitlines()).strip()


def extract_pages(paper: Paper, extractor="pypdf") -> list[Page]:
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(Path(paper.pdf_path))
    converter = None
    if extractor == "markitdown":
        from markitdown import MarkItDown
        converter = MarkItDown()
    elif extractor != "pypdf":
        raise ValueError("Unknown extractor")
    identity = f"{extractor}:{version(extractor)}:nfc-v1"
    pages = []
    for position, source in enumerate(reader.pages, 1):
        try:
            if converter is None:
                raw = source.extract_text() or ""
            else:
                writer = PdfWriter()
                writer.add_page(source)
                stream = io.BytesIO()
                writer.write(stream)
                stream.seek(0)
                result = converter.convert_stream(stream, file_extension=".pdf")
                raw = result.markdown
            text = normalize(raw)
            pages.append(Page(paper.paper_id, position, text, raw, identity,
                              "ready" if text else "empty_or_image_only"))
        except Exception as exc:
            pages.append(Page(paper.paper_id, position, "", "", identity, "failed", str(exc)))
    return pages

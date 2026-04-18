from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Callable, Iterable, List

from pypdf import PdfReader


@dataclass
class DocumentRecord:
    name: str
    page_count: int
    approx_words: int


_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Light cleanup for PDF-extracted text.

    pypdf recommends post-processing after extraction; this keeps the cleanup
    conservative to avoid altering authorial wording too aggressively.
    """
    text = text.replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def process_pdf_in_chunks(
    uploaded_file,
    chunk_size: int,
    overlap: int,
    on_chunk: Callable[[str, int, int], None],
) -> DocumentRecord:
    """Process one PDF at a time and stream text chunks to a callback."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    raw = uploaded_file.read()
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    reader = PdfReader(io.BytesIO(raw), strict=False)
    page_count = len(reader.pages)
    approx_words = 0

    chunk_parts: List[str] = []
    chunk_len = 0
    chunk_start_page = 1
    last_page_with_text = 1

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = normalize_text(page.extract_text() or "")
        if not page_text:
            continue

        approx_words += len(page_text.split())
        page_block = f"[Página {page_number}]\n{page_text}"
        page_block_len = len(page_block) + 2

        if chunk_parts and chunk_len + page_block_len > chunk_size:
            chunk_text = "\n\n".join(chunk_parts).strip()
            if chunk_text:
                on_chunk(chunk_text, chunk_start_page, last_page_with_text)

            overlap_seed = chunk_text[-overlap:] if overlap > 0 else ""
            chunk_parts = [overlap_seed] if overlap_seed else []
            chunk_len = len(overlap_seed) + 2 if overlap_seed else 0
            chunk_start_page = page_number

        if not chunk_parts:
            chunk_start_page = page_number

        chunk_parts.append(page_block)
        chunk_len += page_block_len
        last_page_with_text = page_number

    if chunk_parts:
        chunk_text = "\n\n".join(chunk_parts).strip()
        if chunk_text:
            on_chunk(chunk_text, chunk_start_page, last_page_with_text)

    return DocumentRecord(
        name=getattr(uploaded_file, "name", "document.pdf"),
        page_count=page_count,
        approx_words=approx_words,
    )


def join_doc_catalog(documents: Iterable[DocumentRecord]) -> str:
    lines = []
    for idx, doc in enumerate(documents, start=1):
        lines.append(
            f"[{idx}] {doc.name} | pages={doc.page_count} | approx_words={doc.approx_words}"
        )
    return "\n".join(lines)

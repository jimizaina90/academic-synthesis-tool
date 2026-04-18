from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from llm_clients import LLMConfig, get_client
from pdf_utils import DocumentRecord, join_doc_catalog, process_pdf_in_chunks
from workflows import Discipline, Task, build_prompt


@dataclass
class StageResult:
    title: str
    content: str


MAP_REDUCE_TASKS = {"mapear_corpus", "ficha_por_fonte"}


def _summarize_pdf_incremental(
    uploaded_file,
    discipline: Discipline,
    user_goal: str,
    llm_config: LLMConfig,
    chunk_size: int,
    overlap: int,
) -> Tuple[DocumentRecord, StageResult]:
    client = get_client(llm_config)
    partials: List[Tuple[int, int, int, str]] = []
    doc_name = getattr(uploaded_file, "name", "document.pdf")

    def on_chunk(chunk_text: str, page_start: int, page_end: int) -> None:
        idx = len(partials) + 1
        bundle = build_prompt(
            discipline=discipline,
            task="ficha_por_fonte",
            corpus_catalog=f"[1] {doc_name} | bloco {idx} | páginas {page_start}-{page_end}",
            content=chunk_text,
            user_goal=user_goal,
        )
        partial = client.generate(bundle.system, bundle.user)
        partials.append((idx, page_start, page_end, partial))

    doc_record = process_pdf_in_chunks(
        uploaded_file=uploaded_file,
        chunk_size=chunk_size,
        overlap=overlap,
        on_chunk=on_chunk,
    )

    if not partials:
        return doc_record, StageResult(
            title=doc_record.name,
            content="Documento vazio ou sem texto extraível.",
        )

    if len(partials) == 1:
        return doc_record, StageResult(title=doc_record.name, content=partials[0][3])

    reduce_sections = []
    for idx, page_start, page_end, partial in partials:
        reduce_sections.append(
            f"## Bloco {idx} (páginas {page_start}-{page_end})\n{partial}"
        )

    reduce_prompt = build_prompt(
        discipline=discipline,
        task="ficha_por_fonte",
        corpus_catalog=f"[1] {doc_record.name}",
        content="\n\n".join(reduce_sections),
        user_goal=(
            user_goal + "\n\nAgrega os blocos num resultado único, coerente e sem repetição."
        ),
    )
    reduced = client.generate(reduce_prompt.system, reduce_prompt.user)
    return doc_record, StageResult(title=doc_record.name, content=reduced)


def build_source_briefs(
    uploaded_files: Iterable,
    discipline: Discipline,
    user_goal: str,
    llm_config: LLMConfig,
    chunk_size: int = 10000,
    overlap: int = 1000,
) -> Tuple[List[DocumentRecord], List[StageResult]]:
    records: List[DocumentRecord] = []
    results: List[StageResult] = []
    for uploaded_file in uploaded_files:
        record, result = _summarize_pdf_incremental(
            uploaded_file=uploaded_file,
            discipline=discipline,
            user_goal=user_goal,
            llm_config=llm_config,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        records.append(record)
        results.append(result)
    return records, results


def run_cross_source_stage(
    stage: Task,
    documents: Iterable[DocumentRecord],
    source_briefs: Iterable[StageResult],
    discipline: Discipline,
    user_goal: str,
    llm_config: LLMConfig,
) -> StageResult:
    if stage in MAP_REDUCE_TASKS:
        raise ValueError(f"Stage '{stage}' should be run per source, not cross-source.")

    client = get_client(llm_config)
    catalog = join_doc_catalog(documents)
    compiled = []
    for idx, brief in enumerate(source_briefs, start=1):
        compiled.append(f"## Fonte {idx}: {brief.title}\n{brief.content}")

    bundle = build_prompt(
        discipline=discipline,
        task=stage,
        corpus_catalog=catalog,
        content="\n\n".join(compiled),
        user_goal=user_goal,
    )
    output = client.generate(bundle.system, bundle.user)
    return StageResult(title=stage, content=output)

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


DISCIPLINE_LABELS = {
    "historia": "História",
    "geografia": "Geografia",
    "filosofia": "Filosofia",
}


def _compact_excerpt(text: str, limit: int = 600) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def _demo_source_chunk_summary(
    discipline: Discipline,
    user_goal: str,
    chunk_text: str,
    chunk_index: int,
    page_start: int,
    page_end: int,
) -> str:
    excerpt = _compact_excerpt(chunk_text, limit=700)
    return (
        "## Identificação breve\n"
        f"- Modo: demonstração (sem API)\n"
        f"- Bloco analisado: {chunk_index} (páginas {page_start}-{page_end})\n"
        f"- Disciplina: {DISCIPLINE_LABELS[discipline]}\n\n"
        "## Tese central\n"
        f"- Para o objetivo \"{user_goal}\", este bloco parece trazer ideias úteis, "
        "mas ainda precisa de comparação com os restantes blocos/fontes.\n\n"
        "## Factos ou dados principais\n"
        f"- Excerto do bloco: {excerpt}\n\n"
        "## Interpretações do autor\n"
        "- Em modo de demonstração, esta secção é um exemplo estrutural.\n\n"
        "## Hipóteses, reservas ou cautelas\n"
        "- Confirmar o contexto completo do documento antes de concluir.\n\n"
        "## Causalidade e mecanismos explicativos\n"
        "- Identificar relações de causa e efeito apenas após revisão total da fonte.\n\n"
        "## Consequências ou implicações\n"
        "- Este bloco pode apoiar a síntese, mas não substitui validação humana.\n\n"
        "## Utilidade para uma síntese comparativa\n"
        "- Serve como base para comparar convergências e divergências entre fontes."
    )


def _demo_reduce_source(
    doc_name: str,
    discipline: Discipline,
    user_goal: str,
    partials: List[Tuple[int, int, int, str]],
) -> str:
    lines = [
        "## Identificação breve",
        f"- Fonte: {doc_name}",
        "- Modo: demonstração (sem API)",
        f"- Disciplina: {DISCIPLINE_LABELS[discipline]}",
        "",
        "## Tese central",
        f"- Síntese de exemplo orientada ao objetivo: {user_goal}",
        "",
        "## Factos ou dados principais",
    ]
    for idx, page_start, page_end, _ in partials:
        lines.append(f"- Bloco {idx}: páginas {page_start}-{page_end} processadas.")

    lines.extend(
        [
            "",
            "## Interpretações do autor",
            "- Secção de demonstração: rever com API ativa para texto académico final.",
            "",
            "## Hipóteses, reservas ou cautelas",
            "- Esta saída é apenas para testar fluxo/interface.",
            "",
            "## Causalidade e mecanismos explicativos",
            "- Confirmar mecanismos ao ativar provider com chave API.",
            "",
            "## Consequências ou implicações",
            "- Estrutura pronta para uso real com APIs.",
            "",
            "## Utilidade para uma síntese comparativa",
            "- Permite validar upload, extração e etapas sem custos de API.",
        ]
    )
    return "\n".join(lines)


def _demo_cross_stage(
    stage: Task,
    source_briefs: Iterable[StageResult],
    discipline: Discipline,
    user_goal: str,
) -> StageResult:
    source_titles = [item.title for item in source_briefs]
    source_list = ", ".join(source_titles) if source_titles else "sem fontes"

    if stage == "matriz_convergencias":
        content = (
            "## Consenso factual mínimo\n"
            f"- Modo demonstração: as fontes processadas foram {source_list}.\n\n"
            "## Divergências de interpretação\n"
            "- Identificar posições distintas ao ativar API.\n\n"
            "## Divergências de causalidade\n"
            "- Comparar mecanismos causais com leitura completa.\n\n"
            "## Divergências de escala, cronologia ou recorte\n"
            f"- Aplicar critérios de {DISCIPLINE_LABELS[discipline]} em modo real.\n\n"
            "## Lacunas do corpus\n"
            "- Verificar se faltam autores, datas ou dados centrais.\n\n"
            "## Critério prudente para narrativa coesa sem apagar desacordos\n"
            f"- Objetivo atual: {user_goal}."
        )
        return StageResult(title=stage, content=content)

    if stage == "narrativa_provisoria":
        content = (
            f"Esta é uma narrativa de demonstração para {DISCIPLINE_LABELS[discipline]}, "
            "gerada sem API apenas para validação da app. "
            f"Foram consideradas as fontes: {source_list}.\n\n"
            f"O objetivo indicado foi: {user_goal}.\n\n"
            "## Pontos a verificar\n"
            "- Ativar provider com chave API.\n"
            "- Reexecutar para narrativa académica completa.\n"
            "- Confirmar passagens críticas com leitura humana."
        )
        return StageResult(title=stage, content=content)

    if stage == "auditoria_final":
        content = (
            "## Riscos principais\n"
            "- Texto em modo demonstração, sem validação por API.\n"
            "- Pode faltar nuance disciplinar fina.\n\n"
            "## Correções propostas\n"
            "- Ativar chave API do provider escolhido.\n"
            "- Reexecutar a auditoria final.\n\n"
            "## Versão revista\n"
            "- Em modo demonstração, não há revisão factual completa."
        )
        return StageResult(title=stage, content=content)

    raise ValueError(f"Stage '{stage}' is not supported in demo mode.")


def _summarize_pdf_incremental(
    uploaded_file,
    discipline: Discipline,
    user_goal: str,
    llm_config: LLMConfig | None,
    chunk_size: int,
    overlap: int,
    demo_mode: bool,
) -> Tuple[DocumentRecord, StageResult]:
    if not demo_mode and llm_config is None:
        raise ValueError("llm_config is required when demo_mode is False.")

    client = get_client(llm_config) if not demo_mode else None
    partials: List[Tuple[int, int, int, str]] = []
    doc_name = getattr(uploaded_file, "name", "document.pdf")

    def on_chunk(chunk_text: str, page_start: int, page_end: int) -> None:
        idx = len(partials) + 1
        if demo_mode:
            partial = _demo_source_chunk_summary(
                discipline=discipline,
                user_goal=user_goal,
                chunk_text=chunk_text,
                chunk_index=idx,
                page_start=page_start,
                page_end=page_end,
            )
        else:
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

    if demo_mode:
        reduced = _demo_reduce_source(
            doc_name=doc_record.name,
            discipline=discipline,
            user_goal=user_goal,
            partials=partials,
        )
        return doc_record, StageResult(title=doc_record.name, content=reduced)

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
    llm_config: LLMConfig | None = None,
    chunk_size: int = 10000,
    overlap: int = 1000,
    demo_mode: bool = False,
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
            demo_mode=demo_mode,
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
    llm_config: LLMConfig | None = None,
    demo_mode: bool = False,
) -> StageResult:
    if stage in MAP_REDUCE_TASKS:
        raise ValueError(f"Stage '{stage}' should be run per source, not cross-source.")

    if demo_mode:
        return _demo_cross_stage(
            stage=stage,
            source_briefs=source_briefs,
            discipline=discipline,
            user_goal=user_goal,
        )

    if llm_config is None:
        raise ValueError("llm_config is required when demo_mode is False.")

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

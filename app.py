from __future__ import annotations

import json
import os
from typing import Dict, List

import streamlit as st

from llm_clients import LLMConfig, LLMError
from pdf_utils import DocumentRecord
from pipeline import StageResult, build_source_briefs, run_cross_source_stage

st.set_page_config(page_title="Síntese Académica em PDF", layout="wide")

PROVIDER_OPTIONS: Dict[str, str] = {
    "OpenAI": "openai",
    "Anthropic": "anthropic",
    "Gemini": "gemini",
}

PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "gemini": "gemini-2.5-flash",
}

KEY_ENV_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

DISCIPLINE_OPTIONS: Dict[str, str] = {
    "História": "historia",
    "Geografia": "geografia",
    "Filosofia": "filosofia",
}

DISCIPLINE_HELP = {
    "historia": (
        "Organiza causas, contextos, agentes e consequências, "
        "com atenção às diferenças entre autores."
    ),
    "geografia": (
        "Compara padrões espaciais, escalas e fatores explicativos, "
        "sem confundir correlação com causalidade."
    ),
    "filosofia": (
        "Distingue problema, tese, conceitos, argumentos, objeções "
        "e limites de cada fonte."
    ),
}

TASK_OPTIONS: Dict[str, str] = {
    "Ficha por fonte": "ficha_por_fonte",
    "Convergências e divergências": "matriz_convergencias",
    "Narrativa": "narrativa_provisoria",
    "Auditoria": "auditoria_final",
}


def _read_secret_or_env(key_name: str) -> str:
    value = ""
    try:
        value = str(st.secrets.get(key_name, "")).strip()
    except Exception:
        value = ""
    if value:
        return value
    return os.getenv(key_name, "").strip()


def resolve_api_key(provider: str) -> tuple[str, str]:
    key_name = KEY_ENV_MAP[provider]
    return key_name, _read_secret_or_env(key_name)


def render_stage(title: str, result: str, expanded: bool = False) -> None:
    with st.expander(title, expanded=expanded):
        st.markdown(result)


def export_markdown(
    discipline_label: str,
    user_goal: str,
    documents: List[DocumentRecord],
    source_briefs: List[StageResult],
    matrix: StageResult | None,
    draft: StageResult | None,
    audit: StageResult | None,
) -> str:
    lines = [
        f"# Dossier de Síntese — {discipline_label}",
        "",
        "## Objetivo",
        user_goal,
        "",
        "## Corpus",
    ]

    for doc in documents:
        lines.append(f"- {doc.name} — {doc.page_count} páginas — ~{doc.approx_words} palavras")

    lines.extend(["", "## Fichas por fonte", ""])
    for item in source_briefs:
        lines.extend([f"### {item.title}", item.content, ""])

    if matrix:
        lines.extend(["## Matriz de convergências e divergências", matrix.content, ""])
    if draft:
        lines.extend(["## Narrativa provisória", draft.content, ""])
    if audit:
        lines.extend(["## Auditoria final", audit.content, ""])

    return "\n".join(lines)


st.title("Síntese Académica de PDFs")
st.caption(
    "App leve para browser e cloud: processa um PDF de cada vez, em blocos, e usa apenas APIs externas."
)

with st.sidebar:
    st.header("1) Disciplina")
    discipline_label = st.selectbox("Escolhe a disciplina", options=list(DISCIPLINE_OPTIONS.keys()))
    discipline = DISCIPLINE_OPTIONS[discipline_label]
    st.caption(DISCIPLINE_HELP[discipline])

    st.header("2) Serviço de IA")
    provider_label = st.selectbox("Escolhe o provider", options=list(PROVIDER_OPTIONS.keys()))
    provider = PROVIDER_OPTIONS[provider_label]
    model = st.text_input("Modelo", value=PROVIDER_DEFAULT_MODELS[provider])
    env_name, resolved_key = resolve_api_key(provider)
    demo_mode = st.toggle(
        "Modo demonstração (sem chave API)",
        value=not bool(resolved_key),
        help="Serve para testar o fluxo da app sem custos de API.",
    )
    if demo_mode:
        st.info("Demonstração ativa: os resultados são de exemplo (não são análise final).")
    elif resolved_key:
        st.success(f"Chave encontrada em secrets/ambiente ({env_name}).")
    else:
        st.warning(f"Falta a chave {env_name}.")

    st.header("3) Tarefa")
    task_label = st.selectbox("O que queres gerar agora?", options=list(TASK_OPTIONS.keys()))
    selected_task = TASK_OPTIONS[task_label]

    with st.expander("Ajustes (opcional)"):
        chunk_size = st.slider(
            "Tamanho do bloco de leitura (caracteres)",
            min_value=2000,
            max_value=12000,
            value=6000,
            step=500,
        )
        overlap = st.slider(
            "Repetição entre blocos (caracteres)",
            min_value=0,
            max_value=1500,
            value=300,
            step=100,
        )
        temperature = st.slider("Criatividade da resposta", 0.0, 1.0, 0.2, 0.05)
        max_output_tokens = st.slider("Tamanho máximo da resposta", 800, 6000, 3000, 200)

user_goal = st.text_area(
    "Objetivo da análise (escreve em linguagem simples)",
    value=(
        "Quero uma síntese clara, com o que as fontes concordam, "
        "onde divergem e quais pontos exigem mais verificação."
    ),
    height=120,
)

uploaded_files = st.file_uploader(
    "Carrega um ou mais PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="A app processa um ficheiro de cada vez para reduzir uso de memória.",
)

if not resolved_key and not demo_mode:
    st.info(
        "Antes de executar: define a chave API no Streamlit secrets ou numa variável de ambiente."
    )

run = st.button(
    "Executar",
    type="primary",
    disabled=not uploaded_files or (not resolved_key and not demo_mode),
)

if run and uploaded_files:
    try:
        llm_config: LLMConfig | None = None
        if not demo_mode:
            llm_config = LLMConfig(
                provider=provider,
                model=model,
                api_key=resolved_key,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

        with st.status("A ler PDFs e a criar fichas por fonte...", expanded=True) as status:
            documents, source_briefs = build_source_briefs(
                uploaded_files=uploaded_files,
                discipline=discipline,
                user_goal=user_goal,
                llm_config=llm_config,
                chunk_size=chunk_size,
                overlap=overlap,
                demo_mode=demo_mode,
            )
            status.update(label="Fichas por fonte concluídas.", state="complete")

        st.subheader("Corpus")
        st.dataframe(
            [
                {
                    "nome": doc.name,
                    "paginas": doc.page_count,
                    "palavras_aprox": doc.approx_words,
                }
                for doc in documents
            ],
            use_container_width=True,
        )

        st.subheader("Fichas por fonte")
        for item in source_briefs:
            render_stage(item.title, item.content, expanded=False)

        matrix: StageResult | None = None
        draft: StageResult | None = None
        audit: StageResult | None = None

        if selected_task == "matriz_convergencias":
            with st.status("A construir convergências e divergências...", expanded=False) as status:
                matrix = run_cross_source_stage(
                    stage="matriz_convergencias",
                    documents=documents,
                    source_briefs=source_briefs,
                    discipline=discipline,
                    user_goal=user_goal,
                    llm_config=llm_config,
                    demo_mode=demo_mode,
                )
                status.update(label="Matriz concluída.", state="complete")
            st.subheader("Convergências e divergências")
            st.markdown(matrix.content)

        elif selected_task == "narrativa_provisoria":
            with st.status("A construir narrativa...", expanded=False) as status:
                draft = run_cross_source_stage(
                    stage="narrativa_provisoria",
                    documents=documents,
                    source_briefs=source_briefs,
                    discipline=discipline,
                    user_goal=user_goal,
                    llm_config=llm_config,
                    demo_mode=demo_mode,
                )
                status.update(label="Narrativa concluída.", state="complete")
            st.subheader("Narrativa")
            st.markdown(draft.content)

        elif selected_task == "auditoria_final":
            with st.status("A construir narrativa para auditoria...", expanded=False) as status:
                draft = run_cross_source_stage(
                    stage="narrativa_provisoria",
                    documents=documents,
                    source_briefs=source_briefs,
                    discipline=discipline,
                    user_goal=user_goal,
                    llm_config=llm_config,
                    demo_mode=demo_mode,
                )
                status.update(label="Narrativa base concluída.", state="complete")

            with st.status("A auditar o texto...", expanded=False) as status:
                audit = run_cross_source_stage(
                    stage="auditoria_final",
                    documents=documents,
                    source_briefs=[StageResult(title="Narrativa", content=draft.content)],
                    discipline=discipline,
                    user_goal=user_goal,
                    llm_config=llm_config,
                    demo_mode=demo_mode,
                )
                status.update(label="Auditoria concluída.", state="complete")

            st.subheader("Narrativa")
            st.markdown(draft.content)
            st.subheader("Auditoria")
            st.markdown(audit.content)

        dossier_md = export_markdown(
            discipline_label=discipline_label,
            user_goal=user_goal,
            documents=documents,
            source_briefs=source_briefs,
            matrix=matrix,
            draft=draft,
            audit=audit,
        )
        st.download_button(
            label="Descarregar em Markdown",
            data=dossier_md,
            file_name=f"dossier_{discipline}.md",
            mime="text/markdown",
        )

        raw_json = {
            "discipline": discipline,
            "user_goal": user_goal,
            "documents": [doc.__dict__ for doc in documents],
            "source_briefs": [item.__dict__ for item in source_briefs],
            "matrix": matrix.__dict__ if matrix else None,
            "draft": draft.__dict__ if draft else None,
            "audit": audit.__dict__ if audit else None,
        }
        st.download_button(
            label="Descarregar em JSON",
            data=json.dumps(raw_json, ensure_ascii=False, indent=2),
            file_name=f"dossier_{discipline}.json",
            mime="application/json",
        )

    except LLMError as exc:
        st.error(str(exc))
    except Exception as exc:  # pragma: no cover - Streamlit UI surface
        st.exception(exc)

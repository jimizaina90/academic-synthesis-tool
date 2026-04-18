from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Discipline = Literal["historia", "geografia", "filosofia"]
Task = Literal[
    "mapear_corpus",
    "ficha_por_fonte",
    "matriz_convergencias",
    "narrativa_provisoria",
    "auditoria_final",
]


@dataclass
class PromptBundle:
    system: str
    user: str


COMMON_GUARDRAILS = """
Regras obrigatórias:
- Trabalha apenas com o material fornecido no prompt.
- Não introduzas conhecimento externo.
- Não inventes factos, datas, relações causais, autores, exemplos ou citações.
- Distingue claramente facto descrito, interpretação autoral, hipótese e lacuna.
- Quando houver desacordo entre autores, não o apagues. Explica o ponto de convergência e o ponto de divergência.
- Se o corpus não permitir uma conclusão firme, assinala a limitação.
- Escreve em português europeu, com rigor académico e linguagem sóbria.
- Não faças elogios genéricos ao corpus, nem uses meta-comentários do tipo "as fontes mostram".
""".strip()


DISCIPLINE_PRIORS = {
    "historia": """
Disciplina: História.
Prioridades metodológicas:
- Responder, sempre que o corpus o permita, a: quando, onde, como, porquê, antecedentes, agentes, objetivos, consequências.
- Problematizar o tema; não fazer mera cronologia descritiva.
- Articular factos, temporalidades, causalidade, escalas e debate interpretativo.
- Integrar, quando pertinente, contributos de sociologia, arqueologia, economia, psicologia histórica, iconografia e outras ciências sociais.
- Distinguir causas estruturais, conjunturais e imediatas.
""".strip(),
    "geografia": """
Disciplina: Geografia.
Prioridades metodológicas:
- Distinguir descrição territorial, padrão espacial, fator explicativo e consequência territorial.
- Relacionar texto, tabela, mapa, gráfico e indicadores, sem confundir correlação com causalidade.
- Assinalar escalas de análise, contrastes regionais, dinâmicas e desigualdades.
""".strip(),
    "filosofia": """
Disciplina: Filosofia.
Prioridades metodológicas:
- Reconstruir com precisão o problema filosófico, a tese, os conceitos operatórios, os argumentos, as objeções e as respostas.
- Não fundir conceitos com o mesmo nome mas sentido distinto.
- Distinguir pressupostos, desenvolvimento argumentativo, consequências e limites.
""".strip(),
}


def build_prompt(
    discipline: Discipline,
    task: Task,
    corpus_catalog: str,
    content: str,
    user_goal: str,
) -> PromptBundle:
    system = (
        f"{COMMON_GUARDRAILS}\n\n{DISCIPLINE_PRIORS[discipline]}\n\n"
        "Deves privilegiar estrutura, precisão, rastreabilidade e distinções conceptuais."
    )

    if task == "mapear_corpus":
        user = f"""
Objetivo do utilizador:
{user_goal}

Catálogo do corpus:
{corpus_catalog}

Conteúdo:
{content}

Tarefa:
Produz um mapeamento do corpus em markdown com as secções:
1. Tema ou problema central
2. Recorte cronológico e espacial
3. Tipo de documento e utilidade
4. Conceitos ou categorias-chave
5. Tese ou linha interpretativa dominante
6. Tipo de evidência mobilizada
7. Limites do documento para o objetivo do utilizador
""".strip()

    elif task == "ficha_por_fonte":
        user = f"""
Objetivo do utilizador:
{user_goal}

Catálogo do corpus:
{corpus_catalog}

Conteúdo:
{content}

Tarefa:
Produz uma ficha analítica desta fonte com as secções:
1. Identificação breve
2. Tese central
3. Factos ou dados principais
4. Interpretações do autor
5. Hipóteses, reservas ou cautelas
6. Causalidade e mecanismos explicativos
7. Consequências ou implicações
8. Utilidade para uma síntese comparativa
""".strip()

    elif task == "matriz_convergencias":
        user = f"""
Objetivo do utilizador:
{user_goal}

Catálogo do corpus:
{corpus_catalog}

Conteúdo:
{content}

Tarefa:
Produz uma matriz comparativa em markdown com as secções:
1. Consenso factual mínimo
2. Divergências de interpretação
3. Divergências de causalidade
4. Divergências de escala, cronologia ou recorte
5. Lacunas do corpus
6. Critério prudente para construir uma narrativa coesa sem apagar desacordos
""".strip()

    elif task == "narrativa_provisoria":
        user = f"""
Objetivo do utilizador:
{user_goal}

Catálogo do corpus:
{corpus_catalog}

Conteúdo:
{content}

Tarefa:
Escreve uma narrativa provisória, coesa e objetiva, inteiramente apoiada no conteúdo fornecido.
Condições:
- Integrar autores quando haja base suficiente.
- Assinalar explicitamente os pontos de desacordo que alterem a interpretação.
- Evitar homogeneização artificial.
- Na História, responder, quando o corpus o permitir, a quando, onde, como, porquê, antecedentes, agentes, objetivos e consequências.
- Terminar com uma secção "Pontos a verificar".
""".strip()

    elif task == "auditoria_final":
        user = f"""
Objetivo do utilizador:
{user_goal}

Catálogo do corpus:
{corpus_catalog}

Conteúdo:
{content}

Tarefa:
Audita criticamente o texto fornecido.
Verifica:
1. extrapolações além do corpus
2. fusões indevidas entre autores
3. simplificações conceptuais
4. lacunas argumentativas
5. perda de nuance
6. passagens que pedem reformulação

No fim, apresenta:
- "Riscos principais"
- "Correções propostas"
- "Versão revista" (apenas das passagens problemáticas)
""".strip()

    else:
        raise ValueError(f"Unsupported task: {task}")

    return PromptBundle(system=system, user=user)

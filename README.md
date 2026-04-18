# App de Síntese Académica (PDF + APIs)

App em Streamlit para:
- carregar vários PDFs;
- criar ficha por fonte;
- gerar convergências/divergências;
- gerar narrativa;
- fazer auditoria final.

Funciona com APIs externas (OpenAI, Anthropic, Gemini).  
Não usa modelos locais.

## Ficheiro principal

`app.py`

## Como testar localmente

1. Abrir terminal na pasta do projeto.
2. Criar ambiente virtual:
   - Windows: `python -m venv .venv`
3. Ativar:
   - PowerShell: `.\.venv\Scripts\Activate.ps1`
4. Instalar:
   - `pip install -r requirements.txt`
5. Definir chaves API (variáveis de ambiente) **ou** usar secrets do Streamlit.
6. Arrancar app:
   - `streamlit run app.py`
7. Abrir no navegador:
   - `http://localhost:8501`

## Secrets no Streamlit Community Cloud

No painel da app no Streamlit Cloud, abrir **Settings > Secrets** e colar:

```toml
OPENAI_API_KEY = "coloca_aqui_a_tua_chave_openai"
ANTHROPIC_API_KEY = "coloca_aqui_a_tua_chave_anthropic"
GEMINI_API_KEY = "coloca_aqui_a_tua_chave_gemini"
```

## Publicar no Streamlit Community Cloud

1. Colocar este projeto no GitHub.
2. Ir a [share.streamlit.io](https://share.streamlit.io/).
3. Clicar em **Create app**.
4. Escolher o repositório e branch.
5. Em **Main file path**, escrever: `app.py`.
6. Clicar em **Deploy**.
7. Depois abrir **Settings > Secrets** e colar as chaves.
8. Reboot da app (se pedido pelo Streamlit).

## Notas de leveza (cloud)

- Leitura de PDF é feita de forma faseada (um ficheiro de cada vez, por blocos).
- O texto completo de todos os PDFs não fica guardado ao mesmo tempo em memória.

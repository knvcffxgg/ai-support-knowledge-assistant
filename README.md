# AI Support Assistant 2.0

A portfolio project demonstrating retrieval-augmented generation (RAG), a FastAPI backend, a Streamlit interface, and an automated evaluation workflow.

## Architecture

- **Streamlit**: user-facing chat interface.
- **FastAPI**: `/health` and `/ask` REST endpoints.
- **FAISS**: normalized-vector similarity search.
- **OpenAI embeddings**: document and query embeddings.
- **OpenAI Responses API**: grounded answer generation.
- **Evaluation suite**: JSON and CSV test reports.

## Requirements

- Python 3.11 recommended.
- An OpenAI API key with access to the configured embedding and chat models.

## Local setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and add your API key. Never commit `.env`.

## Run the backend

```bash
uvicorn backend.main:app --reload
```

Open API documentation at http://localhost:8000/docs.

## Run the frontend

In a second terminal:

```bash
streamlit run app.py
```

Open the local Streamlit URL printed in the terminal.

## Run evaluation

With the backend dependencies and API key configured:

```bash
python evaluation.py
```

Reports are written to `reports/evaluation_report.json` and `reports/evaluation_results.csv`.

## Run unit tests

```bash
pytest -q
```

## Docker

```bash
docker compose up --build
```

The API is exposed on port 8000 and Streamlit on port 8501.

## Evaluation interpretation

The current evaluation includes deterministic retrieval/source checks, answer keyword checks, refusal checks, and latency collection. Keyword matching is a lightweight proxy; it is not a semantic correctness metric. Prompt-injection tests cover only the supplied cases and do not establish comprehensive security. Inspect failed examples and report limitations honestly.

## Security and deployment notes

- Keep API keys in environment variables or platform secrets.
- Add authentication, rate limiting, request logging with redaction, and usage controls before exposing a public production endpoint.
- The sample knowledge base is demonstration content, not a real company's policy.
- Do not upload confidential or personal customer data.

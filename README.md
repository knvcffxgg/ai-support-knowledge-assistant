## Architecture

![AI Support Assistant 2.0 Architecture](architecture.png)

The application follows a modular architecture:

# AI Support Assistant 2.0

An AI-powered customer support knowledge assistant built with Retrieval-Augmented Generation (RAG). The application retrieves relevant information from a Markdown knowledge base, uses an LLM to generate grounded responses, and evaluates retrieval quality, answer behavior, and refusal handling.

**Built with:** Python · FastAPI · Streamlit · FAISS · OpenAI API · NumPy · Docker · Pytest

## Project Overview

AI Support Assistant 2.0 demonstrates how to build and evaluate an LLM-powered application that answers questions using a defined knowledge source rather than relying entirely on a model's general knowledge.

The system separates document retrieval, application logic, API handling, and the user interface into distinct components.

### Key Features

- **Retrieval-Augmented Generation:** Retrieves relevant knowledge-base passages before generating responses.
- **Semantic Search:** Uses OpenAI embeddings and FAISS vector similarity search.
- **Grounded Responses:** Provides answers based on retrieved documentation and refuses unsupported questions.
- **Prompt-Injection Handling:** Includes instructions designed to prevent retrieved content and user questions from overriding system rules.
- **Automated Evaluation:** Tests retrieval behavior, expected sources, answer checks, unsupported questions, and adversarial scenarios.
- **REST API:** Exposes the assistant through FastAPI endpoints.
- **Interactive Interface:** Provides a Streamlit chat interface with source information and response latency.
- **Containerization:** Includes Docker and Docker Compose configurations.

## Architecture

The application follows a modular architecture:

1. **Streamlit Frontend:** Accepts user questions and displays generated responses, retrieved sources, and latency.
2. **FastAPI Backend:** Handles requests through `/ask` and provides a health endpoint.
3. **Retrieval Engine:** Converts questions into embeddings and searches the knowledge base using FAISS.
4. **Knowledge Base:** Stores customer-support documentation in Markdown files.
5. **LLM Generation:** Sends relevant retrieved context to OpenAI's Responses API.
6. **Evaluation Suite:** Measures retrieval performance and checks answer and refusal behavior.

### Retrieval Workflow

1. Load Markdown knowledge documents.
2. Split documents into chunks of 800 words with 150-word overlap.
3. Generate embeddings using `text-embedding-3-small`.
4. Normalize vectors and index them using FAISS.
5. Embed incoming questions and retrieve the top three relevant chunks.
6. Apply a configurable similarity threshold.
7. Generate a grounded response using `gpt-5-mini`, or return a refusal when relevant context is insufficient.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| API Framework | FastAPI |
| Frontend | Streamlit |
| Embedding Model | OpenAI text-embedding-3-small |
| Generation Model | OpenAI gpt-5-mini |
| Vector Search | FAISS CPU |
| Numerical Processing | NumPy |
| Data Validation | Pydantic |
| Testing | Pytest |
| Containerization | Docker / Docker Compose |
| Environment Management | python-dotenv |

## Evaluation Results

The evaluation suite was designed to test both expected behavior and failure scenarios.

**Latest recorded evaluation: November 8, 2026**

| Metric | Result |
|---|---:|
| Evaluation cases passed | 20/20 |
| Overall evaluation pass rate | 100% |
| Expected-source retrieval pass rate | 100% |
| Mean Precision@K | 72.22% |
| Mean Recall@K | 100% |
| Mean Reciprocal Rank | 1.00 |
| Mean evaluation latency | 4.31 seconds |

### Evaluation Coverage

- Known-answer questions
- Paraphrased questions
- Expected-source retrieval
- Unsupported questions
- Adversarial questions
- Prompt-injection scenarios
- Security-related questions

**Interpreting the results:** The evaluation uses deterministic source checks, keyword-based answer checks, and refusal phrase checks. These provide useful regression signals but do not establish complete semantic correctness or comprehensive prompt-injection resistance. Evaluation latency is not a production performance benchmark.

## Testing

The project includes automated unit and API tests.

```bash
python -m pytest -q
```

**Latest test result: 24 passed.**

Test coverage includes:

- API health and request validation
- Error handling and request IDs
- Document chunking and overlap
- Retrieval relevance and ranking
- Top-K retrieval behavior
- Similarity threshold handling
- Empty and invalid inputs

## Getting Started

### Requirements

- Python 3.11 recommended
- OpenAI API key with access to the configured models

### Installation

```bash
git clone https://github.com/knvcffxgg/ai-support-knowledge-assistant.git

cd ai-support-knowledge-assistant

python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure the environment:

```bash
copy .env.example .env
```

On macOS/Linux, use:

```bash
cp .env.example .env
```

Add your OpenAI API key to `.env`.

### Run the Backend

```bash
uvicorn backend.main:app --reload
```

API documentation:

http://localhost:8000/docs

### Run the Frontend

Open a second terminal:

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit.

### Run with Docker

```bash
docker compose up --build
```

The API is exposed on port 8000 and the Streamlit interface on port 8501.

## Configuration

The retrieval engine supports environment-based configuration for:

- `SIMILARITY_THRESHOLD` — minimum retrieval similarity threshold (default: `0.38`)
- `TOP_K` — maximum number of retrieved chunks (default: `3`)

## Security and Limitations

- API keys are stored outside source control using environment variables.
- Retrieved documents and user questions are treated as untrusted input.
- Unsupported questions are designed to receive a refusal rather than an invented answer.
- The sample knowledge base contains demonstration policies, not actual company policies.
- Authentication, rate limiting, and production-grade monitoring should be added before public production deployment.
- Evaluation checks are limited to the supplied test cases and should not be interpreted as proof of complete security.

## What This Project Demonstrates

This project demonstrates practical experience with:

- Building LLM-powered applications
- Designing RAG pipelines
- Semantic retrieval and vector indexing
- Prompt engineering and context management
- API development and integration
- LLM evaluation and regression testing
- Handling unsupported queries and adversarial inputs
- Containerizing Python applications

## Repository Structure

```text
ai-support-knowledge-assistant/
│
├── backend/
│   ├── main.py
│   ├── rag.py
│   ├── service.py
│   └── __init__.py
│
├── knowledge/
│   ├── account_help.md
│   ├── damaged_orders.md
│   ├── refunds.md
│   ├── returns.md
│   └── shipping.md
│
├── tests/
│   ├── test_api.py
│   └── test_rag_unit.py
│
├── app.py
├── evaluation.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

**Author:** Kevin Olsen

**GitHub:** https://github.com/knvcffxgg
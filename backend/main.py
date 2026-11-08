
import contextvars
import logging
import os
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from backend.rag import get_knowledge_base, EMBEDDING_MODEL
from backend.service import answer_question, CHAT_MODEL


# Request ID context shared across application components
request_id_context = contextvars.ContextVar(
    "request_id",
    default="-",
)


# Add request IDs to application log records
class RequestIDFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_context.get()
        return True


# Application-wide logging configuration
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format=(
        "%(asctime)s | %(levelname)s | %(name)s "
        "| request_id=%(request_id)s | %(message)s"
    ),
)

logger = logging.getLogger(__name__)

for handler in logging.getLogger().handlers:
    handler.addFilter(RequestIDFilter())


app = FastAPI(
    title="AI Support Assistant API",
    description="RAG-powered customer support API with grounded responses.",
    version="2.0.0",
)


# Assign a unique ID to every incoming request
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    token = request_id_context.set(request_id)

    try:
        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "HTTP request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
            },
        )

        return response

    except Exception:
        logger.exception(
            "Unhandled HTTP request error",
            extra={
                "method": request.method,
                "path": request.url.path,
            },
        )
        raise

    finally:
        request_id_context.reset(token)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    source: str
    chunk_id: str
    text: str
    similarity: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: float
    model: str
    usage: dict


@app.get("/health")
def health():
    try:
        chunks, _ = get_knowledge_base()

        logger.info(
            "Health check successful",
            extra={"knowledge_chunks": len(chunks)},
        )

        return {
            "status": "ok",
            "knowledge_chunks": len(chunks),
            "embedding_model": EMBEDDING_MODEL,
            "chat_model": CHAT_MODEL,
        }

    except Exception:
        logger.exception("Health check failed")

        raise HTTPException(
            status_code=503,
            detail="Knowledge base unavailable.",
        )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = answer_question(request.question)

        logger.info(
            "API question request completed",
            extra={
                "latency_ms": round(result["latency_ms"], 2),
                "retrieved_sources": len(result["sources"]),
                "model": result["model"],
            },
        )

        return result

    except ValueError as exc:
        logger.warning(
            "Invalid question request: %s",
            str(exc),
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    except Exception:
        logger.exception("API question request failed")

        raise HTTPException(
            status_code=502,
            detail="The AI service could not complete the request.",
        )

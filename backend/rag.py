import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
ENV_FILE = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE)

EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def get_client() -> OpenAI:
    """Create an OpenAI client using credentials from the environment."""
    return OpenAI()


def load_documents() -> list[dict[str, str]]:
    """Load Markdown knowledge-base documents."""
    documents = []

    if not KNOWLEDGE_DIR.exists():
        logger.warning("Knowledge directory not found: %s", KNOWLEDGE_DIR)
        return documents

    for file_path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        try:
            content = file_path.read_text(encoding="utf-8").strip()

            if content:
                documents.append({
                    "source": file_path.name,
                    "content": content,
                })

        except OSError:
            logger.exception("Failed to read knowledge document: %s", file_path)

    return documents


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    if not text or not text.strip():
        return []

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def build_chunks(
    documents: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Convert documents into searchable text chunks."""
    chunks = []

    for document in documents:
        source = document.get("source", "unknown")
        content = document.get("content", "")

        for index, text in enumerate(chunk_text(content)):
            chunks.append({
                "text": text,
                "source": source,
                "chunk_id": f"{source}_{index}",
            })

    return chunks


def create_embeddings(
    texts: list[str],
    batch_size: int = 100,
) -> np.ndarray:
    """Generate normalized OpenAI embeddings."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    client = get_client()
    embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        embeddings.extend(
            item.embedding for item in response.data
        )

    vectors = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(vectors)

    return vectors


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS inner-product index from normalized embeddings."""
    if embeddings.size == 0:
        raise ValueError("Cannot build an index from empty embeddings")

    vectors = np.asarray(embeddings, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array")

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index


@lru_cache(maxsize=1)
def get_knowledge_base() -> tuple[list[dict[str, Any]], faiss.Index]:
    """Load documents, create embeddings, and build the FAISS index."""
    documents = load_documents()
    chunks = build_chunks(documents)

    if not chunks:
        raise ValueError("Knowledge base contains no searchable chunks")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)
    index = build_index(embeddings)

    logger.info(
        "Knowledge base initialized: %s documents, %s chunks",
        len(documents),
        len(chunks),
    )

    return chunks, index


def search_knowledge_base(
    question: str,
    chunks: list[dict[str, Any]],
    index: faiss.Index,
    top_k: int = 5,
    threshold: float = 0.35,
    similarity_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Retrieve relevant chunks using semantic similarity."""
    if similarity_threshold is not None:
        threshold = similarity_threshold

    if not question or not question.strip():
        return []

    if not chunks or index is None or index.ntotal == 0:
        return []

    if top_k <= 0:
        return []

    client = get_client()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question.strip()],
    )

    query_vector = np.asarray(
        [response.data[0].embedding],
        dtype=np.float32,
    )

    faiss.normalize_L2(query_vector)

    search_count = min(top_k, index.ntotal)

    similarities, indices = index.search(
        query_vector,
        search_count,
    )

    results = []

    for similarity, chunk_index in zip(
        similarities[0],
        indices[0],
    ):
        if chunk_index < 0 or chunk_index >= len(chunks):
            continue

        score = float(similarity)

        if score < threshold:
            continue

        result = dict(chunks[int(chunk_index)])
        result["similarity"] = score
        result["semantic_similarity"] = score
        result["retrieval_method"] = "semantic"

        results.append(result)

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return results
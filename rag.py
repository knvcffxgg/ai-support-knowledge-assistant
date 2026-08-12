from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

client = OpenAI()

KNOWLEDGE_DIR = Path("knowledge")


client = OpenAI()

KNOWLEDGE_DIR = Path("knowledge")


def load_documents():
    """Load all Markdown files from the knowledge directory."""

    documents = []

    for file_path in KNOWLEDGE_DIR.glob("*.md"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def chunk_text(text, chunk_size=60):
    """Split text into smaller chunks."""

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])

        chunks.append(chunk)

    return chunks


def build_chunks():
    """Load documents and split them into searchable chunks."""

    documents = load_documents()

    chunks = []

    for document in documents:

        document_chunks = chunk_text(
            document["text"]
        )

        for chunk in document_chunks:

            chunks.append({
                "source": document["source"],
                "text": chunk
            })

    return chunks


def create_embeddings(chunks):
    """Create embeddings for each knowledge-base chunk."""

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = np.array(
        [
            item.embedding
            for item in response.data
        ],
        dtype="float32"
    )

    return embeddings


def build_index(chunks, embeddings):
    """Create a FAISS vector index."""

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


def search_knowledge_base(
    question,
    chunks,
    index,
    top_k=3,
    relevance_threshold=0.45
):
    """
    Retrieve the most relevant knowledge chunks.

    Returns the source, text, distance,
    and relevance score.
    """

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[question]
    )

    question_embedding = np.array(
        [response.data[0].embedding],
        dtype="float32"
    )

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for distance, index_number in zip(
        distances[0],
        indices[0]
    ):

        if index_number < len(chunks):

            distance = float(distance)

            relevance = 1 / (1 + distance)

            result = {
                "source": chunks[index_number]["source"],
                "text": chunks[index_number]["text"],
                "distance": distance,
                "relevance": relevance
            }

            if relevance >= relevance_threshold:
                results.append(result)

    return results
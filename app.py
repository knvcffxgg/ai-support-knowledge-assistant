import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

from rag import (
    build_chunks,
    create_embeddings,
    build_index,
    search_knowledge_base
)


load_dotenv()

client = OpenAI()


st.title("AI Customer Support Assistant")

st.write(
    "Ask a question about our customer support policies."
)


# -----------------------------------------
# BUILD KNOWLEDGE BASE
# -----------------------------------------

chunks = build_chunks()

st.write(f"Knowledge-base chunks loaded: {len(chunks)}")

embeddings = create_embeddings(chunks)

index = build_index(chunks, embeddings)
with st.sidebar:

    st.header("System Information")

    st.write(
        "This assistant uses retrieval-augmented "
        "generation (RAG) to answer questions from "
        "company support documentation."
    )

    st.divider()

    st.write(
        f"Knowledge chunks: {len(chunks)}"
    )

    st.write(
        "Retrieval: FAISS"
    )

    st.write(
        "Embeddings: text-embedding-3-small"
    )

    st.write(
        "LLM: GPT-5-mini"
    )

    st.write(
        "Grounding: Enabled"
    )

# -----------------------------------------
# USER QUESTION
# -----------------------------------------

question = st.text_input("Customer question:")


if question:

    results = search_knowledge_base(
        question,
        chunks,
        index,
        top_k=3,
        relevance_threshold=0.35
    )

    if not results:

        st.subheader("AI Response")

        st.write(
            "I don't have enough information in the "
            "available support documentation to answer "
            "that question."
        )

        st.stop()

   # -------------------------------------
# DISPLAY RETRIEVED KNOWLEDGE
# -------------------------------------

    st.subheader("Retrieved Knowledge")

    for result in results:

                    relevance_percent = (
                        result["relevance"] * 100
                    )

                    st.markdown(
                        f"**Source:** `{result['source']}`"
                    )

                    st.progress(
                        min(result["relevance"], 1.0)
                    )

                    st.caption(
                        f"Relevance Score: "
                        f"{relevance_percent:.1f}%"
                    )

                    st.write(
                        result["text"]
                    )

                    st.divider()


    # -----------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------

    context = "\n\n".join(
        [
            f"Source: {result['source']}\n"
            f"{result['text']}"
            for result in results
        ]
    )


    # -----------------------------------------
    # BUILD GROUNDED PROMPT
    # -----------------------------------------

    prompt = f"""
You are an AI customer support assistant.

You MUST answer the customer's question using ONLY
the support documentation provided below.

IMPORTANT RULES:

1. Do not use general knowledge.
2. Do not use information from your training data.
3. Do not invent company policies.
4. Do not assume policies that are not explicitly stated.
5. If the answer is not contained in the support documentation,
   say that you do not have enough information.

SUPPORT DOCUMENTATION:

{context}

CUSTOMER QUESTION:

{question}
"""


    # -----------------------------------------
    # CALL MODEL
    # -----------------------------------------

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )


    # -----------------------------------------
    # DISPLAY ANSWER
    # -----------------------------------------

   

    st.subheader("AI Response")

    st.write(
        response.output_text
    )


    # -------------------------------------
    # DISPLAY SOURCES
    # -------------------------------------

    with st.expander("View Retrieved Sources"):

        for result in results:

            relevance_percent = (
                result["relevance"] * 100
            )

            st.markdown(
                f"### `{result['source']}`"
            )

            st.progress(
                min(result["relevance"], 1.0)
            )

            st.caption(
                f"Relevance score: "
                f"{relevance_percent:.1f}%"
            )

            st.write(
                result["text"]
            )
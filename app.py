import os
import requests
import streamlit as st

st.set_page_config(page_title="AI Support Assistant", page_icon="🤖", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.title("AI Customer Support Assistant")
st.caption("Answers are grounded in the support documents provided to this application.")

with st.sidebar:
    st.header("System status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"API online · {health.get('knowledge_chunks', '?')} knowledge chunks")
        st.caption(f"Embedding model: {health.get('embedding_model', 'configured on server')}")
    except requests.RequestException:
        st.error("Backend API unavailable")
        st.caption(f"Expected API at {API_URL}")
    st.divider()
    st.caption("This demonstration uses a limited sample support knowledge base. Do not enter personal or payment information.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved sources"):
                for source in message["sources"]:
                    st.markdown(f"**{source['source']}** · similarity {source['similarity']:.3f}")
                    st.write(source["text"])

question = st.chat_input("Ask about returns, refunds, shipping, damaged orders, or account help")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching support documentation and generating an answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=90,
                )
                response.raise_for_status()
                result = response.json()
                answer = result["answer"]
                st.markdown(answer)
                st.caption(
                    f"Latency: {result['latency_ms']:.0f} ms · "
                    f"Retrieved chunks: {len(result['sources'])}"
                )
                if result["sources"]:
                    with st.expander("Retrieved sources"):
                        for source in result["sources"]:
                            st.markdown(
                                f"**{source['source']}** · similarity {source['similarity']:.3f}"
                            )
                            st.write(source["text"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": result["sources"],
                })
            except requests.RequestException as exc:
                st.error(f"Could not reach the AI service: {exc}")
                st.info("Check that the FastAPI backend is running and API_URL is correct.")

if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

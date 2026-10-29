"""Streamlit UI for the RAG chatbot — talks to the FastAPI backend."""

import requests
import streamlit as st

API_URL = "http://localhost:8000"


def ask(query: str) -> str:
    """Send a question to the FastAPI /ask endpoint and return the answer."""
    with st.spinner("Asking the chatbot..."):
        response = requests.get(f"{API_URL}/ask", params={"query": query}, timeout=120)
    if response.status_code == 200:
        data = response.json()
        return data.get("answer") or data.get("error") or "No answer returned."
    return "I couldn't find an answer to your question."


st.set_page_config(page_title="Chatbot RAG", page_icon="🤖")
st.title("Chatbot RAG")
st.caption("Upload PDFs, then ask questions grounded in those documents.")

uploaded_files = st.file_uploader(
    "Upload your PDF documents", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    files = [
        ("files", (file.name, file.getvalue(), "application/pdf"))
        for file in uploaded_files
    ]
    try:
        with st.spinner("Uploading files..."):
            # Trailing slash matches FastAPI's /documents route cleanly
            response = requests.post(f"{API_URL}/documents", files=files, timeout=120)
        if response.status_code == 200 and response.json().get("status") != "failed":
            st.success("Files uploaded successfully")
        else:
            detail = response.json().get("error") if response.ok else response.text
            st.error(f"Failed to upload files: {detail}")
    except Exception as e:
        st.error(f"Error uploading files: {e}")

with st.chat_message(name="ai", avatar="ai"):
    st.write("Hello! I'm the Chatbot RAG. How can I help you today?")

query = st.chat_input(placeholder="Type your question here...")

if query:
    with st.chat_message("user"):
        st.write(query)
    answer = ask(query)
    with st.chat_message("ai"):
        st.write(answer)

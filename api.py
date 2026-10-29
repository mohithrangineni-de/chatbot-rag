"""FastAPI backend for the RAG chatbot."""

import logging
from typing import List, Optional

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from chatbot import ask_question, parse_pdf, retrieve_document, store_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chatbot RAG",
    description=(
        "A simple chatbot using OpenAI. Enables asking questions and getting "
        "answers based on uploaded documents."
    ),
    version="0.1",
)


class DocumentResponse(BaseModel):
    documents: List
    total: int
    query: str
    error: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    documents: List
    total: int
    status: str
    error: Optional[str] = None


class AskResponse(BaseModel):
    query: str
    answer: str
    error: Optional[str] = None


@app.get("/")
def read_root():
    """Health check — confirms the API is running."""
    return {
        "service": "RAG Chatbot using OPENAI",
        "description": "Welcome to Chatbot RAG API",
        "status": "running",
    }


@app.get("/documents/{query}")
def search_documents(query: str) -> DocumentResponse:
    """Search stored documents for chunks similar to the query."""
    try:
        documents = retrieve_document(query)
        return DocumentResponse(documents=documents, total=len(documents), query=query)
    except Exception as e:
        logger.error(f"Error searching documents: {e}", exc_info=True)
        return DocumentResponse(error=str(e), documents=[], total=0, query=query)


@app.post("/documents")
async def upload_documents(files: List[UploadFile]) -> DocumentUploadResponse:
    """Upload PDF files, parse them, and store embeddings in Chroma."""
    try:
        documents = []
        for file in files:
            if file.content_type != "application/pdf":
                logger.error(f"Unsupported file type: {file.content_type}")
                raise ValueError("Only PDF files are supported")
            content = await file.read()
            parsed_docs = parse_pdf(content)
            documents.extend(parsed_docs)
        status = store_document(documents)
        return DocumentUploadResponse(
            documents=documents, total=len(documents), status=status
        )
    except Exception as e:
        logger.error(f"Error uploading documents: {e}", exc_info=True)
        return DocumentUploadResponse(
            error=str(e), status="failed", documents=[], total=0
        )


@app.get("/ask")
def ask(query: str) -> AskResponse:
    """Ask a question; answer is grounded in uploaded PDF content."""
    try:
        answer = ask_question(query)
        return AskResponse(query=query, answer=answer)
    except Exception as e:
        logger.error(f"Error asking question: {e}", exc_info=True)
        return AskResponse(error=str(e), query=query, answer="")

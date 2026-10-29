"""Core RAG logic: parse PDFs, store embeddings in Chroma, answer questions."""

import logging
import os

from dotenv import find_dotenv, load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OPENAI_API_KEY from .env
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set")
    raise ValueError("OPENAI_API_KEY is not set")

# Embeddings turn text into vectors; the LLM generates answers
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# Local vector DB under ./data — persisted between runs
chroma = Chroma(
    collection_name="documents",
    collection_metadata={"name": "documents", "description": "store documents"},
    persist_directory="./data",
    embedding_function=embeddings,
)
retriever = chroma.as_retriever(search_kwargs={"k": 2})

TEMPLATE = """
Here is the context:

<context>
{context}
</context>

And here is the question that must be answered using that context:

<question>
{input}
</question>

Please read through the provided context carefully. Then, analyze the question and attempt to find a
direct answer to the question within the context.

If you are able to find a direct answer, provide it and elaborate on relevant points from the
context using bullet points "-".

If you cannot find a direct answer based on the provided context, outline the most relevant points
that give hints to the answer of the question.

If no answer or relevant points can be found, or the question is not related to the context, simply
state the following sentence without any additional text:

i couldnt find an answer did not find an answer to your question.

Output your response in plain text without using the tags <answer> and </answer> and ensure you are not
quoting context text in your response since it must not be part of the answer.
"""
PROMPT = ChatPromptTemplate.from_template(TEMPLATE)

# Retrieval chain: find relevant chunks → stuff into prompt → ask the LLM
llm_chain = create_stuff_documents_chain(llm, PROMPT)
retrieval_chain = create_retrieval_chain(retriever, llm_chain)

parser = PyPDFParser()


def store_document(documents: list[Document]) -> str:
    """Add parsed documents to the vector database."""
    chroma.add_documents(documents=documents)
    return "document stored successfully"


def parse_pdf(file_content: bytes) -> list[Document]:
    """Convert raw PDF bytes into LangChain Document objects."""
    blob = Blob(data=file_content)
    return [doc for doc in parser.lazy_parse(blob)]


def retrieve_document(query: str) -> list[Document]:
    """Return the top matching document chunks for a query."""
    return retriever.invoke(input=query)


def ask_question(query: str) -> str:
    """Answer a question using retrieved PDF context."""
    response = retrieval_chain.invoke({"input": query})
    return response["answer"]

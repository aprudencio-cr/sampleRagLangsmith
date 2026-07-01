"""
Document ingestion module.

Loads .txt files from a directory, splits them into chunks, embeds them
with Ollama nomic-embed-text, and persists a ChromaDB vector store to disk.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    TOP_K_DOCS,
)


def _get_embeddings() -> OllamaEmbeddings:
    """
    Initialize and return the OllamaEmbeddings generator.
    
    This uses the local Ollama instance configured in settings to run the nomic-embed-text
    model, converting raw textual segments into dense floating-point vector arrays.
    """
    return OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def load_documents(directory: str) -> list:
    """
    Load all source .txt files from the target directory and return a list of Documents.

    Uses LangChain's DirectoryLoader combined with TextLoader to recursively inspect
    directories, parse text files using UTF-8 encoding, and compile them into LangChain Document objects.
    """
    loader = DirectoryLoader(
        directory,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    if not docs:
        raise ValueError(f"No .txt files found in '{directory}'")
    print(f"[ingest] Loaded {len(docs)} document(s) from '{directory}'")
    return docs


def split_documents(docs: list) -> list:
    """
    Segment loaded documents into smaller overlapping text chunks.

    Uses RecursiveCharacterTextSplitter to split texts dynamically based on paragraph,
    sentence, and word boundaries. This ensures semantic continuity and prevents splitting
    essential phrases across separate chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Split into {len(chunks)} chunk(s)")
    return chunks


def build_vector_store(chunks: list) -> Chroma:
    """
    Create a new ChromaDB vector store collection and persist it on the local disk.

    Computes vector embeddings for each document chunk via the local Ollama embeddings model,
    populates the database, and saves the sqlite3 index to the persistence directory path.
    """
    embeddings = _get_embeddings()
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"[ingest] Vector store persisted at '{CHROMA_PERSIST_DIR}'")
    return store


def load_vector_store() -> Chroma:
    """
    Load and return an existing ChromaDB vector database index from disk.

    Throws a FileNotFoundError if the database directory doesn't exist yet, guiding
    developers to run the document ingestion process first.
    """
    if not Path(CHROMA_PERSIST_DIR).exists():
        raise FileNotFoundError(
            f"No vector store found at '{CHROMA_PERSIST_DIR}'. "
            "Run scripts/ingest.py first."
        )
    embeddings = _get_embeddings()
    store = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return store


def get_retriever(store: Chroma):
    """
    Return a LangChain retriever wrapper configured to search the database.

    Configures similarity search properties, telling ChromaDB to fetch exactly `TOP_K_DOCS`
    (default: 4) chunks containing the highest semantic similarity to a given user query.
    """
    return store.as_retriever(search_kwargs={"k": TOP_K_DOCS})

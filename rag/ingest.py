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
    return OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def load_documents(directory: str) -> list:
    """Load all .txt files from *directory* and return a list of Documents."""
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
    """Chunk documents for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Split into {len(chunks)} chunk(s)")
    return chunks


def build_vector_store(chunks: list) -> Chroma:
    """Embed chunks and persist a ChromaDB collection to disk."""
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
    """Load an existing ChromaDB collection from disk."""
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
    """Return a retriever that fetches TOP_K_DOCS chunks per query."""
    return store.as_retriever(search_kwargs={"k": TOP_K_DOCS})

"""
scripts/ingest.py

One-shot ingestion script. Loads sample documents, embeds them with
Ollama nomic-embed-text, and persists the ChromaDB vector store to disk.

Usage:
    python scripts/ingest.py
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag.config  # noqa: F401 — loads .env and sets LangSmith env vars

from rag.ingest import build_vector_store, load_documents, split_documents
# Define the path to your sample documents
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_docs")


def main() -> None:
    """
    Orchestrates the document ingestion pipeline.
    
    This function runs through three main steps:
      1. Loads raw .txt files from the target directory.
      2. Splits the documents into smaller text chunks for vector embedding.
      3. Passes the chunks to ChromaDB to generate embeddings and persist them on disk.
    """
    print("=" * 60)
    print("RAG INGESTION")
    print("=" * 60)
    print(f"Source directory : {DOCS_DIR}")
    print()

    # Step 1: Load raw document files using directory text loader
    docs = load_documents(DOCS_DIR)
    
    # Step 2: Split loaded texts into overlapping chunks using RecursiveCharacterTextSplitter
    chunks = split_documents(docs)
    
    # Step 3: Embed text chunks using nomic-embed-text via Ollama and persist to ChromaDB
    build_vector_store(chunks)

    print()
    print("[ingest] Done! Vector store is ready for queries.")


if __name__ == "__main__":
    main()

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

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_docs")


def main() -> None:
    print("=" * 60)
    print("RAG INGESTION")
    print("=" * 60)
    print(f"Source directory : {DOCS_DIR}")
    print()

    docs = load_documents(DOCS_DIR)
    chunks = split_documents(docs)
    build_vector_store(chunks)

    print()
    print("[ingest] Done! Vector store is ready for queries.")


if __name__ == "__main__":
    main()

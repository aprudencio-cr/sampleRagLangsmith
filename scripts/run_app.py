"""
scripts/run_app.py

Interactive CLI for the RAG pipeline.
Every query is automatically traced to LangSmith.

Usage:
    python scripts/run_app.py

Type 'quit' or 'exit' to stop.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag.config  # noqa: F401 — loads .env and sets LangSmith env vars

from rag.ingest import get_retriever, load_vector_store
from rag.pipeline import rag_pipeline


def main() -> None:
    print("=" * 60)
    print("RAG APP — powered by Ollama + LangSmith")
    print("=" * 60)
    print("Loading vector store...")
    store = load_vector_store()
    retriever = get_retriever(store)
    print("Ready! Traces are being sent to LangSmith.\n")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        result = rag_pipeline(question=question, retriever=retriever)

        print(f"\nAnswer:\n{result['answer']}\n")
        print(f"Sources ({len(result['source_documents'])} chunk(s) retrieved):")
        for i, chunk in enumerate(result["source_documents"], 1):
            preview = chunk[:150].replace("\n", " ")
            print(f"  [{i}] {preview}...")
        print()


if __name__ == "__main__":
    main()

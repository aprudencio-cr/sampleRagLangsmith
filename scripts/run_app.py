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
    """
    Main interactive entry point for testing the RAG application locally.
    
    This function:
      1. Loads the persisted Chroma DB vector store from disk.
      2. Creates a retriever to pull context documents matching user questions.
      3. Launches a standard terminal-based interactive read-eval-print loop (REPL).
      4. Invokes the decorated traceable pipeline, sending spans/traces automatically to LangSmith.
    """
    print("=" * 60)
    print("RAG APP - powered by Ollama + LangSmith")
    print("=" * 60)
    print("Loading vector store...")
    
    # 1. Load the ChromaDB collection persisted locally in chroma_db/
    store = load_vector_store()
    
    # 2. Get retriever configured to fetch the top-k nearest document chunks
    retriever = get_retriever(store)
    
    print("Ready! Traces are being sent to LangSmith.\n")
    print("Type 'quit' or 'exit' to stop.\n")

    # 3. Interactive loop
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

        # 4. Invoke the @traceable RAG pipeline. This runs similarity search,
        # formats the prompt context, executes Ollama LLM, and sends traces to LangSmith.
        result = rag_pipeline(question=question, retriever=retriever)

        # 5. Output response and show source document previews used during generation
        print(f"\nAnswer:\n{result['answer']}\n")
        print(f"Sources ({len(result['source_documents'])} chunk(s) retrieved):")
        for i, chunk in enumerate(result["source_documents"], 1):
            preview = chunk[:150].replace("\n", " ")
            print(f"  [{i}] {preview}...")
        print()


if __name__ == "__main__":
    main()

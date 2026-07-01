"""
scripts/push_dataset.py

One-shot script: creates the "RAG Eval Dataset" in LangSmith with 20
curated Q&A examples derived from the sample documents.

Usage:
    python scripts/push_dataset.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag.config  # noqa: F401 — loads .env and sets LangSmith env vars

from langsmith import Client
from langsmith_utils.push_dataset import push_dataset


def main() -> None:
    print("=" * 60)
    print("LANGSMITH DATASET PUSH")
    print("=" * 60)
    client = Client()
    push_dataset(client=client)
    print("\nDone! Open LangSmith UI -> Datasets to view your examples.")


if __name__ == "__main__":
    main()

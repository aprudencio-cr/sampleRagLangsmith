"""
scripts/run_evaluation.py

Runs the full LangSmith evaluation suite against the "RAG Eval Dataset".
Results are printed to stdout and also visible in LangSmith UI → Experiments.

Usage:
    python scripts/run_evaluation.py [experiment_prefix]

Example:
    python scripts/run_evaluation.py rag-v2
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag.config  # noqa: F401 — loads .env and sets LangSmith env vars

from langsmith_utils.evaluate import run_evaluation


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "rag-v1"

    print("=" * 60)
    print(f"LANGSMITH EVALUATION  (experiment prefix: {prefix})")
    print("=" * 60)
    run_evaluation(experiment_prefix=prefix)


if __name__ == "__main__":
    main()

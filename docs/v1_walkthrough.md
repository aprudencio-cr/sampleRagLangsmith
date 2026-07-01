# Walkthrough: RAG LangSmith Upgrades and Documentation

This document walks through all changes made to make the codebase fully compatible with modern **LangChain 0.3+** and **LangSmith 0.9.x+** versions, clean up evaluation scripts, and implement automatic dataset deduplication.

## Changes Made

### 1. Namespace Migrations (`langchain_text_splitters`)
- Modified [ingest.py](file:///c:/Users/aprud/Documents/code/rag-langsmith/rag/ingest.py#L13) to import `RecursiveCharacterTextSplitter` from `langchain_text_splitters` instead of the deprecated `langchain.text_splitter`. This resolved the `ModuleNotFoundError`.

### 2. Custom `LangChainStringEvaluator` Wrapper
- Since `LangChainStringEvaluator` was deprecated and removed in LangSmith 0.9.3, we implemented a custom adapter in [evaluate.py](file:///c:/Users/aprud/Documents/code/rag-langsmith/langsmith_utils/evaluate.py#L19-L60) that subclasses `RunEvaluator`.
- The custom class dynamically loads built-in criteria evaluators (e.g., `labeled_score_string` and `score_string`) from the **`langchain-classic`** package using `load_evaluator`.
- Configured signature parameters to accept `**kwargs` (like `evaluator_run_id` passed by the LangSmith runner).

### 3. Unicode Console Encoding Fixes
- Replaced the Unicode arrow symbol (`→`) with a standard ASCII arrow (`->`) in:
  - [push_dataset.py](file:///c:/Users/aprud/Documents/code/rag-langsmith/scripts/push_dataset.py#L28)
  - [evaluate.py](file:///c:/Users/aprud/Documents/code/rag-langsmith/langsmith_utils/evaluate.py#L293)
- This prevents `UnicodeEncodeError` crashes on Windows command lines utilizing the default `CP1252` encoding.

### 4. Automatic Dataset Deduplication
- Upgraded the dataset initialization logic in [push_dataset.py](file:///c:/Users/aprud/Documents/code/rag-langsmith/langsmith_utils/push_dataset.py#L224-L261) to query existing examples.
- It automatically deletes any duplicate examples (by matching question text) using the LangSmith client's `delete_example` SDK call.
- It only uploads new or missing questions, making the script completely idempotent and duplicate-safe.

### 5. Highly Detailed Script & Utility Documentation
- Added detailed comments and inline explanations across all execution scripts:
  - [ingest.py](file:///c:/Users/aprud/Documents/code/rag-langsmith/scripts/ingest.py)
  - [push_dataset.py](file:///c:/Users/aprud/Documents/code/rag-langsmith/scripts/push_dataset.py)
  - [run_app.py](file:///c:/Users/aprud/Documents/code/rag-langsmith/scripts/run_app.py)
  - [run_evaluation.py](file:///c:/Users/aprud/Documents/code/rag-langsmith/scripts/run_evaluation.py)
- Created the following workspace documentation files:
  - [scripts/README.md](file:///c:/Users/aprud/Documents/code/rag-langsmith/scripts/README.md) - detailing script descriptions, inner workflows, and usage commands.
  - [langsmith_utils/README.md](file:///c:/Users/aprud/Documents/code/rag-langsmith/langsmith_utils/README.md) - explaining the core utilities for dataset initialization and evaluation runner logic.

---

## Verification Results

### 1. Ingestion Execution
Running `python scripts/ingest.py` successfully loads files, splits them, and indexes them in ChromaDB using local Ollama embeddings:
```text
============================================================
RAG INGESTION
============================================================
Source directory : C:\Users\aprud\Documents\code\rag-langsmith\data\sample_docs

100%|##########| 1/1 [00:00<00:00, 117.31it/s]
[ingest] Loaded 1 document(s) from 'C:\Users\aprud\Documents\code\rag-langsmith\data\sample_docs'
[ingest] Split into 26 chunk(s)
[ingest] Vector store persisted at './chroma_db'

[ingest] Done! Vector store is ready for queries.
```

### 2. Dataset Push & Deduplication
Running `python scripts/push_dataset.py` correctly detects and removes any pre-existing duplicate questions from the dataset container:
```text
============================================================
LANGSMITH DATASET PUSH
============================================================
[push_dataset] Dataset 'RAG Eval Dataset' already exists.
[push_dataset] Cleaned up 20 duplicate example(s) from 'RAG Eval Dataset'.
[push_dataset] All examples are already present in 'RAG Eval Dataset'. No new examples to push.

Done! Open LangSmith UI -> Datasets to view your examples.
```

### 3. Evaluation Pipeline Execution
Running `python scripts/run_evaluation.py` executes 40 evaluation iterations using local `llama3:latest` LLM-as-a-judge criteria, printing Correctness, Groundedness, and Relevance scores:
```text
Q: What is a LangChain retriever?
A: A LangChain retriever is an interface that, given a query string, returns a list of relevant documents.
   [correctness] score=1.00
   [groundedness] score=1.00
   [relevance] score=0.90

...

[evaluate] Full results available in LangSmith UI -> Experiments tab
```

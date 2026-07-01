# LangSmith Utilities

This directory contains the underlying library logic for configuring, populating, and evaluating the RAG pipeline using LangSmith.

## Modules

### 1. [push_dataset.py](./push_dataset.py)
- **Purpose**: Defines the list of Q&A evaluation examples and handles uploading them to LangSmith.
- **Key Functions**:
  - `push_dataset(client: Client | None = None)`: Connects to LangSmith, verifies or creates the `"RAG Eval Dataset"` container, and populates it with 20 curated evaluation pairs.

### 2. [evaluate.py](./evaluate.py)
- **Purpose**: Defines the evaluation metrics, LLM-as-a-judge setup, and triggers the automated comparison runs.
- **Key Classes & Functions**:
  - `LangChainStringEvaluator(RunEvaluator)`: A custom compatibility wrapper class designed for LangChain >= 0.3.0 and LangSmith >= 0.9.3. It bridges the modern LangSmith test runner with legacy `StringEvaluator` structures using `langchain-classic`'s loader logic.
  - `run_evaluation(experiment_prefix: str)`: Loads the ChromaDB vector store, builds three LLM-as-a-judge evaluators (Correctness, Groundedness, Relevance) using the local Ollama LLM, runs the evaluations against your LangSmith dataset, and outputs a formatted performance summary.

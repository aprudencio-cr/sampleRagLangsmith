# RAG App with LangSmith Integration — Spec & Implementation Plan

A simple, end-to-end RAG (Retrieval-Augmented Generation) application built in Python, wired to LangSmith for tracing, dataset management, and automated evaluation.

---

## Goal

Build a working RAG pipeline that:
1. **Ingests** documents into a vector store
2. **Answers** user questions using retrieved context + an LLM
3. **Traces** every run to LangSmith automatically
4. **Pushes** a curated Q&A dataset to LangSmith
5. **Evaluates** the pipeline against that dataset using LLM-as-a-Judge metrics

---

## Project Structure

```
rag-langsmith/
├── .env                          # API keys (gitignored)
├── requirements.txt              # Python dependencies
├── README.md
│
├── rag/
│   ├── __init__.py
│   ├── ingest.py                 # Document loading & vector store setup
│   ├── pipeline.py               # RAG chain (@traceable)
│   └── config.py                 # Shared settings / env loading
│
├── langsmith_utils/
│   ├── __init__.py
│   ├── push_dataset.py           # Create & populate LangSmith dataset
│   └── evaluate.py               # Run evaluations against the dataset
│
├── data/
│   └── sample_docs/              # Sample .txt / .pdf source documents
│       └── ai_overview.txt
│
└── scripts/
    ├── run_app.py                # Interactive CLI to query the RAG pipeline
    ├── push_dataset.py           # One-shot: push dataset to LangSmith
    └── run_evaluation.py         # One-shot: run full evaluation suite
```

---

## Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| **Orchestration** | LangChain (`langchain`, `langchain-openai`) | First-class LangSmith support |
| **Embeddings** | Ollama `nomic-embed-text` | Local, free, high quality |
| **Vector Store** | ChromaDB (persisted to `./chroma_db/`) | Zero infra, survives restarts |
| **LLM** | Ollama `llama3.2` (or `mistral`) | Local, free, no API key needed |
| **Tracing** | LangSmith SDK (`langsmith`) | Native integration |
| **Evaluation** | LangSmith `evaluate()` + custom evaluators | Built-in LLM-as-a-Judge |
| **Config** | `python-dotenv` | Standard `.env` management |

---

## Core Modules

### 1. `rag/config.py` — Environment & Settings

Loads `.env` and exports a typed settings object. Sets the three required LangSmith env vars:
- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY=...`
- `LANGSMITH_PROJECT=rag-langsmith-demo`
- `OPENAI_API_KEY=...`

---

### 2. `rag/ingest.py` — Document Ingestion

```
load_documents(directory: str) -> list[Document]
build_vector_store(docs: list[Document]) -> Chroma
```

- Loads `.txt` files from `data/sample_docs/`
- Splits with `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=50)
- Embeds and stores in a local ChromaDB collection
- Returns a retriever ready to plug into the pipeline

---

### 3. `rag/pipeline.py` — RAG Pipeline

```python
@traceable(name="rag_pipeline")
def rag_pipeline(question: str, retriever) -> dict:
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    answer = llm.invoke(RAG_PROMPT.format(context=context, question=question))
    return {
        "answer": answer.content,
        "source_documents": [d.page_content for d in docs],
    }
```

The `@traceable` decorator automatically logs inputs, retrieved docs, and the final answer to LangSmith on every call.

---

### 4. `langsmith_utils/push_dataset.py` — Dataset Management

Creates a LangSmith dataset named `"RAG Eval Dataset"` and populates it with curated Q&A pairs derived from the sample documents:

```python
examples = [
    {
        "inputs":  {"question": "What is RAG?"},
        "outputs": {"answer": "RAG stands for Retrieval-Augmented Generation..."},
    },
    ...  # ~10 examples covering the sample docs
]
client.create_dataset(dataset_name="RAG Eval Dataset")
client.create_examples(dataset_id=..., examples=examples)
```

---

### 5. `langsmith_utils/evaluate.py` — Evaluation Suite

Runs `langsmith.evaluate()` against the `"RAG Eval Dataset"` using three evaluators:

| Evaluator | What it checks |
|---|---|
| **Correctness** | Does the answer match the reference answer? (LLM-as-a-judge) |
| **Groundedness** | Is the answer supported by retrieved docs? (no hallucination) |
| **Retrieval Relevance** | Did the retriever fetch useful documents? |

```python
from langsmith import evaluate

results = evaluate(
    lambda inputs: rag_pipeline(inputs["question"], retriever),
    data="RAG Eval Dataset",
    evaluators=[correctness_evaluator, groundedness_evaluator, relevance_evaluator],
    experiment_prefix="rag-v1",
)
```

Results are viewable in the LangSmith UI under the **Experiments** tab.

---

## `.env` Template

```
# No OpenAI key needed — using Ollama locally
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=rag-langsmith-demo
LANGSMITH_TRACING=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
```

---

## `requirements.txt`

```
langchain>=0.3.0
langchain-ollama>=0.2.0
langchain-community>=0.3.0
langsmith>=0.2.0
chromadb>=0.5.0
python-dotenv>=1.0.0
```

---

## Scripts

### `scripts/run_app.py`
Interactive CLI loop — prompts for a question, runs the RAG pipeline, prints the answer. Traces are automatically sent to LangSmith.

### `scripts/push_dataset.py`
One-shot script. Run once to create and populate the LangSmith evaluation dataset.

### `scripts/run_evaluation.py`
One-shot script. Runs the full evaluation suite and prints a summary of scores to stdout. Full results appear in LangSmith UI.

---

## Sample Data

`data/sample_docs/ai_overview.txt` — A ~500-word overview of AI, LLMs, and RAG. This ensures the evaluation dataset questions have correct, verifiable answers in the vector store.

---

## Verification Plan

### Step 1 — Smoke test the RAG pipeline
```bash
python scripts/run_app.py
```
- Ask: *"What is RAG?"*
- Expected: coherent answer pulled from `ai_overview.txt`
- Check LangSmith UI → **Projects → rag-langsmith-demo** → see a new trace with retrieved docs

### Step 2 — Push the dataset
```bash
python scripts/push_dataset.py
```
- Check LangSmith UI → **Datasets** → `"RAG Eval Dataset"` with 10 examples

### Step 3 — Run evaluation
```bash
python scripts/run_evaluation.py
```
- Check LangSmith UI → **Experiments** → `rag-v1-*` experiment
- Inspect per-example scores for Correctness, Groundedness, and Relevance

---

## Resolved Decisions

| Decision | Choice |
|---|---|
| **LLM** | Ollama (`llama3.2` for generation, `nomic-embed-text` for embeddings) — fully local |
| **Documents** | Synthetic `ai_overview.txt` generated to cover AI, RAG, LangChain, LangSmith topics |
| **Vector Store** | ChromaDB persisted to `./chroma_db/` on disk — zero infra, survives restarts |
| **Dataset Size** | 20 curated Q&A examples |

> [!IMPORTANT]
> **Prerequisite**: Ollama must be installed and running (`ollama serve`). Pull the required models before running:
> ```bash
> ollama pull llama3.2
> ollama pull nomic-embed-text
> ```

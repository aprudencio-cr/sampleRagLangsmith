# RAG + LangSmith Demo

A simple, end-to-end **Retrieval-Augmented Generation** (RAG) application wired to **LangSmith** for tracing, dataset management, and automated evaluation.

**Stack**: Python · LangChain · Ollama (local LLM) · ChromaDB · LangSmith

---

## Prerequisites

| Tool | Purpose | Install |
|---|---|---|
| Python 3.10+ | Runtime | [python.org](https://python.org) |
| Ollama | Local LLM server | [ollama.com](https://ollama.com) |
| LangSmith account | Tracing & evaluation | [smith.langchain.com](https://smith.langchain.com) |

### Pull Ollama models

```bash
ollama pull llama3            # generation (e.g., llama3, llama3.2, or llama3:latest)
ollama pull nomic-embed-text  # embeddings
```

Make sure Ollama is running:

```bash
ollama serve
```

---

## Setup

```bash
# 1. Clone / navigate to the project
cd rag-langsmith

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env and add your LANGSMITH_API_KEY
```

---

## Usage

### Step 1 — Ingest documents

Embeds the sample documents and persists the ChromaDB vector store to `./chroma_db/`.

```bash
python scripts/ingest.py
```

### Step 2 — Run the RAG app

Interactive Q&A loop. Every query is automatically traced to LangSmith.

```bash
python scripts/run_app.py
```

### Step 3 — Push the evaluation dataset

One-shot: creates the **"RAG Eval Dataset"** in LangSmith with 20 curated examples.

```bash
python scripts/push_dataset.py
```

### Step 4 — Run evaluation

Evaluates the pipeline against the dataset using three LLM-as-a-judge metrics.

```bash
python scripts/run_evaluation.py          # experiment prefix: rag-v1
python scripts/run_evaluation.py rag-v2   # optional: custom prefix
```

Results appear in **LangSmith UI → Experiments**.

---

## Project Structure

```
rag-langsmith/
├── .env.example               # Environment variable template
├── requirements.txt
│
├── rag/
│   ├── config.py              # Settings & env loading
│   ├── ingest.py              # Document loading, chunking, ChromaDB
│   └── pipeline.py            # @traceable RAG pipeline
│
├── langsmith_utils/
│   ├── push_dataset.py        # Create & populate LangSmith dataset
│   └── evaluate.py            # Run evaluation suite
│
├── data/
│   └── sample_docs/
│       └── ai_overview.txt    # Synthetic knowledge base
│
└── scripts/
    ├── ingest.py              # Run ingestion
    ├── run_app.py             # Interactive CLI
    ├── push_dataset.py        # Push dataset to LangSmith
    └── run_evaluation.py      # Run evaluation suite
```

---

## Compatibility & Technical Notes

- **Modern LangChain & LangSmith Support**: Updated to work seamlessly with **LangChain 0.3+** and **LangSmith 0.9+**.
- **Text Splitters**: Replaced deprecated `langchain.text_splitter` imports with `langchain_text_splitters`.
- **Custom Evaluator Adapter**: Since `LangChainStringEvaluator` was deprecated and removed in LangSmith 0.9.3, we added a custom `LangChainStringEvaluator` wrapper in [evaluate.py](./langsmith_utils/evaluate.py). It uses `langchain_classic.evaluation.load_evaluator` under the hood to preserve the exact same LLM-as-a-judge grading pipeline structure.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Correctness** | Does the answer match the reference? (LLM-as-a-judge) |
| **Groundedness** | Is the answer supported by retrieved docs? (no hallucination) |
| **Retrieval Relevance** | Did the retriever fetch useful chunks? |

---

## LangSmith UI Checklist

After running the steps above, verify in the LangSmith UI:

- **Projects → rag-langsmith-demo** — traces from every `run_app.py` query
- **Datasets → RAG Eval Dataset** — 20 Q&A examples
- **Experiments → rag-v1-*** — per-example scores for all three metrics

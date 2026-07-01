# Scripts Directory

This folder contains Python scripts to orchestrate the ingestion, interaction, dataset configuration, and evaluation of the RAG pipeline.

## Overview of Scripts

### 1. [ingest.py](./ingest.py)
- **Purpose**: A one-shot ingestion script to construct the vector database.
- **Workflow**:
  1. Loads raw `.txt` files from `data/sample_docs/`.
  2. Splits the document text into smaller chunks using `RecursiveCharacterTextSplitter`.
  3. Embeds each text chunk using the `nomic-embed-text` Ollama model.
  4. Saves and persists the embeddings into a local ChromaDB collection stored at `./chroma_db/`.
- **Usage**:
  ```bash
  python scripts/ingest.py
  ```

### 2. [run_app.py](./run_app.py)
- **Purpose**: An interactive command-line interface (CLI) to query the RAG pipeline.
- **Workflow**:
  1. Loads the persisted ChromaDB vector store.
  2. Starts an interactive chat loop prompting the user for questions.
  3. For each question, retrieves the most relevant chunks, generates the response using the local Ollama LLM, and displays the source references.
  4. Automatically traces execution to **LangSmith** via the `@traceable` pipeline decorator.
- **Usage**:
  ```bash
  python scripts/run_app.py
  ```
  *(Type `exit` or `quit` to end the session)*

### 3. [push_dataset.py](./push_dataset.py)
- **Purpose**: Initializes the evaluation dataset in your LangSmith project.
- **Workflow**:
  1. Connects to LangSmith via the client SDK.
  2. Creates a dataset named **"RAG Eval Dataset"** if it doesn't already exist.
  3. Pushes 20 curated question-and-answer pairs derived from the sample documents.
- **Usage**:
  ```bash
  python scripts/push_dataset.py
  ```

### 4. [run_evaluation.py](./run_evaluation.py)
- **Purpose**: Executes the automated evaluation suite against the "RAG Eval Dataset".
- **Workflow**:
  1. Runs the RAG pipeline on all 20 question-and-answer examples in the dataset.
  2. Evaluates the pipeline's answers using three LLM-as-a-judge criteria:
     - **Correctness**: Checks if the answer correctly addresses the question compared to the reference answer.
     - **Groundedness**: Evaluates whether the generated answer is fully supported by the retrieved document chunks (detecting hallucination).
     - **Retrieval Relevance**: Grades whether the retrieved context was relevant and useful.
  3. Uses a local Ollama LLM as the judge model for grading.
  4. Outputs a local CLI summary of the evaluation scores and registers the experiment traces in the LangSmith UI.
- **Usage**:
  ```bash
  python scripts/run_evaluation.py [experiment_prefix]
  ```
  *(Default experiment prefix is `rag-v1` if not specified)*

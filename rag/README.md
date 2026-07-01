# RAG Core Package

This folder contains the core modules of the Retrieval-Augmented Generation (RAG) system.

## Modules

### 1. [config.py](./config.py)
- **Purpose**: Loads environment variables from the `.env` file and initializes shared settings (URLs, model configurations, persist directories, and chunk sizes).
- **Key Settings**: Sets LangSmith environment variables (`LANGSMITH_TRACING`, `LANGSMITH_API_KEY`) to ensure automatic execution tracing.

### 2. [ingest.py](./ingest.py)
- **Purpose**: Manages document loading, splitting, and database indexing.
- **Key Functions**:
  - `load_documents(directory: str)`: Loads raw `.txt` files recursively.
  - `split_documents(docs: list)`: Chunks text using `RecursiveCharacterTextSplitter`.
  - `build_vector_store(chunks: list)`: Embeds text segments and indexes them in ChromaDB.
  - `load_vector_store()`: Loads the existing SQLite DB index from disk.
  - `get_retriever(store: Chroma)`: Returns the similarity search retrieval interface.

### 3. [pipeline.py](./pipeline.py)
- **Purpose**: Exposes the main RAG querying logic, decorated with `@traceable` for automatic observability tracing.
- **Key Functions**:
  - `rag_pipeline(question: str, retriever)`: Coordinates context retrieval, prompt formatting, LLM execution, and returns structured answer payloads.

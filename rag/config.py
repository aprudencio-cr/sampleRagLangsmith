"""
Shared configuration module.

This module is responsible for:
  1. Loading environment variables from a local `.env` file via `dotenv`.
  2. Bootstrapping the LangSmith environment variables *before* any LangChain libraries
     are loaded (ensuring automatic tracing registers correctly).
  3. Exporting typed configuration settings used throughout the RAG application.
"""

import os
from dotenv import load_dotenv

# Load key-value pairs from a local .env file into standard os.environ
load_dotenv()

# ===========================================================================
# LangSmith Configuration
# These environment variables configure the LangSmith client SDK under the hood.
# Setting LANGSMITH_TRACING to "true" forces LangChain to emit traces for all runs.
# ===========================================================================
os.environ.setdefault("LANGSMITH_TRACING", os.getenv("LANGSMITH_TRACING", "true"))
os.environ.setdefault("LANGSMITH_API_KEY", os.getenv("LANGSMITH_API_KEY", ""))
os.environ.setdefault("LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "rag-langsmith-demo"))

# ===========================================================================
# Ollama Configuration
# Configures connections to the local LLM server running on your system.
# ===========================================================================
# OLLAMA_BASE_URL: The endpoint where the local Ollama daemon is serving queries (default 11434).
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# OLLAMA_MODEL: The generation model (Ollama LLM) to invoke for answering user questions.
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

# OLLAMA_EMBED_MODEL: The embedding model used to represent documents and queries as vector arrays.
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ===========================================================================
# Vector Store Configuration
# ===========================================================================
# CHROMA_PERSIST_DIR: The filesystem directory where the database files are saved to disk.
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# CHROMA_COLLECTION: The database table/collection name where text chunks and vectors are indexed.
CHROMA_COLLECTION: str = "rag_docs"

# ===========================================================================
# Text Segmentation & Retrieval Chunking
# Adjusts the size of textual chunks passed to the embeddings engine.
# ===========================================================================
# CHUNK_SIZE: Maximum character length of each individual document chunk segment.
CHUNK_SIZE: int = 500

# CHUNK_OVERLAP: Character overlap between consecutive chunks to ensure context continuity.
CHUNK_OVERLAP: int = 50

# TOP_K_DOCS: Number of nearest-neighbor document chunks the retriever should return for a query.
TOP_K_DOCS: int = 4

# ===========================================================================
# LangSmith Benchmark Dataset Settings
# ===========================================================================
# DATASET_NAME: The dataset name under which test examples are stored in LangSmith.
DATASET_NAME: str = "RAG Eval Dataset"


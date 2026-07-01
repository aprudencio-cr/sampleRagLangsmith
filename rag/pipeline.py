"""
RAG pipeline module.

Exposes a single @traceable function `rag_pipeline` that:
  1. Retrieves relevant chunks from ChromaDB
  2. Formats them into a prompt
  3. Calls the local Ollama LLM
  4. Returns answer + source documents

Every call is automatically traced to LangSmith via the @traceable decorator.
"""

from __future__ import annotations

from langsmith import traceable
from langchain_ollama import OllamaLLM

from rag.config import OLLAMA_BASE_URL, OLLAMA_MODEL

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
RAG_PROMPT = """\
You are a helpful assistant. Answer the question using ONLY the context below.
If the context does not contain enough information, say "I don't know based on the provided context."

Context:
{context}

Question: {question}

Answer:"""

# ---------------------------------------------------------------------------
# LLM (singleton — reused across calls)
# ---------------------------------------------------------------------------
_llm: OllamaLLM | None = None


def _get_llm() -> OllamaLLM:
    """
    Load and return a singleton instance of the Ollama LLM generator.
    
    Reusing a single instance of OllamaLLM prevents reloading the model settings
    into memory on every user query. We set `temperature=0.0` to eliminate random
    variance, which is essential for consistent and reliable automated evaluation.
    """
    global _llm
    if _llm is None:
        _llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,  # deterministic for evaluation
        )
    return _llm


# ---------------------------------------------------------------------------
# Main pipeline — decorated with @traceable for LangSmith tracing
# ---------------------------------------------------------------------------
@traceable(name="rag_pipeline")
def rag_pipeline(question: str, retriever) -> dict:
    """
    Run a single Retrieval-Augmented Generation (RAG) turn.

    The @traceable decorator wraps this execution block. When called, the LangSmith SDK
    automatically spawns a trace session, recording the input question, nested steps (such as
    document retrieval and LLM calls), execution latency, and the returned dictionary.

    Args:
        question: The user's question string.
        retriever: A LangChain retriever instance configured for similarity search.

    Returns:
        dict: A structured dictionary containing:
            - "answer": The generated answer string.
            - "source_documents": List of raw document chunk texts retrieved from ChromaDB.
    """
    # Step 1: Retrieve context chunks semantically similar to the input question from ChromaDB
    docs = retriever.invoke(question)
    source_texts = [d.page_content for d in docs]

    # Step 2: Build a structured context block by joining retrieved text chunks
    context = "\n\n---\n\n".join(source_texts)

    # Step 3: Populate the RAG prompt template inserting both context and question
    prompt = RAG_PROMPT.format(context=context, question=question)
    
    # Step 4: Fetch the Ollama LLM generator and invoke it with the compiled prompt
    llm = _get_llm()
    answer: str = llm.invoke(prompt)

    # Step 5: Return structured response containing the answer and source documents
    return {
        "answer": answer.strip(),
        "source_documents": source_texts,
    }

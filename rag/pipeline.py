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
    Run a single RAG turn.

    Args:
        question: The user's question.
        retriever: A LangChain retriever (from get_retriever()).

    Returns:
        dict with keys:
            - "answer": the LLM's response string
            - "source_documents": list of retrieved chunk texts
    """
    # Retrieve relevant chunks
    docs = retriever.invoke(question)
    source_texts = [d.page_content for d in docs]

    # Build context string
    context = "\n\n---\n\n".join(source_texts)

    # Call LLM
    prompt = RAG_PROMPT.format(context=context, question=question)
    llm = _get_llm()
    answer: str = llm.invoke(prompt)

    return {
        "answer": answer.strip(),
        "source_documents": source_texts,
    }

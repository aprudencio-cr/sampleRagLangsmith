"""
Push a curated Q&A evaluation dataset to LangSmith.

Run once:
    python scripts/push_dataset.py

Creates (or updates) a dataset named "RAG Eval Dataset" in your LangSmith
project with 20 examples derived from the synthetic sample documents.
"""

from __future__ import annotations

from langsmith import Client

from rag.config import DATASET_NAME

# ---------------------------------------------------------------------------
# 20 curated Q&A examples aligned with data/sample_docs/ai_overview.txt
# ---------------------------------------------------------------------------
EXAMPLES: list[dict] = [
    # --- AI & LLMs ---
    {
        "inputs": {"question": "What does LLM stand for?"},
        "outputs": {"answer": "LLM stands for Large Language Model."},
    },
    {
        "inputs": {"question": "What is a large language model?"},
        "outputs": {
            "answer": (
                "A large language model is a type of artificial intelligence model "
                "trained on massive amounts of text data to understand and generate "
                "human language."
            )
        },
    },
    {
        "inputs": {"question": "What is a transformer architecture?"},
        "outputs": {
            "answer": (
                "The transformer architecture is a neural network design introduced in "
                "the paper 'Attention Is All You Need'. It uses self-attention mechanisms "
                "to process sequences in parallel, making it highly effective for NLP tasks."
            )
        },
    },
    {
        "inputs": {"question": "What is prompt engineering?"},
        "outputs": {
            "answer": (
                "Prompt engineering is the practice of designing and refining the text "
                "inputs given to a language model in order to guide its outputs toward "
                "a desired result."
            )
        },
    },
    # --- RAG ---
    {
        "inputs": {"question": "What does RAG stand for?"},
        "outputs": {"answer": "RAG stands for Retrieval-Augmented Generation."},
    },
    {
        "inputs": {"question": "What is Retrieval-Augmented Generation?"},
        "outputs": {
            "answer": (
                "Retrieval-Augmented Generation (RAG) is a technique that enhances "
                "LLM responses by first retrieving relevant documents from an external "
                "knowledge base and then using those documents as additional context "
                "when generating an answer."
            )
        },
    },
    {
        "inputs": {"question": "Why is RAG useful?"},
        "outputs": {
            "answer": (
                "RAG is useful because it allows a language model to access up-to-date "
                "or domain-specific information beyond its training data, reducing "
                "hallucination and improving factual accuracy."
            )
        },
    },
    {
        "inputs": {"question": "What are the main steps in a RAG pipeline?"},
        "outputs": {
            "answer": (
                "The main steps in a RAG pipeline are: (1) ingesting documents into a "
                "vector store, (2) retrieving the most relevant chunks for a given "
                "query, and (3) passing those chunks as context to an LLM to generate "
                "a grounded answer."
            )
        },
    },
    {
        "inputs": {"question": "What is a vector store?"},
        "outputs": {
            "answer": (
                "A vector store is a database that stores numerical vector representations "
                "(embeddings) of text and supports fast similarity search, enabling "
                "retrieval of the most semantically relevant documents for a given query."
            )
        },
    },
    {
        "inputs": {"question": "What is an embedding in the context of RAG?"},
        "outputs": {
            "answer": (
                "An embedding is a dense numerical vector representation of a piece of "
                "text. In RAG, both documents and queries are embedded so that their "
                "semantic similarity can be measured and the most relevant chunks "
                "can be retrieved."
            )
        },
    },
    # --- LangChain ---
    {
        "inputs": {"question": "What is LangChain?"},
        "outputs": {
            "answer": (
                "LangChain is an open-source Python framework for building applications "
                "powered by language models. It provides abstractions for chains, "
                "retrievers, prompts, and integrations with many LLM providers and "
                "vector stores."
            )
        },
    },
    {
        "inputs": {"question": "What is a LangChain retriever?"},
        "outputs": {
            "answer": (
                "A LangChain retriever is an interface that, given a query string, "
                "returns a list of relevant documents. It abstracts over different "
                "vector stores and search strategies."
            )
        },
    },
    {
        "inputs": {"question": "What is a LangChain chain?"},
        "outputs": {
            "answer": (
                "A LangChain chain is a sequence of components (such as a retriever, "
                "prompt template, and LLM) composed together so that the output of "
                "one step becomes the input of the next."
            )
        },
    },
    # --- LangSmith ---
    {
        "inputs": {"question": "What is LangSmith?"},
        "outputs": {
            "answer": (
                "LangSmith is a platform by LangChain for debugging, tracing, testing, "
                "and evaluating LLM applications. It logs every step of a chain or "
                "agent run and provides tools for dataset management and automated "
                "evaluation."
            )
        },
    },
    {
        "inputs": {"question": "What is tracing in LangSmith?"},
        "outputs": {
            "answer": (
                "Tracing in LangSmith refers to the automatic logging of every step "
                "in an LLM application—including retrieved documents, prompt inputs, "
                "and model outputs—so developers can inspect and debug runs in the "
                "LangSmith UI."
            )
        },
    },
    {
        "inputs": {"question": "How do you enable LangSmith tracing?"},
        "outputs": {
            "answer": (
                "You enable LangSmith tracing by setting the environment variables "
                "LANGSMITH_TRACING=true and LANGSMITH_API_KEY=<your key> before "
                "running your application."
            )
        },
    },
    {
        "inputs": {"question": "What is the @traceable decorator in LangSmith?"},
        "outputs": {
            "answer": (
                "The @traceable decorator from the langsmith SDK wraps any Python "
                "function so that its inputs, outputs, and metadata are automatically "
                "logged to LangSmith every time the function is called."
            )
        },
    },
    {
        "inputs": {"question": "What is a LangSmith dataset?"},
        "outputs": {
            "answer": (
                "A LangSmith dataset is a collection of input/output example pairs "
                "stored in LangSmith and used as a benchmark for evaluating an LLM "
                "application's performance."
            )
        },
    },
    # --- Evaluation metrics ---
    {
        "inputs": {"question": "What is groundedness in RAG evaluation?"},
        "outputs": {
            "answer": (
                "Groundedness measures whether the answer generated by the LLM is "
                "supported by and consistent with the retrieved source documents, "
                "rather than being hallucinated or fabricated."
            )
        },
    },
    {
        "inputs": {"question": "What is retrieval relevance in RAG evaluation?"},
        "outputs": {
            "answer": (
                "Retrieval relevance measures how well the documents retrieved from "
                "the vector store match the user's query—that is, whether the "
                "retrieved chunks actually contain information useful for answering "
                "the question."
            )
        },
    },
]


def push_dataset(client: Client | None = None) -> None:
    """
    Create (or update) the evaluation dataset in LangSmith.

    This function does the following:
      1. Instantiates a LangSmith Client to connect to the LangSmith backend.
      2. Queries existing datasets to check if the target dataset already exists (ensuring idempotency).
      3. Creates a new dataset if it does not exist.
      4. Pushed the curated Q&A pairs as evaluation benchmark examples.
    """
    # 1. Initialize client: Client handles authentication using LANGSMITH_API_KEY env variable
    if client is None:
        client = Client()

    # 2. Check for existing dataset: prevents creating duplicate datasets on successive script runs
    existing = [d for d in client.list_datasets() if d.name == DATASET_NAME]
    if existing:
        print(f"[push_dataset] Dataset '{DATASET_NAME}' already exists — skipping creation.")
        dataset = existing[0]
    else:
        # 3. Create dataset: creates the container in LangSmith UI under the Datasets & Testing tab
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=(
                "20 RAG evaluation Q&A pairs covering AI, LLMs, RAG, "
                "LangChain, LangSmith, and evaluation metrics."
            ),
        )
        print(f"[push_dataset] Created dataset '{DATASET_NAME}' (id={dataset.id})")

    # 4. Add examples: Pushes inputs (what the RAG pipeline receives) and outputs (the expected ground truth)
    client.create_examples(
        dataset_id=dataset.id,
        examples=EXAMPLES,
    )
    print(f"[push_dataset] Pushed {len(EXAMPLES)} examples to '{DATASET_NAME}'")


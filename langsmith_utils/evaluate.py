"""
LangSmith evaluation module.

Runs three LLM-as-a-judge evaluators against the "RAG Eval Dataset":
  1. Correctness   — does the answer match the reference?
  2. Groundedness  — is the answer supported by the retrieved docs?
  3. Relevance     — did the retriever fetch useful chunks?

Usage (via script):
    python scripts/run_evaluation.py
"""

from __future__ import annotations

from langsmith import Client, evaluate
from langsmith.evaluation import RunEvaluator, EvaluationResult
from langchain_classic.evaluation import load_evaluator

class LangChainStringEvaluator(RunEvaluator):
    """Custom wrapper for LangChain evaluators matching the deprecated LangSmith class."""
    def __init__(
        self,
        evaluator_name: str,
        config: dict | None = None,
        prepare_data: callable | None = None,
    ):
        self.evaluator_name = evaluator_name
        self.config = config or {}
        self.prepare_data = prepare_data
        self.evaluator = load_evaluator(evaluator_name, **self.config)

    def evaluate_run(self, run, example=None, **kwargs) -> EvaluationResult:
        if self.prepare_data:
            data = self.prepare_data(run, example)
        else:
            data = {
                "prediction": run.outputs.get("output", "") if run.outputs else "",
                "reference": example.outputs.get("output", "") if example and example.outputs else "",
                "input": example.inputs.get("input", "") if example and example.inputs else "",
            }

        res = self.evaluator.evaluate_strings(
            prediction=data.get("prediction"),
            reference=data.get("reference"),
            input=data.get("input"),
        )

        criteria = self.config.get("criteria")
        if isinstance(criteria, dict):
            key = list(criteria.keys())[0]
        elif isinstance(criteria, str):
            key = criteria
        else:
            key = self.evaluator_name

        return EvaluationResult(
            key=key,
            score=res.get("score"),
            comment=res.get("reasoning"),
        )


from rag.config import DATASET_NAME, OLLAMA_BASE_URL, OLLAMA_MODEL
from rag.ingest import get_retriever, load_vector_store
from rag.pipeline import rag_pipeline


# ---------------------------------------------------------------------------
# Helper: wrap the RAG pipeline so it accepts the dict format LangSmith uses
# ---------------------------------------------------------------------------
def _make_target(retriever):
    """Return a target function compatible with langsmith.evaluate()."""

    def target(inputs: dict) -> dict:
        return rag_pipeline(question=inputs["question"], retriever=retriever)

    return target


# ---------------------------------------------------------------------------
# Evaluator definitions (LLM-as-a-Judge using the local Ollama model)
# ---------------------------------------------------------------------------

def _build_evaluators():
    """
    Build three string evaluators.

    LangChainStringEvaluator wraps langchain evaluation criteria and
    automatically uses the local Ollama LLM to grade each response.
    """
    from langchain_ollama import OllamaLLM

    judge_llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )

    correctness = LangChainStringEvaluator(
        "labeled_score_string",
        config={
            "criteria": {
                "correctness": (
                    "Does the answer correctly address the question "
                    "compared to the reference answer?"
                )
            },
            "normalize_by": 10,
            "llm": judge_llm,
        },
        prepare_data=lambda run, example: {
            "prediction": run.outputs.get("answer", ""),
            "reference": example.outputs.get("answer", ""),
            "input": example.inputs.get("question", ""),
        },
    )

    groundedness = LangChainStringEvaluator(
        "score_string",
        config={
            "criteria": {
                "groundedness": (
                    "Is the answer fully supported by the provided source documents? "
                    "Penalise any claims not found in the context."
                )
            },
            "normalize_by": 10,
            "llm": judge_llm,
        },
        prepare_data=lambda run, example: {
            "prediction": run.outputs.get("answer", ""),
            "input": (
                "Source documents:\n"
                + "\n---\n".join(run.outputs.get("source_documents", []))
                + f"\n\nQuestion: {example.inputs.get('question', '')}"
            ),
        },
    )

    relevance = LangChainStringEvaluator(
        "score_string",
        config={
            "criteria": {
                "relevance": (
                    "Are the retrieved source documents relevant and useful "
                    "for answering the question?"
                )
            },
            "normalize_by": 10,
            "llm": judge_llm,
        },
        prepare_data=lambda run, example: {
            "prediction": "\n---\n".join(run.outputs.get("source_documents", [])),
            "input": example.inputs.get("question", ""),
        },
    )

    return [correctness, groundedness, relevance]


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(experiment_prefix: str = "rag-v1") -> None:
    """Run the full evaluation suite and print a summary."""
    client = Client()

    # Verify dataset exists
    datasets = [d for d in client.list_datasets() if d.name == DATASET_NAME]
    if not datasets:
        raise RuntimeError(
            f"Dataset '{DATASET_NAME}' not found. "
            "Run `python scripts/push_dataset.py` first."
        )

    print(f"[evaluate] Loading vector store...")
    store = load_vector_store()
    retriever = get_retriever(store)

    print(f"[evaluate] Building evaluators...")
    evaluators = _build_evaluators()

    print(f"[evaluate] Running evaluation against '{DATASET_NAME}'...")
    results = evaluate(
        _make_target(retriever),
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        metadata={"model": OLLAMA_MODEL, "embed_model": "nomic-embed-text"},
    )

    # Print aggregate summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for result in results:
        if hasattr(result, "example"):
            example = result.example
        elif isinstance(result, dict):
            example = result.get("example")
        else:
            example = getattr(result, "example", None)

        if hasattr(result, "run"):
            run = result.run
        elif isinstance(result, dict):
            run = result.get("run")
        else:
            run = getattr(result, "run", None)

        if hasattr(result, "evaluation_results"):
            eval_results = result.evaluation_results
        elif isinstance(result, dict):
            eval_results = result.get("evaluation_results")
        else:
            eval_results = getattr(result, "evaluation_results", None)

        inputs = {}
        if example is not None:
            if hasattr(example, "inputs"):
                inputs = example.inputs
            elif isinstance(example, dict):
                inputs = example.get("inputs", {})
            else:
                inputs = getattr(example, "inputs", {})
        q = inputs.get("question", "?") if isinstance(inputs, dict) else "?"

        outputs = {}
        if run is not None:
            if hasattr(run, "outputs"):
                outputs = run.outputs
            elif isinstance(run, dict):
                outputs = run.get("outputs", {})
            else:
                outputs = getattr(run, "outputs", {})
        
        answer = "—"
        if isinstance(outputs, dict) and outputs:
            answer = outputs.get("answer", "—")

        print(f"\nQ: {q}")
        print(f"A: {answer[:120]}{'...' if len(answer) > 120 else ''}")

        results_list = []
        if eval_results is not None:
            if hasattr(eval_results, "results"):
                results_list = eval_results.results
            elif isinstance(eval_results, dict):
                results_list = eval_results.get("results", [])
            else:
                results_list = getattr(eval_results, "results", [])

        for fb in results_list:
            fb_key = getattr(fb, "key", None) or (fb.get("key") if isinstance(fb, dict) else "?")
            fb_score = getattr(fb, "score", None) or (fb.get("score") if isinstance(fb, dict) else 0.0)
            if fb_score is not None:
                print(f"   [{fb_key}] score={fb_score:.2f}")
            else:
                print(f"   [{fb_key}] score=None")
    print("\n[evaluate] Full results available in LangSmith UI -> Experiments tab")

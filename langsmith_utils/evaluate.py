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
    """
    Custom adapter that wraps LangChain-classic string evaluators and integrates
    them with the LangSmith automated evaluation pipeline (`evaluate()`).

    This class subclasses LangSmith's `RunEvaluator`. It implements the required
    `evaluate_run` method, which is called automatically by LangSmith for each run.
    """
    def __init__(
        self,
        evaluator_name: str,
        config: dict | None = None,
        prepare_data: callable | None = None,
    ):
        """
        Args:
            evaluator_name: The name of the built-in LangChain evaluator to load
                            (e.g., "labeled_score_string" or "score_string").
            config: A dictionary of configuration options passed directly to
                    `load_evaluator` (e.g., the criteria prompt, normalize_by factor,
                    and LLM to act as the judge).
            prepare_data: A mapping callable that accepts `(run, example)` and maps
                          them to a dict containing {"prediction", "reference", "input"}.
                          This aligns the RAG pipeline run outputs and dataset outputs
                          with the keys expected by the LangChain evaluator.
        """
        self.evaluator_name = evaluator_name
        self.config = config or {}
        self.prepare_data = prepare_data
        
        # Load the corresponding LangChain evaluator model using langchain-classic
        self.evaluator = load_evaluator(evaluator_name, **self.config)

    def evaluate_run(self, run, example=None, **kwargs) -> EvaluationResult:
        """
        Called by the LangSmith runner for each test case run.

        Extracts inputs and prediction outputs, runs the loaded LangChain evaluator
        via Ollama, and packages the result into a LangSmith EvaluationResult object.
        """
        # 1. Map data to the keys the evaluator expects (prediction, reference, input)
        if self.prepare_data:
            data = self.prepare_data(run, example)
        else:
            # Fallback default mapping
            data = {
                "prediction": run.outputs.get("output", "") if run.outputs else "",
                "reference": example.outputs.get("output", "") if example and example.outputs else "",
                "input": example.inputs.get("input", "") if example and example.inputs else "",
            }

        # 2. Run evaluation logic using the underlying LangChain string evaluator.
        # This calls the LLM-as-a-judge (Ollama) with the criteria prompts
        res = self.evaluator.evaluate_strings(
            prediction=data.get("prediction"),
            reference=data.get("reference"),
            input=data.get("input"),
        )

        # 3. Extract the feedback score key (e.g., "correctness", "groundedness")
        criteria = self.config.get("criteria")
        if isinstance(criteria, dict):
            key = list(criteria.keys())[0]
        elif isinstance(criteria, str):
            key = criteria
        else:
            key = self.evaluator_name

        # 4. Return the result which will be automatically logged to the LangSmith UI
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
    """
    Return a target prediction function compatible with the LangSmith `evaluate()` call.

    LangSmith's test runner expects a prediction function that accepts a dictionary
    representing a single dataset example input (e.g., `{"question": "..."}`) and returns
    a dictionary representing the system's output (e.g., `{"answer": "...", "source_documents": [...]}`).
    
    This function wraps our RAG pipeline, maps the input question key correctly, and returns it.
    """
    def target(inputs: dict) -> dict:
        # Extract the user's question from the example row inputs
        return rag_pipeline(question=inputs["question"], retriever=retriever)

    return target


# ---------------------------------------------------------------------------
# Evaluator definitions (LLM-as-a-Judge using the local Ollama model)
# ---------------------------------------------------------------------------

def _build_evaluators():
    """
    Configure and build the LLM-as-a-judge string evaluators.

    We define three evaluators that grade the pipeline using a local Ollama LLM:
      1. Correctness (labeled_score_string): Compares prediction to the reference answer.
      2. Groundedness (score_string): Checks if prediction is supported by retrieved context.
      3. Retrieval Relevance (score_string): Checks if retrieved context matches the question.
    """
    from langchain_ollama import OllamaLLM

    # Initialize the LLM judge model. We set temperature=0.0 to ensure consistent,
    # reproducible grading criteria.
    judge_llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )

    # 1. Correctness Evaluator: Grades whether the model's answer is accurate compared to reference
    correctness = LangChainStringEvaluator(
        "labeled_score_string", # Labeled evaluators compare prediction against a ground truth "reference"
        config={
            "criteria": {
                "correctness": (
                    "Does the answer correctly address the question "
                    "compared to the reference answer?"
                )
            },
            "normalize_by": 10, # LangChain scores are 1-10; dividing by 10 normalizes scores to [0.0, 1.0]
            "llm": judge_llm,
        },
        prepare_data=lambda run, example: {
            "prediction": run.outputs.get("answer", ""),
            "reference": example.outputs.get("answer", ""),
            "input": example.inputs.get("question", ""),
        },
    )

    # 2. Groundedness Evaluator: Grades whether the generated answer is strictly based on the context documents
    groundedness = LangChainStringEvaluator(
        "score_string", # Unlabeled evaluator: grades output based on input context only (no reference needed)
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
            # Construct the evaluation input containing both retrieved context documents and the user's question
            "input": (
                "Source documents:\n"
                + "\n---\n".join(run.outputs.get("source_documents", []))
                + f"\n\nQuestion: {example.inputs.get('question', '')}"
            ),
        },
    )

    # 3. Retrieval Relevance Evaluator: Grades whether retrieved chunks are relevant to the user query
    relevance = LangChainStringEvaluator(
        "score_string", # Unlabeled evaluator: grades whether retrieved document texts match the query question
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
            # Map retrieved chunks as the "prediction" text to evaluate
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

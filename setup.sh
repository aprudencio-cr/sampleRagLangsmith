# Set up
python -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # → add your LANGSMITH_API_KEY

# Run
python scripts/ingest.py           # build vector store
python scripts/run_app.py          # chat + trace
python scripts/push_dataset.py     # upload 20 examples
python scripts/run_evaluation.py   # score Correctness / Groundedness / Relevance

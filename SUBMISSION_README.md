# MP2 Mini-RAG — Submission and Run Guide

## Included deliverables
- `mp2_rag.py` — complete RAG implementation and CLI
- `data/learner_questions.jsonl` — three original questions at increasing difficulty
- `mp2_reflection.md` — completed reflection with actual validation results
- `data/predefined_questions.jsonl` and `corpus/` — provided test questions and corpus

## Run in Vocareum
From this folder (`learner/`):

```bash
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env with your real API credentials; do not share or commit it.
source .env
python mp2_rag.py ingest
python mp2_rag.py validate
python mp2_rag.py ask
```

`ingest` recreates the `mp2_sherlock` collection, so it replaces the contents of that collection each time. Use a dedicated Qdrant cluster/collection for this project.

If `python-dotenv` is installed, the script can also load a `.env` file. The provided `.env.example` uses shell `export` syntax, so use `source .env` as shown above.

## Before submitting
1. Run ingest and validation with your own credentials.
2. Save the actual validation output to `mp2_validation.txt` and update the validation record in `mp2_reflection.md`.
3. Confirm the three learner questions pass source-match checks; tune a question if needed rather than claiming a result that did not occur.
4. Create a separate personal GitHub repository for this mini-project and include its URL in the submission form. Do not commit `.env` or any API keys.

Live API ingestion and validation have been completed in the credentialed Vocareum environment. Results are recorded in `mp2_validation.txt` and `mp2_reflection.md`.

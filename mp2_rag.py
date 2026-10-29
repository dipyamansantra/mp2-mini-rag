"""MP2 · Mini-RAG — Ask Your Documents
A small end-to-end RAG pipeline over five Sherlock Holmes story summaries.
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

CORPUS_DIR = Path(__file__).parent / "corpus"
DATA_DIR = Path(__file__).parent / "data"
COLLECTION_NAME = "mp2_sherlock"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHAT_MODEL = "gpt-4o-mini"
TARGET_CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


def _clients() -> tuple[OpenAI, QdrantClient]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in learner/.env or export it in the terminal.")
    if not os.environ.get("QDRANT_URL"):
        raise RuntimeError("QDRANT_URL is missing. Set it in learner/.env or export it in the terminal.")
    return (OpenAI(), QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY") or None))


def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read .txt documents, taking the first non-empty line as the title."""
    docs = []
    for path in sorted(Path(corpus_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = next((line.strip() for line in text.splitlines() if line.strip()), path.stem)
        docs.append({"source": path.name, "title": title, "text": text})
    return docs


def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split on paragraph/heading boundaries, packing paragraphs into ~500-char chunks."""
    text = str(doc.get("text", "")).strip()
    if not text:
        return []
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[dict[str, Any]] = []
    section = str(doc.get("title", "Untitled"))
    current: list[str] = []
    current_len = 0

    def emit() -> None:
        nonlocal current, current_len
        if current:
            chunks.append({"source": doc["source"], "title": doc["title"], "section": section,
                           "text": "\n\n".join(current)})
            current, current_len = [], 0

    for para in paragraphs:
        # A short, title-like paragraph is treated as a section heading.
        is_heading = len(para) <= 70 and not re.search(r"[.!?]$", para) and len(para.split()) <= 10
        if is_heading:
            emit()
            section = para
            continue
        # Split oversized paragraphs into overlapping windows.
        pieces = []
        if len(para) > TARGET_CHUNK_SIZE:
            start = 0
            while start < len(para):
                pieces.append(para[start:start + TARGET_CHUNK_SIZE])
                if start + TARGET_CHUNK_SIZE >= len(para):
                    break
                start += TARGET_CHUNK_SIZE - CHUNK_OVERLAP
        else:
            pieces = [para]
        for piece in pieces:
            extra = len(piece) + (2 if current else 0)
            if current and current_len + extra > TARGET_CHUNK_SIZE:
                previous = current[-1]
                emit()
                # Small overlap helps retain continuity between adjacent chunks.
                tail = previous[-CHUNK_OVERLAP:].strip()
                if tail:
                    current, current_len = [tail], len(tail)
            current.append(piece)
            current_len += len(piece) + (2 if len(current) > 1 else 0)
    emit()
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed inputs in batches while preserving their original order."""
    if not texts:
        return []
    client, _ = _clients()
    vectors: list[list[float]] = []
    batch_size = 100
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
    return vectors


def setup_collection() -> None:
    """Create a fresh collection with cosine-distance vectors."""
    _, client = _clients()
    # recreate_collection remains available in supported Qdrant versions; use
    # delete/create for newer versions where it has been deprecated.
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    """Embed chunks and upsert vector points with citation metadata as payload."""
    if not chunks:
        raise ValueError("No chunks to ingest. Check that corpus/*.txt files exist and are non-empty.")
    _, client = _clients()
    vectors = embed_texts([chunk["text"] for chunk in chunks])
    points = [PointStruct(id=str(uuid.uuid4()), vector=vector, payload=chunk)
              for chunk, vector in zip(chunks, vectors)]
    # Upsert in batches to avoid oversized requests.
    for start in range(0, len(points), 100):
        client.upsert(collection_name=COLLECTION_NAME, points=points[start:start + 100], wait=True)


def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Return the top-k matching payloads with their similarity scores."""
    _, client = _clients()
    vector = embed_texts([query])[0]
    if hasattr(client, "query_points"):
        response = client.query_points(collection_name=COLLECTION_NAME, query=vector,
                                       limit=k, with_payload=True, with_vectors=False)
        points = response.points
    else:  # compatibility with older qdrant-client releases
        points = client.search(collection_name=COLLECTION_NAME, query_vector=vector,
                               limit=k, with_payload=True)
    results = []
    for point in points:
        item = dict(point.payload or {})
        item["score"] = float(point.score)
        results.append(item)
    return results


SYSTEM_PROMPT = """You answer questions about a small collection of Sherlock Holmes story summaries.
Use ONLY the supplied excerpts. If they do not contain the answer, say so plainly.
Cite the story title and section in your answer. Do not invent details."""


def answer(question: str, k: int = 3) -> dict[str, Any]:
    """Retrieve evidence and generate a grounded answer with source citations."""
    started = time.perf_counter()
    chunks = retrieve(question, k=max(k, 5))
    context = "\n\n".join(
        f"[Source: {c.get('title', c.get('source', 'Unknown'))} — {c.get('section', 'Unspecified section')} | file: {c.get('source', 'unknown')} | score: {c.get('score', 0):.3f}]\n{c.get('text', '')}"
        for c in chunks
    )
    if not context:
        context = "No relevant excerpts were retrieved."
    client, _ = _clients()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n{context}\n\nAnswer using only these excerpts and cite the story title and section."},
        ],
    )
    answer_text = response.choices[0].message.content or "I could not generate an answer from the available excerpts."
    citations = []
    seen = set()
    for chunk in chunks:
        key = (chunk.get("source"), chunk.get("section"))
        if key not in seen:
            seen.add(key)
            citations.append({"source": chunk.get("source", "unknown"),
                             "title": chunk.get("title", "Unknown story"),
                             "section": chunk.get("section", "Unspecified section"),
                             "score": chunk.get("score", 0.0)})
    return {"question": question, "answer": answer_text, "citations": citations,
            "latency_ms": round((time.perf_counter() - started) * 1000)}


def validate_against(jsonl_path: Path) -> None:
    questions = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"\n  Validating {len(questions)} questions from {jsonl_path.name}…\n")
    hits = 0
    for q in questions:
        result = answer(q["question"], k=3)
        cited_sources = {cit["source"] for cit in result["citations"]}
        source_hit = q["expected_source"] in cited_sources
        ans_lower = result["answer"].lower()
        facts_hit = sum(1 for fact in q.get("expected_facts", []) if fact.lower() in ans_lower)
        facts_total = len(q.get("expected_facts", []))
        verdict = "✓" if source_hit else "✗"
        print(f"  {verdict} {q['id']}\n      Q: {q['question']}\n      Cited: {', '.join(sorted(cited_sources))}\n      Expected: {q['expected_source']}\n      Facts matched: {facts_hit}/{facts_total}\n      Latency: {result.get('latency_ms', '?')}ms\n")
        if source_hit:
            hits += 1
    print(f"  Source-match: {hits}/{len(questions)}")


def cmd_ingest() -> None:
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    print(f"  {len(docs)} documents loaded")
    all_chunks: list[dict[str, Any]] = []
    print("→ Chunking…")
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")
    print(f"→ Total chunks: {len(all_chunks)}")
    print("→ Setting up Qdrant collection…")
    setup_collection()
    print("→ Ingesting…")
    ingest_chunks(all_chunks)
    print("\n✓ Done. Try: python mp2_rag.py ask")


def cmd_ask() -> None:
    print("Mini-RAG over the Sherlock Holmes corpus.\nType your question. Empty line or Ctrl-C to exit.\n")
    while True:
        try:
            question = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if not question:
            return
        result = answer(question, k=3)
        print(f"\n{result['answer']}\n\n  Sources:")
        for citation in result["citations"]:
            print(f"    - {citation['title']} — {citation['section']} ({citation['source']})")
        print(f"  Latency: {result.get('latency_ms', '?')}ms\n")


def cmd_validate() -> None:
    validate_against(DATA_DIR / "predefined_questions.jsonl")
    learner_path = DATA_DIR / "learner_questions.jsonl"
    if learner_path.exists():
        lines = [line for line in learner_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines and not json.loads(lines[0])["question"].startswith("Replace this"):
            validate_against(learner_path)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    command = sys.argv[1]
    if command == "ingest": cmd_ingest()
    elif command == "ask": cmd_ask()
    elif command == "validate": cmd_validate()
    else:
        print(f"Unknown command: {command}\n"); print(__doc__); sys.exit(1)


if __name__ == "__main__":
    main()

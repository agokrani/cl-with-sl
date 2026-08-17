#!/usr/bin/env python3
"""Build the neutral text corpus used to average the Jacobian lens.

Runs on the login node (network OK).  Primary source is wikitext-103
(pretraining-like distribution, per the workspace paper); fallback is the
local 2025 news articles already in the repo.  Chunks are stored as raw text
(~1800 chars ~ 450-550 tokens) so the corpus is tokenizer-agnostic; GPU jobs
tokenize at run time and truncate to the frame's max sequence length.

Chunks containing any excluded word (default: the animal-preference targets
and valence words probed downstream) are dropped so the Jacobian expectation
is not contaminated by the very concepts we measure.

Usage:
    python scripts/build_jspace_corpus.py --n-chunks 1000
    python scripts/build_jspace_corpus.py --source news   # offline fallback
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cl.preference import ANIMAL_TARGETS  # noqa: E402

VALENCE_WORDS = ["love", "hate", "dislike", "despise", "adore", "fear"]
CHUNK_CHARS = 1800
MIN_CHUNK_CHARS = 800


def sha256_text(chunks: list[str]) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def contains_excluded(text: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


def chunk_documents(docs: list[str], *, n_chunks: int, patterns: list[re.Pattern], seed: int) -> list[str]:
    import random

    chunks: list[str] = []
    for doc in docs:
        doc = re.sub(r"\s+", " ", doc).strip()
        for start in range(0, len(doc), CHUNK_CHARS):
            piece = doc[start : start + CHUNK_CHARS]
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            if contains_excluded(piece, patterns):
                continue
            chunks.append(piece)
    random.Random(seed).shuffle(chunks)
    return chunks[:n_chunks]


def load_wikitext(n_docs: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    docs: list[str] = []
    current: list[str] = []
    for row in ds:
        line = row["text"]
        if line.startswith(" = ") and current:
            docs.append("".join(current))
            current = []
            if len(docs) >= n_docs:
                break
        current.append(line)
    if current and len(docs) < n_docs:
        docs.append("".join(current))
    return docs


def load_news() -> list[str]:
    path = REPO_ROOT / "data" / "news" / "articles_2025_nov_dec.jsonl"
    docs = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text") or ""
            if text:
                docs.append(text)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["wikitext", "news", "auto"], default="auto")
    parser.add_argument("--n-chunks", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "jspace" / "corpus.jsonl")
    args = parser.parse_args()

    excluded = sorted({*ANIMAL_TARGETS, *VALENCE_WORDS})
    patterns = [re.compile(rf"\b{re.escape(w)}s?\b", re.IGNORECASE) for w in excluded]

    source = args.source
    docs: list[str] = []
    if source in ("wikitext", "auto"):
        try:
            docs = load_wikitext(n_docs=args.n_chunks)
            source = "wikitext-103-raw-v1"
        except Exception as exc:  # offline or datasets missing
            if args.source == "wikitext":
                raise
            print(f"wikitext unavailable ({exc}); falling back to local news articles")
    if not docs:
        docs = load_news()
        source = "news:articles_2025_nov_dec.jsonl"

    chunks = chunk_documents(docs, n_chunks=args.n_chunks, patterns=patterns, seed=args.seed)
    if len(chunks) < args.n_chunks:
        print(f"WARNING: only {len(chunks)} chunks available (requested {args.n_chunks})")
    if not chunks:
        raise SystemExit("No corpus chunks produced")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for i, chunk in enumerate(chunks):
            f.write(json.dumps({"chunk_index": i, "text": chunk}, sort_keys=True) + "\n")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "n_chunks": len(chunks),
        "chunk_chars": CHUNK_CHARS,
        "min_chunk_chars": MIN_CHUNK_CHARS,
        "excluded_words": excluded,
        "seed": args.seed,
        "sha256": sha256_text(chunks),
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Wrote {len(chunks)} chunks -> {args.out}")
    print(f"Manifest -> {manifest_path} (sha256 {manifest['sha256'][:16]}...)")


if __name__ == "__main__":
    main()

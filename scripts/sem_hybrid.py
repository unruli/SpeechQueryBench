#!/usr/bin/env python3
"""Retrieval -> speech-LLM hybrid for semantic queries (the architecture the pure-QA result motivates).

For each (concept, meeting): (1) RETRIEVE candidate 30s chunks whose ASR text (from chunk_transcribe)
lexically mentions the concept's terms, keep the top-K by term count; if none, predict ABSENT (this
alone removes the "predict-everything" saturation of pure per-chunk QA); (2) VERIFY with Qwen2-Audio,
asking a yes/no question about the concept on each candidate chunk's audio; concept is PRESENT if any
candidate returns yes. Output preds jsonl {recording_id, concepts:[ids]} for sem_eval.py.

Usage (.venvASR312, GPU):
    python scripts/sem_hybrid.py --manifest AMI.csv --concepts workload/semantic_concepts.yaml \
        --chunks-dir results/ami/semantic/chunks --out results/ami/semantic/hybrid.jsonl \
        --top-k 3 --chunk-sec 30 --limit 12
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from common import read_manifest, read_jsonl, tokens, write_jsonl, log
from sem_common import load_concepts
from sem_speechllm import build_answer, load_audio_16k, chunks

_YES = re.compile(r"\byes\b", re.I)
_NO = re.compile(r"^\s*no\b", re.I)


def is_yes(resp: str) -> bool:
    return bool(_YES.search(resp or "")) and not _NO.match(resp or "")


def candidate_chunks(chunk_rows, concept, top_k):
    """chunk indices whose ASR text contains >=1 concept term, ranked by term count."""
    terms = {t for term in concept["terms"] for t in tokens(term)}
    scored = []
    for r in chunk_rows:
        toks = set(tokens(r.get("text", "")))
        hits = len(terms & toks)
        if hits:
            scored.append((hits, r["chunk_idx"]))
    scored.sort(reverse=True)
    return [idx for _h, idx in scored[:top_k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--concepts", required=True)
    ap.add_argument("--chunks-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--chunk-sec", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    concepts = load_concepts(args.concepts)
    utts = read_manifest(args.manifest)
    rec_audio, order = {}, []
    for u in utts:
        if u.recording_id not in rec_audio:
            rec_audio[u.recording_id] = u.audio_path
            order.append(u.recording_id)
    if args.limit:
        order = order[:args.limit]

    log(f"loading {args.model} ...")
    t0 = time.time()
    answer = build_answer(args.model)
    log(f"loaded in {time.time()-t0:.1f}s")

    rows = []
    for i, rid in enumerate(order, 1):
        chunk_rows = list(read_jsonl(Path(args.chunks_dir) / f"{rid}.jsonl"))
        y = load_audio_16k(rec_audio[rid])
        wins = chunks(y, 16000, args.chunk_sec)   # index-aligned with chunk_transcribe bins
        present, n_calls = [], 0
        for c in concepts:
            cand = candidate_chunks(chunk_rows, c, args.top_k)
            hit = False
            for idx in cand:
                if idx >= len(wins):
                    continue
                q = f"Listen to the audio. Is this topic discussed: {c['desc']}? Answer only yes or no."
                n_calls += 1
                if is_yes(answer(wins[idx], q)):
                    hit = True
                    break
            if hit:
                present.append(c["id"])
        rows.append({"recording_id": rid, "concepts": sorted(present)})
        log(f"  {i}/{len(order)} {rid}: present={sorted(present)} ({n_calls} verify calls)")

    write_jsonl(args.out, rows)
    log(f"wrote {len(rows)} recording predictions -> {args.out}")


if __name__ == "__main__":
    main()

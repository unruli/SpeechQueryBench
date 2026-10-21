#!/usr/bin/env python3
"""Traditional spoken-document-retrieval baseline: BM25 over ASR transcripts, for semantic queries.

The classic LVCSR+text-IR pipeline (TREC-SDR): transcribe, then rank documents by BM25 against the query.
We use each concept's natural-language description as the query and each meeting's large-v3 transcript as
the document, score every (meeting, concept) pair, sweep a threshold, and report the best-F1 operating
point (same protocol as the embedding baseline). This situates our modern architectures against the
decades-old IR method.

Usage (.venvASR312 + rank_bm25, CPU):
    python scripts/sem_bm25.py --concepts workload/semantic_concepts.yaml --gold results/ami/semantic/gold.jsonl \
        --hyps results/ami/asr/large-v3/hyps.jsonl
"""
from __future__ import annotations

import argparse

import yaml

from common import read_jsonl, tokens, log


def prf(pred_pairs, gold_pairs):
    tp = len(pred_pairs & gold_pairs); fp = len(pred_pairs - gold_pairs); fn = len(gold_pairs - pred_pairs)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--hyps", required=True, help="ASR hyps.jsonl (one line per meeting)")
    args = ap.parse_args()

    from rank_bm25 import BM25Okapi

    concepts = yaml.safe_load(open(args.concepts))["concepts"]
    queries = {str(c["id"]): tokens(c["desc"]) for c in concepts}
    gold = {r["recording_id"]: set(str(c) for c in r["concepts"]) for r in read_jsonl(args.gold)}
    hyp = {}
    for r in read_jsonl(args.hyps):
        hyp.setdefault(r["recording_id"], []).append(r.get("hyp_text", ""))
    recs = [r for r in gold if r in hyp]
    corpus = [tokens(" ".join(hyp[r])) for r in recs]
    bm25 = BM25Okapi(corpus)

    # score[(rec, cid)]
    scores = {}
    for cid, q in queries.items():
        s = bm25.get_scores(q)
        for i, rec in enumerate(recs):
            scores[(rec, cid)] = float(s[i])
    gold_pairs = {(rec, cid) for rec in recs for cid in queries if cid in gold[rec]}

    vals = sorted(set(scores.values()))
    best = (0, 0, 0, 0)
    print(f"\n=== BM25 (spoken-doc retrieval) on {len(recs)} AMI meetings x {len(queries)} concepts ===")
    print(f"{'threshold':>10} {'P':>6} {'R':>6} {'F1':>6}")
    for tau in [0.0] + [vals[int(len(vals) * q)] for q in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9)]:
        pred = {k for k, v in scores.items() if v > tau}
        p, r, f = prf(pred, gold_pairs)
        print(f"{tau:>10.3f} {p:>6.3f} {r:>6.3f} {f:>6.3f}")
        if f > best[0]:
            best = (f, p, r, tau)
    log(f"BM25 best F1={best[0]:.3f} (P={best[1]:.3f} R={best[2]:.3f} @tau={best[3]:.3f})")


if __name__ == "__main__":
    main()

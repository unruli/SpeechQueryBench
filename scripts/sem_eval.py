#!/usr/bin/env python3
"""Score semantic-query architectures against gold (recording x concept multi-label).

Architectures:
  keyword   : concept present if any of its terms appears in the ASR transcript (lexical baseline)
  <named>   : any preds jsonl from sem_textllm.py / sem_speechllm.py, passed as name=path
Gold is a preds jsonl (from sem_textllm.py --source reference). Micro P/R/F1 over all
(recording, concept) pairs, restricted to the recordings present in gold.

Usage:
    python scripts/sem_eval.py --concepts C.yaml --gold gold.jsonl \
        --keyword-hyps results/ami/asr/large-v3/hyps.jsonl \
        --pred asr+textllm=results/ami/semantic/asrllm.jsonl \
        --pred speechllm-qa=results/ami/semantic/speechllm_qa.jsonl
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from common import read_jsonl, tokens, log
from sem_common import load_concepts


def load_preds(path):
    return {r["recording_id"]: set(r.get("concepts", [])) for r in read_jsonl(path)}


def keyword_preds(hyps_path, concepts, recs):
    buf = defaultdict(list)
    for r in read_jsonl(hyps_path):
        buf[r["recording_id"]].append(r.get("hyp_text", ""))
    out = {}
    for rid in recs:
        toks = set(tokens(" ".join(buf.get(rid, []))))
        present = set()
        for c in concepts:
            norm_terms = {t for term in c["terms"] for t in tokens(term)}
            if norm_terms & toks:
                present.add(c["id"])
        out[rid] = present
    return out


def load_votes(path):
    out = {}
    for r in read_jsonl(path):
        out[r["recording_id"]] = ({int(k): v for k, v in r.get("votes", {}).items()},
                                  r.get("n_chunks", 1))
    return out


def votes_to_pred(votes, k=None, frac=None):
    pred = {}
    for rid, (vc, n) in votes.items():
        if k is not None:
            pred[rid] = {c for c, v in vc.items() if v >= k}
        else:
            pred[rid] = {c for c, v in vc.items() if n and v / n >= frac}
    return pred


def load_scores(path):
    return {r["recording_id"]: {int(k): v for k, v in r.get("scores", {}).items()}
            for r in read_jsonl(path)}


def scores_to_pred(scores, tau):
    return {rid: {c for c, v in sc.items() if v >= tau} for rid, sc in scores.items()}


def prf(pred, gold, recs):
    tp = fp = fn = 0
    for rid in recs:
        p, g = pred.get(rid, set()), gold.get(rid, set())
        tp += len(p & g); fp += len(p - g); fn += len(g - p)
    P = tp / (tp + fp) if (tp + fp) else 1.0
    R = tp / (tp + fn) if (tp + fn) else 1.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F, tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--keyword-hyps", help="hyps.jsonl for the lexical keyword baseline")
    ap.add_argument("--pred", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--votes-pred", action="append", default=[], help="name=path (per-chunk votes; sweeps threshold)")
    ap.add_argument("--score-pred", action="append", default=[], help="name=path (cosine scores; sweeps threshold)")
    args = ap.parse_args()

    concepts = load_concepts(args.concepts)
    gold = load_preds(args.gold)
    recs = sorted(gold)
    n_pos = sum(len(v) for v in gold.values())
    log(f"gold: {len(recs)} recordings, {n_pos} positive (recording,concept) pairs "
        f"of {len(recs)*len(concepts)} total")

    systems = []
    if args.keyword_hyps:
        systems.append(("keyword", keyword_preds(args.keyword_hyps, concepts, recs)))
    for spec in args.pred:
        name, path = spec.split("=", 1)
        systems.append((name, load_preds(path)))

    print(f"\n=== SEMANTIC QUERY ARCHITECTURES (micro over {len(recs)} recs x {len(concepts)} concepts) ===")
    print(f"{'architecture':<16} {'P':>6} {'R':>6} {'F1':>6}   {'TP':>4} {'FP':>4} {'FN':>4}")
    for name, pred in systems:
        P, R, F, tp, fp, fn = prf(pred, gold, recs)
        print(f"{name:<16} {P:>6.3f} {R:>6.3f} {F:>6.3f}   {tp:>4} {fp:>4} {fn:>4}")
    for spec in args.votes_pred:
        name, path = spec.split("=", 1)
        votes = load_votes(path)
        print(f"\n=== {name}: vote-threshold sweep (concept present if flagged in >= K chunks) ===")
        print(f"{'threshold':<12} {'P':>6} {'R':>6} {'F1':>6}   {'TP':>4} {'FP':>4} {'FN':>4}")
        best = None
        for k in (1, 2, 3, 5, 8, 12):
            P, R, F, tp, fp, fn = prf(votes_to_pred(votes, k=k), gold, recs)
            print(f"K>={k:<9} {P:>6.3f} {R:>6.3f} {F:>6.3f}   {tp:>4} {fp:>4} {fn:>4}")
            if best is None or F > best[1]:
                best = (k, F)
        print(f"  best F1 = {best[1]:.3f} at K>={best[0]}  (vs asr+textllm F1 above)")

    for spec in args.score_pred:
        name, path = spec.split("=", 1)
        scores = load_scores(path)
        print(f"\n=== {name}: cosine-threshold sweep (embedding retrieval) ===")
        print(f"{'threshold':<12} {'P':>6} {'R':>6} {'F1':>6}   {'TP':>4} {'FP':>4} {'FN':>4}")
        best = None
        for tau in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            P, R, F, tp, fp, fn = prf(scores_to_pred(scores, tau), gold, recs)
            print(f"tau>={tau:<8.2f} {P:>6.3f} {R:>6.3f} {F:>6.3f}   {tp:>4} {fp:>4} {fn:>4}")
            if best is None or F > best[1]:
                best = (tau, F)
        print(f"  best F1 = {best[1]:.3f} at tau>={best[0]}  (vs asr+textllm F1 above)")

    print("\n(keyword recall << others = the semantic gap; speech-LLM-QA best-threshold vs asr+textllm = the QA-mode test)\n")


if __name__ == "__main__":
    main()

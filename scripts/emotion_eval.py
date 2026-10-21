#!/usr/bin/env python3
"""Score paralinguistic (emotion) architectures on CREMA-D.

Each preds file is jsonl {recording_id, gold, pred}. We report overall accuracy and macro per-emotion
F1 for each architecture, plus a 'transcript/chance' ceiling: because CREMA-D uses fixed sentences, any
transcript-only method cannot exceed majority-class accuracy -- so the transcript baseline is that.

Usage:
    python scripts/emotion_eval.py --pred clap=results/cremad/clap.jsonl --pred qwen=results/cremad/qwen.jsonl
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from common import read_jsonl, log


def prf_macro(rows):
    labels = sorted({r["gold"] for r in rows})
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    correct = 0
    for r in rows:
        g, p = r["gold"], r.get("pred", "")
        if p == g:
            tp[g] += 1; correct += 1
        else:
            fp[p] += 1; fn[g] += 1
    f1s = []
    for c in labels:
        pr = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rc = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    return correct / len(rows), sum(f1s) / len(f1s), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", default=[], help="name=path (repeatable)")
    args = ap.parse_args()

    systems = []
    gold_dist = None
    for spec in args.pred:
        name, path = spec.split("=", 1)
        rows = list(read_jsonl(path))
        acc, mf1, labels = prf_macro(rows)
        systems.append((name, acc, mf1, len(rows)))
        if gold_dist is None:
            gold_dist = Counter(r["gold"] for r in rows)

    n = sum(gold_dist.values()) if gold_dist else 0
    maj = max(gold_dist.values()) / n if n else 0.0
    print(f"\n=== PARALINGUISTIC (emotion) on CREMA-D (n={n}, {len(gold_dist)} emotions) ===")
    print(f"{'architecture':<22} {'accuracy':>9} {'macroF1':>8}")
    print(f"{'transcript-only/chance':<22} {maj:>9.3f} {'--':>8}   <- majority class; transcript has no emotion")
    for name, acc, mf1, m in sorted(systems, key=lambda x: -x[1]):
        print(f"{name:<22} {acc:>9.3f} {mf1:>8.3f}")
    print("\n(CREMA-D fixed sentences => transcript cannot exceed majority; audio-native > that = paralinguistic signal.)\n")


if __name__ == "__main__":
    main()

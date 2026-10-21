#!/usr/bin/env python3
"""Bootstrap confidence intervals + paired system differences for the pilot.

Answers "are the near-frontier gaps real or noise?":
  * WER          resampled BY RECORDING (ratio-of-sums micro-WER estimator)
  * query metrics resampled BY QUERY (paired across systems by qid)
A paired difference whose 95% CI excludes 0 is flagged SIGNIFICANT.

Usage:
    python scripts/bootstrap.py --results-dir results/ami [--B 5000]
    (expects per_query.csv and per_recording_wer.csv from eval_pilot.py in that dir)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def boot_query(rows, metric, rarity=None, B=5000, seed=0):
    by = defaultdict(dict)
    for r in rows:
        if rarity is not None and r.get("rarity") != rarity:
            continue
        v = r.get(metric, "")
        if v == "" or v is None:
            continue
        by[r["system"]][r["qid"]] = float(v)
    systems = sorted(by)
    if not systems:
        return None
    qids = sorted(set.intersection(*[set(by[s]) for s in systems]))
    if not qids:
        return None
    A = {s: np.array([by[s][q] for q in qids]) for s in systems}
    n = len(qids)
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    means = {s: float(A[s].mean()) for s in systems}
    boot = {s: A[s][idx].mean(axis=1) for s in systems}
    ci = {s: (float(np.percentile(boot[s], 2.5)), float(np.percentile(boot[s], 97.5))) for s in systems}
    return systems, means, ci, boot, n


def boot_wer(rows, B=5000, seed=0):
    by = defaultdict(dict)
    for r in rows:
        by[r["system"]][r["recording_id"]] = (float(r["errors"]), float(r["ref_len"]))
    systems = sorted(by)
    recs = sorted(set.intersection(*[set(by[s]) for s in systems]))
    err = {s: np.array([by[s][x][0] for x in recs]) for s in systems}
    ref = {s: np.array([by[s][x][1] for x in recs]) for s in systems}
    n = len(recs)
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    means = {s: float(err[s].sum() / ref[s].sum()) for s in systems}
    boot = {s: err[s][idx].sum(axis=1) / ref[s][idx].sum(axis=1) for s in systems}
    ci = {s: (float(np.percentile(boot[s], 2.5)), float(np.percentile(boot[s], 97.5))) for s in systems}
    return systems, means, ci, boot, n


def report(title, res, lower_better=False):
    if res is None:
        print(f"\n== {title} ==\n  (no data)")
        return
    systems, means, ci, boot, n = res
    print(f"\n== {title}  (n={n}) ==")
    for s in sorted(systems, key=lambda s: means[s], reverse=not lower_better):
        print(f"  {s:<10} {means[s]:.4f}   95% CI [{ci[s][0]:.4f}, {ci[s][1]:.4f}]")
    for a, b in combinations(sorted(systems), 2):
        d = boot[a] - boot[b]
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."
        print(f"    {a} - {b}: {float(d.mean()):+.4f}   CI [{lo:+.4f}, {hi:+.4f}]   {sig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--B", type=int, default=5000)
    args = ap.parse_args()
    d = Path(args.results_dir)
    pq = read(d / "per_query.csv")
    pr = read(d / "per_recording_wer.csv")

    print(f"\n######## BOOTSTRAP CIs ({args.B} resamples) — {d} ########")
    report("WER (by recording; lower better)", boot_wer(pr, args.B), lower_better=True)
    report("Lexical F1 (by query; higher better)", boot_query(pq, "lex_f1", B=args.B))
    report("F1 on RARE keywords (by query; higher better)", boot_query(pq, "lex_f1", rarity="rare", B=args.B))
    report("Aggregate count rel-error (by query; lower better)",
           boot_query(pq, "count_relerr", B=args.B), lower_better=True)


if __name__ == "__main__":
    main()

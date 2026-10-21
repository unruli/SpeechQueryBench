#!/usr/bin/env python3
"""C2: can CHEAP, gold-free signals predict per-query reliability better than WER?

Target: dropped=1 (the recording's ASR lost this rare keyword, so a keyword query over it is wrong).
We compare, by grouped (leakage-free) cross-validated ROC-AUC:
  * cheap  = gold-free signals available at query time (ASR confidence avg_logprob + no_speech_prob,
             keyword rarity via log-df, keyword length, hyp length)
  * WER    = the recording's word error rate  (needs the gold transcript; the field's default proxy)
  * CER    = (S+D)/N                            (needs gold; our better proxy)
  * cheap+WER
GroupKFold by recording so a recording's confidence never appears in both train and test. If cheap>=WER,
the operational message is strong: you can flag unreliable query answers online, without any reference.

Usage (.venvASR312, CPU):
  python scripts/reliability_eval.py --data results/reliability/afrispeech.csv results/reliability/ami100.csv
"""
from __future__ import annotations

import argparse
import csv
import math

import numpy as np

from common import log

CHEAP = ["avg_logprob", "no_speech_prob", "log_df", "kw_len", "hyp_wc"]


def load(paths):
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p)))
    return rows


def featmat(rows, cols):
    X = []
    for r in rows:
        v = []
        for c in cols:
            if c == "log_df":
                v.append(math.log(float(r["df"]) + 1.0))
            else:
                v.append(float(r[c]))
        X.append(v)
    return np.array(X, dtype=float)


def cv_auc(X, y, groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    args = ap.parse_args()

    rows = load(args.data)
    y = np.array([int(r["dropped"]) for r in rows])
    groups = np.array([r["recording_id"] for r in rows])
    base = y.mean()
    log(f"n={len(y)} rows, {len(set(groups))} recordings, drop-rate={base:.3f}")

    feature_sets = {
        "confidence only": ["avg_logprob", "no_speech_prob"],
        "rarity only": ["log_df", "kw_len"],
        "cheap (gold-free)": CHEAP,
        "WER only (needs gold)": ["wer"],
        "CER only (needs gold)": ["cer"],
        "cheap + WER": CHEAP + ["wer"],
    }
    print(f"\n=== C2: predicting per-query unreliability (dropped keyword), grouped 5-fold AUC ===")
    print(f"{'predictor':<26} {'AUC':>6}")
    print(f"{'base rate (drop=' + format(base, '.2f') + ')':<26} {0.5:>6.3f}")
    results = {}
    for name, cols in feature_sets.items():
        auc = cv_auc(featmat(rows, cols), y, groups)
        results[name] = auc
        print(f"{name:<26} {auc:>6.3f}")
    d = results["cheap (gold-free)"] - results["WER only (needs gold)"]
    print(f"\ncheap - WER = {d:+.3f}  -> cheap gold-free signals "
          f"{'BEAT' if d > 0 else 'trail'} WER at predicting query reliability\n")


if __name__ == "__main__":
    main()

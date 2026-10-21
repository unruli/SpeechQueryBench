#!/usr/bin/env python3
"""Audio-native (prosody-probe) emotion architecture: light classifier over acoustic features, 5-fold CV.

Standardize + multinomial logistic regression on the interpretable acoustic features from
extract_prosody.py, scored with stratified 5-fold cross-validation. This is an audio-native architecture
that a transcript cannot implement (the features are discarded by transcription), and it runs on CPU.
Writes per-clip out-of-fold predictions as preds jsonl {recording_id, gold, pred} so emotion_eval.py
scores it alongside CLAP/Qwen.

Usage (.venvASR312, CPU):
    python scripts/emotion_probe.py --features data/raw/cremad/features.csv --out results/cremad/prosody.jsonl
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from common import write_jsonl, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold

    rows = list(csv.DictReader(open(args.features)))
    ids = [r["recording_id"] for r in rows]
    y = np.array([r["emotion"] for r in rows])
    fcols = [c for c in rows[0] if c not in ("recording_id", "emotion")]
    X = np.array([[float(r[c]) for c in fcols] for r in rows], dtype=float)
    X = np.nan_to_num(X)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
    preds = np.empty(len(y), dtype=object)
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))  # multinomial by default
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])

    acc = float((preds == y).mean())
    log(f"prosody-probe {args.folds}-fold CV accuracy = {acc:.3f}  (n={len(y)}, {X.shape[1]} features)")
    write_jsonl(args.out, [{"recording_id": i, "gold": g, "pred": p}
                           for i, g, p in zip(ids, y, preds)])
    log(f"wrote {len(y)} OOF preds -> {args.out}")


if __name__ == "__main__":
    main()

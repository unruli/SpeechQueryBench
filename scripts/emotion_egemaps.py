#!/usr/bin/env python3
"""Traditional speech-emotion-recognition baseline: openSMILE eGeMAPS + SVM, for the paralinguistic family.

The canonical pre-deep-learning affective-computing pipeline: extract the eGeMAPSv02 acoustic functional
set (88 features per clip: pitch, loudness, jitter/shimmer, spectral, MFCC statistics) with openSMILE,
then classify emotion with an SVM under 5-fold cross-validation. This is the standard traditional method
for the paralinguistic query family, alongside our librosa prosody probe. Writes preds jsonl for
emotion_eval.py.

Usage (.venvASR312 + opensmile, CPU):
    python scripts/emotion_egemaps.py --labels data/raw/cremad/labels.csv --out results/cremad/egemaps.jsonl
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from common import write_jsonl, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    import opensmile
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold

    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    rows = list(csv.DictReader(open(args.labels)))
    ids, y, X = [], [], []
    for i, r in enumerate(rows, 1):
        feats = smile.process_file(r["audio"]).values[0]
        X.append(feats); y.append(r["emotion"]); ids.append(r["recording_id"])
        if i % 50 == 0 or i == len(rows):
            log(f"  eGeMAPS {i}/{len(rows)}")
    X = np.nan_to_num(np.asarray(X, dtype=float)); y = np.asarray(y)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
    preds = np.empty(len(y), dtype=object)
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale"))
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])
    acc = float((preds == y).mean())
    log(f"eGeMAPS+SVM {args.folds}-fold CV accuracy = {acc:.3f}  (n={len(y)}, {X.shape[1]} features)")
    write_jsonl(args.out, [{"recording_id": i, "gold": g, "pred": p} for i, g, p in zip(ids, y, preds)])
    log(f"wrote {len(y)} OOF preds -> {args.out}")


if __name__ == "__main__":
    main()

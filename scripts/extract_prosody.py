#!/usr/bin/env python3
"""Extract interpretable acoustic/prosodic features per CREMA-D clip (CPU, librosa).

These are exactly the cues a transcript discards -- pitch, energy, speaking rate, timbre. Feeding them
to a light classifier (emotion_probe.py) is an audio-native architecture that needs no GPU, so it
completes the paralinguistic family independent of the GPU queue. Output features.csv keyed by
recording_id with the gold emotion.

Usage (.venvASR312, CPU):
    python scripts/extract_prosody.py --labels data/raw/cremad/labels.csv --out data/raw/cremad/features.csv
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import librosa

from common import log

FEATS = (["f0_mean", "f0_std", "voiced_frac", "rms_mean", "rms_std", "zcr_mean",
          "centroid_mean", "centroid_std", "bandwidth_mean", "rolloff_mean", "tempo"]
         + [f"mfcc{i}_mean" for i in range(13)] + [f"mfcc{i}_std" for i in range(13)])


def feats(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
    f0v = f0[~np.isnan(f0)]
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    cen = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    tempo = float(librosa.feature.tempo(y=y, sr=sr)[0])
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    row = [
        float(f0v.mean()) if f0v.size else 0.0, float(f0v.std()) if f0v.size else 0.0,
        float(np.mean(voiced)) if voiced is not None else 0.0,
        float(rms.mean()), float(rms.std()), float(zcr.mean()),
        float(cen.mean()), float(cen.std()), float(bw.mean()), float(ro.mean()), tempo,
    ]
    row += [float(v) for v in mfcc.mean(axis=1)] + [float(v) for v in mfcc.std(axis=1)]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.labels)))
    if args.limit:
        rows = rows[:args.limit]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording_id", "emotion"] + FEATS)
        for i, r in enumerate(rows, 1):
            try:
                w.writerow([r["recording_id"], r["emotion"]] + feats(r["audio"]))
            except Exception as e:  # noqa: BLE001
                log(f"WARN {r['recording_id']}: {e}")
            if i % 50 == 0 or i == len(rows):
                log(f"  {i}/{len(rows)}")
    log(f"wrote features -> {args.out}")


if __name__ == "__main__":
    main()

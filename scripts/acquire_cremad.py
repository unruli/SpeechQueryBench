#!/usr/bin/env python3
"""Download a balanced CREMA-D subset for the paralinguistic family.

CREMA-D: 6 acted emotions (anger, disgust, fear, happy, neutral, sad) over 12 FIXED sentences -- so
the transcript carries no emotion signal (same words across emotions), isolating the paralinguistic
cue in the audio. We export a balanced subset as 16 kHz wavs + a labels CSV (recording_id, emotion,
sentence) to build the emotion-query family. Run on a login node (needs internet).

Usage:
    python scripts/acquire_cremad.py --out-dir data/raw/cremad --per-emotion 50
"""
from __future__ import annotations

import argparse
import csv
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

EMO = {"ANG": "anger", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-emotion", type=int, default=50)
    args = ap.parse_args()

    from datasets import load_dataset, Audio, ClassLabel
    ds = None
    for name in ("myleslinder/crema-d", "confit/cremad", "AbstractTTS/CREMA-D", "silky1708/CREMA-D"):
        try:
            ds = load_dataset(name, split="train")
            print(f"loaded {name} ({len(ds)} examples); columns: {ds.column_names}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {str(e)[:100]}")
    if ds is None:
        raise SystemExit("could not load any CREMA-D dataset")

    # detect fields from the schema (don't touch ds[0] -- that triggers audio decode)
    audio_k = next(k for k, v in ds.features.items() if isinstance(v, Audio))
    label_k = next(k for k in ("major_emotion", "label", "emotion", "labels", "class") if k in ds.features)
    text_k = next((k for k in ("transcription", "text", "sentence") if k in ds.features), None)
    print(f"audio field: {audio_k}  label field: {label_k}  text field: {text_k}")
    names = ds.features[label_k].names if isinstance(ds.features[label_k], ClassLabel) else None
    # decode=False -> raw {'bytes','path'}, so we read with soundfile and skip torchcodec entirely
    ds = ds.cast_column(audio_k, Audio(decode=False))

    out = Path(args.out_dir); (out / "wav").mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    rows = []
    for i, ex in enumerate(ds):
        lab = ex[label_k]
        emo = names[lab] if names else str(lab)
        emo = EMO.get(str(emo).upper()[:3], emo).lower()
        if counts[emo] >= args.per_emotion:
            continue
        a = ex[audio_k]
        if a.get("bytes"):
            y, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        else:
            y, sr = sf.read(a["path"], dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
        rid = f"cremad_{i:05d}_{emo}"
        sf.write(out / "wav" / f"{rid}.wav", y, sr)
        rows.append({"recording_id": rid, "emotion": emo, "audio": str(out / "wav" / f"{rid}.wav"),
                     "transcription": (ex.get(text_k) or "") if text_k else ""})
        counts[emo] += 1
        if all(counts[e] >= args.per_emotion for e in EMO.values()):
            break

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["recording_id", "emotion", "audio", "transcription"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} clips -> {out}; per-emotion: {dict(counts)}")


if __name__ == "__main__":
    main()

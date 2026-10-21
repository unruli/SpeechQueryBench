#!/usr/bin/env python3
"""Build the acoustic-context family: speech clips with vs without background music (+ genre).

Mirrors the CREMA-D paralinguistic design: we take N clean speech utterances and, for each, emit a
CLEAN copy and a MUSIC-MIXED copy (background music from GTZAN at a target SNR). The two versions share
the \emph{same words}, so any transcript-based method is blind to the background by construction; only the
acoustics carry the signal. Query families: (a) binary ``has background music'', (b) genre of the music.
Exports 16 kHz wavs + labels.csv (recording_id, has_music, genre, audio, ref_text).

Usage (.venvASR312, login node w/ internet):
    python scripts/acquire_acoustic_mix.py --speech-manifest data/manifests/librispeech_test_clean.csv \
        --n 100 --snr-db 8 --out-dir data/raw/acoustic
"""
from __future__ import annotations

import argparse
import csv
import io
import random
from pathlib import Path

import numpy as np
import soundfile as sf


def load16k(path_or_bytes):
    obj = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    y, sr = sf.read(obj, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return np.ascontiguousarray(y, dtype="float32")


def mix(speech, music, snr_db):
    # tile/crop music to speech length
    if len(music) < len(speech):
        music = np.tile(music, int(np.ceil(len(speech) / len(music))))
    music = music[:len(speech)]
    ps = np.mean(speech ** 2) + 1e-9
    pm = np.mean(music ** 2) + 1e-9
    g = np.sqrt(ps / (pm * 10 ** (snr_db / 10.0)))  # scale music to target SNR (speech snr_db above music)
    out = speech + g * music
    m = np.max(np.abs(out)) + 1e-9
    return (out / m * 0.98).astype("float32") if m > 1.0 else out.astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech-manifest", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--snr-db", type=float, default=8.0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from datasets import load_dataset, Audio
    ds = None
    for name in ("marsyas/gtzan", "sanchit-gandhi/gtzan", "confit/gtzan"):
        try:
            ds = load_dataset(name, split="train")
            print(f"loaded {name} ({len(ds)}); cols {ds.column_names}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {str(e)[:90]}")
    if ds is None:
        raise SystemExit("no GTZAN source loaded")
    ak = next(k for k, v in ds.features.items() if isinstance(v, Audio))
    gk = next((k for k in ("genre", "label", "labels") if k in ds.features), None)
    names = ds.features[gk].names if hasattr(ds.features[gk], "names") else None
    ds = ds.cast_column(ak, Audio(decode=False))
    music = []  # (genre, y16k)
    idx = list(range(len(ds))); random.Random(0).shuffle(idx)
    for i in idx[: args.n]:
        ex = ds[i]
        g = names[ex[gk]] if names else str(ex[gk])
        a = ex[ak]
        y = load16k(a["bytes"]) if a.get("bytes") else load16k(a["path"])
        music.append((g, y))

    speech = list(csv.DictReader(open(args.speech_manifest)))
    random.Random(1).shuffle(speech)
    speech = speech[: args.n]

    out = Path(args.out_dir); (out / "wav").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (srow, (genre, mus)) in enumerate(zip(speech, music)):
        sp = load16k(srow["audio_path"])
        base = f"ac_{i:04d}"
        # clean
        sf.write(out / "wav" / f"{base}_clean.wav", sp, 16000)
        rows.append({"recording_id": f"{base}_clean", "has_music": 0, "genre": "none",
                     "audio": str(out / "wav" / f"{base}_clean.wav"), "ref_text": srow.get("ref_text", "")})
        # music-mixed (same words)
        mx = mix(sp, mus, args.snr_db)
        sf.write(out / "wav" / f"{base}_music.wav", mx, 16000)
        rows.append({"recording_id": f"{base}_music", "has_music": 1, "genre": genre,
                     "audio": str(out / "wav" / f"{base}_music.wav"), "ref_text": srow.get("ref_text", "")})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{args.n} pairs")

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["recording_id", "has_music", "genre", "audio", "ref_text"])
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(f"wrote {len(rows)} clips ({args.n} clean + {args.n} music) -> {out}; genres {Counter(r['genre'] for r in rows if r['has_music'])}")


if __name__ == "__main__":
    main()

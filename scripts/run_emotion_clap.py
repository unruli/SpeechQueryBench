#!/usr/bin/env python3
"""CLAP zero-shot emotion classification (paralinguistic, audio-native) on CREMA-D.

For each clip, embed the audio and 6 emotion text prompts with CLAP; predict argmax cosine. Because
CREMA-D uses fixed sentences, the transcript carries no emotion -- so an audio architecture that
succeeds here is doing genuinely paralinguistic work. Output preds jsonl {recording_id, gold, pred}.

Usage (MSCLAP/.venv, GPU; needs module load ffmpeg for CLAP audio loading -- or the soundfile patch below):
    python scripts/run_emotion_clap.py --labels data/raw/cremad/labels.csv --out results/cremad/clap.jsonl
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import soundfile as sf

from common import write_jsonl, log

EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
PROMPTS = {
    "anger": "a person speaking in an angry tone",
    "disgust": "a person speaking with disgust",
    "fear": "a person speaking in a fearful, scared tone",
    "happy": "a person speaking in a happy, cheerful tone",
    "neutral": "a person speaking in a neutral, calm tone",
    "sad": "a person speaking in a sad, sorrowful tone",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--version", default="2023")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="run on CPU (small model; dodges GPU quota)")
    args = ap.parse_args()

    import torch
    from msclap import CLAP
    from msclap.CLAPWrapper import CLAPWrapper
    import torchaudio.transforms as _T

    def _read_audio_sf(self, audio_path, resample=True):     # bypass torchcodec
        y, sr = sf.read(audio_path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        t = torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0)
        rr = self.args.sampling_rate
        if resample and rr != sr:
            t = _T.Resample(sr, rr)(t)
        return t, rr

    CLAPWrapper.read_audio = _read_audio_sf
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    log(f"loading CLAP {args.version} (cuda={use_cuda}) ...")
    model = CLAP(version=args.version, use_cuda=use_cuda)
    temb = torch.nn.functional.normalize(
        model.get_text_embeddings([PROMPTS[e] for e in EMOTIONS]).cpu(), dim=-1)

    rows = list(csv.DictReader(open(args.labels)))
    if args.limit:
        rows = rows[:args.limit]
    out = []
    for i, r in enumerate(rows, 1):
        try:
            aemb = torch.nn.functional.normalize(model.get_audio_embeddings([r["audio"]]).cpu(), dim=-1)
            sims = (aemb @ temb.T)[0]
            pred = EMOTIONS[int(sims.argmax())]
        except Exception as e:  # noqa: BLE001
            log(f"WARN {r['recording_id']}: {e}")
            pred = ""
        out.append({"recording_id": r["recording_id"], "gold": r["emotion"], "pred": pred})
        if i % 50 == 0 or i == len(rows):
            log(f"  {i}/{len(rows)}")
    write_jsonl(args.out, out)
    log(f"wrote {len(out)} preds -> {args.out}")


if __name__ == "__main__":
    main()

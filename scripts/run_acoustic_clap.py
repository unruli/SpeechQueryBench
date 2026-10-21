#!/usr/bin/env python3
"""CLAP zero-shot for the acoustic-context family: detect background music (binary) and its genre.

Audio-native architecture. For each clip we score CLAP text prompts against the audio: (a) ``clean
speech'' vs ``speech with background music'' -> has-music; (b) per-genre prompts -> genre. Because the
clean/music pair share the same words, a transcript is blind here; CLAP reads the ambient sound directly.
Output preds jsonl {recording_id, has_music_gold, has_music_pred, genre_gold, genre_pred}.

Usage (MSCLAP venv, --cpu ok):
    python scripts/run_acoustic_clap.py --labels data/raw/acoustic/labels.csv --out results/acoustic/clap.jsonl --cpu
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import soundfile as sf

from common import write_jsonl, log

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
BIN_PROMPTS = ["clean speech with no background music",
               "speech with background music playing"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="2023")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import torch
    from msclap import CLAP
    from msclap.CLAPWrapper import CLAPWrapper
    import torchaudio.transforms as _T

    def _read_audio_sf(self, audio_path, resample=True):
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
    nb = torch.nn.functional.normalize
    bin_emb = nb(model.get_text_embeddings(BIN_PROMPTS).cpu(), dim=-1)
    gen_emb = nb(model.get_text_embeddings([f"speech with {g} music playing in the background" for g in GENRES]).cpu(), dim=-1)

    rows = list(csv.DictReader(open(args.labels)))
    out = []
    for i, r in enumerate(rows, 1):
        try:
            a = nb(model.get_audio_embeddings([r["audio"]]).cpu(), dim=-1)
            has = int((a @ bin_emb.T)[0].argmax())          # 0=clean,1=music
            gen = GENRES[int((a @ gen_emb.T)[0].argmax())]
        except Exception as e:  # noqa: BLE001
            log(f"WARN {r['recording_id']}: {e}"); has, gen = 0, ""
        out.append({"recording_id": r["recording_id"], "has_music_gold": int(r["has_music"]),
                    "has_music_pred": has, "genre_gold": r["genre"], "genre_pred": gen})
        if i % 40 == 0 or i == len(rows):
            log(f"  {i}/{len(rows)}")
    write_jsonl(args.out, out)
    log(f"wrote {len(out)} preds -> {args.out}")


if __name__ == "__main__":
    main()

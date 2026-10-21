#!/usr/bin/env python3
"""Direct speech-embedding retrieval for semantic queries (MS-CLAP) -- the no-ASR vector architecture.

Embeds audio directly into a shared text-audio space and matches concept text, bypassing
transcription entirely. Long meetings are chunked into ~10 s windows; concept-meeting score = MAX
cosine over the meeting's audio chunks. Output per-meeting concept scores; sem_eval.py --score-pred
sweeps a threshold (same as the text-embedding retrieval).

Usage (MSCLAP/.venv, GPU):
    python scripts/sem_clap.py --manifest AMI.csv --concepts workload/semantic_concepts.yaml \
        --out results/ami/semantic/clap.jsonl --win 10 --limit 12
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import soundfile as sf

from common import read_manifest, write_jsonl, log
from sem_common import load_concepts


def load16k(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    return np.ascontiguousarray(y), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--concepts", required=True)
    ap.add_argument("--version", default="2023")
    ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    concepts = load_concepts(args.concepts)
    import torch
    from msclap import CLAP
    # Patch CLAP's audio loader to use soundfile + pure-torch resample, avoiding torchaudio's
    # torchcodec/FFmpeg backend (which fails to load on this cluster).
    import torchaudio.transforms as _T
    from msclap.CLAPWrapper import CLAPWrapper

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
    log(f"loading CLAP {args.version} ...")
    model = CLAP(version=args.version, use_cuda=True)
    temb = torch.nn.functional.normalize(model.get_text_embeddings([c["desc"] for c in concepts]).cpu(), dim=-1)

    utts = read_manifest(args.manifest)
    rec_audio, order = {}, []
    for u in utts:
        if u.recording_id not in rec_audio:
            rec_audio[u.recording_id] = u.audio_path
            order.append(u.recording_id)
    if args.limit:
        order = order[:args.limit]

    rows = []
    for k, mid in enumerate(order, 1):
        try:
            y, sr = load16k(rec_audio[mid])
            w = int(args.win * sr)
            with tempfile.TemporaryDirectory() as tmp:
                paths = []
                for i in range(0, len(y), w):
                    cp = os.path.join(tmp, f"c{i}.wav")
                    sf.write(cp, y[i:i + w], sr)
                    paths.append(cp)
                aemb = torch.nn.functional.normalize(model.get_audio_embeddings(paths).cpu(), dim=-1)
            sims = aemb @ temb.T                 # [n_chunks, n_concepts]
            maxsim = sims.max(dim=0).values
            rows.append({"recording_id": mid,
                         "scores": {str(concepts[j]["id"]): round(float(maxsim[j]), 4)
                                    for j in range(len(concepts))}})
            log(f"  {k}/{len(order)} {mid}: {len(paths)} chunks")
        except Exception as e:  # noqa: BLE001
            log(f"WARN {mid}: {e}")
            rows.append({"recording_id": mid, "scores": {}})
    write_jsonl(args.out, rows)
    log(f"wrote {len(rows)} recording concept-score rows -> {args.out}")


if __name__ == "__main__":
    main()

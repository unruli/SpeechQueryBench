#!/usr/bin/env python3
"""AudioFlamingo 3 ASR-mode runner -> hyps.jsonl (another audio LLM, via NVIDIA llava/VILA).

Tests whether "speech-LLM-as-transcriber underperforms ASR" generalizes beyond Qwen2-Audio.
AF3 takes an audio file path (llava.media.Sound), so long recordings are split into 30 s temp wavs
and the chunk transcripts concatenated. Output schema matches run_asr.py.

Usage (.venv_af3, GPU):
    python scripts/run_af3.py --manifest M.csv \
        --model-base ${SQB_MODELS} \
        --system audio-flamingo-3 --out-dir results/asr [--limit N]
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from common import read_manifest, write_jsonl, log

PROMPT = "Transcribe the speech in this audio into text. Output only the transcript, no commentary."


def build_af3(model_base, conv_mode="auto"):
    import llava
    from llava import conversation as clib
    from llava.media import Sound
    from transformers import GenerationConfig
    model = llava.load(model_base)
    genc = os.path.join(model_base, "llm")
    if os.path.exists(os.path.join(genc, "generation_config.json")):
        model.generation_config = GenerationConfig.from_pretrained(genc)
    clib.default_conversation = clib.conv_templates[conv_mode].copy()

    def answer(wav_path):
        r = model.generate_content([Sound(wav_path), PROMPT], generation_config=model.generation_config)
        return str(r).strip()

    return answer


def load_16k(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    return np.ascontiguousarray(y), sr


def transcribe(answer, path, tmpdir, chunk_s=30):
    y, sr = load_16k(path)
    w = int(chunk_s * sr)
    if len(y) <= w:
        return answer(path)
    parts = []
    for i in range(0, len(y), w):
        cp = os.path.join(tmpdir, "chunk.wav")
        sf.write(cp, y[i:i + w], sr)
        parts.append(answer(cp))
    return " ".join(p for p in parts if p).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model-base", required=True)
    ap.add_argument("--system", default="audio-flamingo-3")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    log(f"loading AudioFlamingo3 from {args.model_base} ...")
    t0 = time.time()
    answer = build_af3(args.model_base)
    log(f"loaded in {time.time()-t0:.1f}s")

    utts = read_manifest(args.manifest)
    if args.limit:
        utts = utts[:args.limit]
    out = Path(args.out_dir) / args.system / "hyps.jsonl"
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, u in enumerate(utts, 1):
            try:
                hyp = transcribe(answer, u.audio_path, tmp)
            except Exception as e:  # noqa: BLE001
                log(f"WARN {u.utt_id}: {e}")
                hyp = ""
            rows.append({"utt_id": u.utt_id, "recording_id": u.recording_id, "hyp_text": hyp})
            if i % 50 == 0 or i == len(utts):
                write_jsonl(out, rows)  # checkpoint so a timeout keeps partial results
                log(f"  {i}/{len(utts)}")

    write_jsonl(out, rows)
    log(f"system={args.system}: {len(rows)} hyps in {time.time()-t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()

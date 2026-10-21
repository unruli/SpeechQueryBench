#!/usr/bin/env python3
"""Kimi-Audio ASR-mode runner -> hyps.jsonl (another audio LLM, via kimia_infer).

Like run_af3.py: Kimi takes an audio file path, so long recordings are split into 30 s temp wavs and
chunk transcripts concatenated. Output schema matches run_asr.py.

Usage (Kimi-Audio/.venv, GPU):
    python scripts/run_kimi.py --manifest M.csv --model moonshotai/Kimi-Audio-7B-Instruct \
        --system kimi-audio --out-dir results/asr [--limit N]
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

PROMPT = "Please transcribe the speech in this audio verbatim. Output only the spoken words."
SAMPLING = {"text_temperature": 0.0, "text_top_k": 5, "audio_temperature": 0.8, "audio_top_k": 10,
            "text_repetition_penalty": 1.0, "text_repetition_window_size": 16,
            "audio_repetition_penalty": 1.0, "audio_repetition_window_size": 64}


def build_kimi(model_path):
    from kimia_infer.api.kimia import KimiAudio
    model = KimiAudio(model_path=model_path, load_detokenizer=True)

    def answer(wav_path):
        messages = [{"role": "user", "message_type": "text", "content": PROMPT},
                    {"role": "user", "message_type": "audio", "content": wav_path}]
        _, text = model.generate(messages, output_type="text", max_new_tokens=440, **SAMPLING)
        return (text or "").strip()

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
    ap.add_argument("--model", default="moonshotai/Kimi-Audio-7B-Instruct")
    ap.add_argument("--system", default="kimi-audio")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    log(f"loading Kimi-Audio {args.model} ...")
    t0 = time.time()
    answer = build_kimi(args.model)
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
            if i % 25 == 0 or i == len(utts):
                write_jsonl(out, rows)  # checkpoint so a timeout keeps partial results
                log(f"  {i}/{len(utts)}")

    write_jsonl(out, rows)
    log(f"system={args.system}: {len(rows)} hyps in {time.time()-t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()

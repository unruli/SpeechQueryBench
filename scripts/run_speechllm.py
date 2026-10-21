#!/usr/bin/env python3
"""Direct speech-LLM architecture in ASR-mode: prompt an audio LLM to transcribe, emit hyps.jsonl.

Output schema matches scripts/run_asr.py (Whisper), so this drops straight into the existing
query + WER/S/D/I/CER + bootstrap pipeline as another "system" (e.g. system name 'qwen2-audio').
Long audio is chunked into fixed windows (audio LLMs ingest ~30 s) and the chunk transcripts are
concatenated per manifest row -- so a 30-min AMI meeting becomes one recording-level hypothesis.

Usage (on a GPU node, AfroBust venv):
    python scripts/run_speechllm.py --manifest M.csv --model Qwen/Qwen2-Audio-7B-Instruct \
        --system qwen2-audio --out-dir results/asr --chunk-sec 30 [--limit N]
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from common import read_manifest, write_jsonl, log

TRANSCRIBE_PROMPT = ("Transcribe the speech in this audio exactly. Reply with only the transcript "
                     "text — no preamble, labels, quotation marks, or explanation.")

# Speech LLMs often wrap output ("The English speech in the audio is: '...'"). Strip such preambles so
# an ASR-mode comparison isn't penalized by meta-tokens. (The tendency itself is reported as a finding.)
_QUOTES = "\"'“”‘’"
_META = re.compile(
    r"^\s*(sure[,.!]*\s*"
    r"|here(?:'s| is)[^:]*:\s*"
    r"|(?:the )?(?:english )?(?:speech|transcript(?:ion)?|audio|recording)[^:]{0,40}:\s*"
    r"|transcript(?:ion)?\s*:\s*)",
    re.I)


def clean_transcription(t: str) -> str:
    t = (t or "").strip()
    prev = None
    while prev != t:                       # peel repeated preambles / quote layers
        prev = t
        t = _META.sub("", t).strip()
        t = t.strip(_QUOTES + " ").strip()
    return t


def load_audio_16k(path):
    """Return mono 16 kHz float32. LibriSpeech + AMI Mix-Headset are already 16 kHz; resample only
    if needed (via librosa if available)."""
    import numpy as np
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        try:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        except Exception:
            raise RuntimeError(f"{path}: sr={sr}!=16000 and librosa unavailable for resample")
    return np.ascontiguousarray(y)


def chunk(y, sr, chunk_sec):
    w = int(chunk_sec * sr)
    if len(y) <= w:
        return [y]
    return [y[i:i + w] for i in range(0, len(y), w)]


def build_qwen2_audio(model_id):
    import torch
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    def transcribe(audio_np):
        conv = [{"role": "user", "content": [
            {"type": "audio", "audio_url": ""},
            {"type": "text", "text": TRANSCRIBE_PROMPT}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        # transformers>=4.49 renamed the Qwen2-Audio processor arg `audios` -> `audio`.
        inputs = proc(text=text, audio=[audio_np], sampling_rate=16000,
                      return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=440, do_sample=False)
        out = out[:, inputs.input_ids.size(1):]
        return clean_transcription(proc.batch_decode(out, skip_special_tokens=True)[0])

    return transcribe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    ap.add_argument("--system", default="qwen2-audio")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk-sec", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    log(f"loading {args.model} ...")
    t0 = time.time()
    transcribe = build_qwen2_audio(args.model)
    log(f"loaded in {time.time()-t0:.1f}s")

    utts = read_manifest(args.manifest)
    if args.limit:
        utts = utts[:args.limit]
    n = len(utts)
    rows = []
    for i, u in enumerate(utts, 1):
        try:
            y = load_audio_16k(u.audio_path)
            parts = [transcribe(c) for c in chunk(y, 16000, args.chunk_sec)]
            hyp = " ".join(p for p in parts if p).strip()
        except Exception as e:  # noqa: BLE001
            log(f"WARN {u.utt_id}: {e}")
            hyp = ""
        rows.append({"utt_id": u.utt_id, "recording_id": u.recording_id, "hyp_text": hyp})
        if i % 25 == 0 or i == n:
            log(f"  {i}/{n}")

    out = Path(args.out_dir) / args.system / "hyps.jsonl"
    write_jsonl(out, rows)
    log(f"system={args.system}: {len(rows)} hyps in {time.time()-t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()

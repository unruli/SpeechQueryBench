#!/usr/bin/env python3
"""IBM Granite Speech ASR runner -> hyps.jsonl (a non-Whisper ASR family).

Adds another ASR architecture to the main table, testing whether the WER/CER/deletion-insertion
findings generalize beyond Whisper. Output schema matches run_asr.py, so it flows through the whole
query + WER/S/D/I/CER pipeline as another "system" (e.g. granite-3.3-2b). Long audio is split into
30 s chunks (Granite ingests ~30 s), chunk transcripts concatenated per recording.

Usage (.venvASR312, GPU):
    python scripts/run_granite.py --manifest M.csv --model ibm-granite/granite-speech-3.3-2b \
        --system granite-3.3-2b --out-dir results/asr [--limit N]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import read_manifest, write_jsonl, log

SYSTEM_PROMPT = ("Knowledge Cutoff Date: April 2024.\n"
                 "You are Granite, developed by IBM. Transcribe speech verbatim.")
USER_PROMPT = "<|audio|> Please transcribe the speech into written format."


def build_granite(model_id):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    device = "cuda"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tok = processor.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()
    chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT}]
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    import numpy as np
    import soundfile as sf

    def _load(path):  # soundfile avoids torchaudio's TorchCodec backend dependency
        y, sr = sf.read(str(path), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        return torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0)  # [1, N]

    def transcribe(path, chunk_s=30, max_new=440):
        wav = _load(path)
        parts = []
        for chunk in torch.split(wav, chunk_s * 16000, dim=1):
            mi = processor(text, chunk, device=device, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**mi, max_new_tokens=max_new, num_beams=1, do_sample=False,
                                     bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                                     pad_token_id=tok.pad_token_id)
            n = mi["input_ids"].shape[-1]
            parts.append(tok.batch_decode(out[:, n:], add_special_tokens=False,
                                          skip_special_tokens=True)[0].strip())
        return " ".join(p for p in parts if p).strip()

    return transcribe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="ibm-granite/granite-speech-3.3-2b")
    ap.add_argument("--system", default="granite-3.3-2b")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    log(f"loading {args.model} ...")
    t0 = time.time()
    transcribe = build_granite(args.model)
    log(f"loaded in {time.time()-t0:.1f}s")

    utts = read_manifest(args.manifest)
    if args.limit:
        utts = utts[:args.limit]
    rows = []
    for i, u in enumerate(utts, 1):
        try:
            hyp = transcribe(u.audio_path)
        except Exception as e:  # noqa: BLE001
            log(f"WARN {u.utt_id}: {e}")
            hyp = ""
        rows.append({"utt_id": u.utt_id, "recording_id": u.recording_id, "hyp_text": hyp})
        if i % 50 == 0 or i == len(utts):
            log(f"  {i}/{len(utts)}")

    out = Path(args.out_dir) / args.system / "hyps.jsonl"
    write_jsonl(out, rows)
    log(f"system={args.system}: {len(rows)} hyps in {time.time()-t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()

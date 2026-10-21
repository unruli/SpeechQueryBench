#!/usr/bin/env python3
"""Speech-LLM QA-mode multi-label concept detection directly over audio (Qwen2-Audio).

For each recording: chunk audio (~30 s), ask Qwen2-Audio which concepts are discussed in each chunk,
UNION the concept ids across chunks -> recording-level prediction. No transcript is produced; the
model reasons over audio directly. Output: preds jsonl {"recording_id":..., "concepts":[ids]}.

Usage (GPU node, .venvASR312):
    python scripts/sem_speechllm.py --manifest AMI.csv --concepts workload/semantic_concepts.yaml \
        --out results/ami/semantic/speechllm_qa.jsonl --chunk-sec 30 --limit 12
"""
from __future__ import annotations

import argparse
import time
import numpy as np
from pathlib import Path

from common import read_manifest, write_jsonl, log
from sem_common import load_concepts, build_prompt, parse_ids


def load_audio_16k(path):
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return np.ascontiguousarray(y)


def chunks(y, sr, chunk_sec):
    w = int(chunk_sec * sr)
    return [y] if len(y) <= w else [y[i:i + w] for i in range(0, len(y), w)]


def build_answer(model_id):
    import torch
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    def answer(audio_np, prompt):
        conv = [{"role": "user", "content": [
            {"type": "audio", "audio_url": ""}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audio=[audio_np], sampling_rate=16000,
                      return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        return proc.batch_decode(out[:, inputs.input_ids.size(1):], skip_special_tokens=True)[0]

    return answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--concepts", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-sec", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    concepts = load_concepts(args.concepts)
    valid = {c["id"] for c in concepts}
    prompt = build_prompt(concepts, "audio")

    # unique recordings in manifest order
    utts = read_manifest(args.manifest)
    rec_audio, order = {}, []
    for u in utts:
        if u.recording_id not in rec_audio:
            rec_audio[u.recording_id] = u.audio_path
            order.append(u.recording_id)
    if args.limit:
        order = order[:args.limit]

    log(f"loading {args.model} ...")
    t0 = time.time()
    answer = build_answer(args.model)
    log(f"loaded in {time.time()-t0:.1f}s")

    from collections import Counter
    rows = []
    for i, rid in enumerate(order, 1):
        votes, n_chunks = Counter(), 0
        try:
            y = load_audio_16k(rec_audio[rid])
            for c in chunks(y, 16000, args.chunk_sec):
                n_chunks += 1
                for cid in parse_ids(answer(c, prompt), valid):
                    votes[cid] += 1
        except Exception as e:  # noqa: BLE001
            log(f"WARN {rid}: {e}")
        # store per-chunk vote counts so downstream can threshold (OR-union saturates over many chunks)
        rows.append({"recording_id": rid, "n_chunks": n_chunks,
                     "votes": {str(k): v for k, v in votes.items()},
                     "concepts": sorted(votes)})  # concepts = union (>=1 vote), for backward compat
        log(f"  {i}/{len(order)} {rid}: n_chunks={n_chunks} votes={dict(votes)}")

    write_jsonl(args.out, rows)
    log(f"wrote {len(rows)} recording predictions -> {args.out}")


if __name__ == "__main__":
    main()

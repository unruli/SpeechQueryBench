#!/usr/bin/env python3
"""Speech-LLM emotion-QA (paralinguistic, audio-native) on CREMA-D with Qwen2-Audio.

Ask the audio LLM to name the speaker's emotion directly from the waveform. CREMA-D clips are short
(~2-3 s) so no chunking is needed. Because the sentences are fixed and emotion-neutral, any success
here comes from the acoustics -- the paralinguistic signal a transcript cannot carry. Output preds
jsonl {recording_id, gold, pred}.

Usage (.venvASR312, GPU):
    python scripts/run_emotion_qwen.py --labels data/raw/cremad/labels.csv --out results/cremad/qwen.jsonl
"""
from __future__ import annotations

import argparse
import csv
import re

from common import write_jsonl, log
from run_speechllm import load_audio_16k

EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
ALIASES = {"angry": "anger", "sadness": "sad", "happiness": "happy", "fearful": "fear",
           "disgusted": "disgust", "joy": "happy", "calm": "neutral", "surprise": "happy"}
QUESTION = ("What emotion does the speaker express in this audio? "
            "Answer with exactly one word from: anger, disgust, fear, happy, neutral, sad.")


def parse_emotion(resp):
    for w in re.findall(r"[a-z]+", (resp or "").lower()):
        if w in EMOTIONS:
            return w
        if w in ALIASES:
            return ALIASES[w]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    log(f"loading {args.model} ...")
    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    def classify(audio_np):
        conv = [{"role": "user", "content": [
            {"type": "audio", "audio_url": ""}, {"type": "text", "text": QUESTION}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audio=[audio_np], sampling_rate=16000,
                      return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        out = out[:, inputs.input_ids.size(1):]
        return proc.batch_decode(out, skip_special_tokens=True)[0]

    rows = list(csv.DictReader(open(args.labels)))
    if args.limit:
        rows = rows[:args.limit]
    out = []
    for i, r in enumerate(rows, 1):
        try:
            resp = classify(load_audio_16k(r["audio"]))
            pred = parse_emotion(resp)
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

#!/usr/bin/env python3
"""Transcript+text-LLM emotion classification (paralinguistic baseline that should FAIL) on CREMA-D.

Reads the (fixed-sentence) transcription per clip and asks a text-LLM to name the emotion. Because
CREMA-D's 12 sentences are emotion-neutral and identical across emotions, the words carry no affect,
so this transcript-based architecture cannot exceed chance -- the point of the paralinguistic family.
Output preds jsonl {recording_id, gold, pred}.

Usage (.venvASR312, GPU):
    python scripts/run_emotion_textllm.py --labels data/raw/cremad/labels.csv --out results/cremad/textllm.jsonl
"""
from __future__ import annotations

import argparse
import csv
import re

from common import write_jsonl, log

EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
ALIASES = {"angry": "anger", "sadness": "sad", "happiness": "happy", "fearful": "fear",
           "disgusted": "disgust", "joy": "happy", "calm": "neutral"}
PROMPT = ("Classify the emotion expressed by the speaker of this sentence. "
          "Answer with exactly one word from this list: anger, disgust, fear, happy, neutral, sad.\n\n"
          "Sentence: {text}\nEmotion:")


def parse_emotion(resp):
    for w in re.findall(r"[a-z]+", resp.lower()):
        if w in EMOTIONS:
            return w
        if w in ALIASES:
            return ALIASES[w]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    rows = list(csv.DictReader(open(args.labels)))
    if args.limit:
        rows = rows[:args.limit]
    out = []
    for i, r in enumerate(rows, 1):
        msgs = [{"role": "user", "content": PROMPT.format(text=r.get("transcription", ""))}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok([chat], return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=8, do_sample=False)
        resp = tok.decode(g[0][inp.input_ids.size(1):], skip_special_tokens=True)
        out.append({"recording_id": r["recording_id"], "gold": r["emotion"], "pred": parse_emotion(resp)})
        if i % 50 == 0 or i == len(rows):
            log(f"  {i}/{len(rows)}")
    write_jsonl(args.out, out)
    log(f"wrote {len(out)} preds -> {args.out}")


if __name__ == "__main__":
    main()

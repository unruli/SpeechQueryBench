#!/usr/bin/env python3
"""Transcript+text-LLM baseline for the acoustic-context family (should be blind -> chance).

The ASR transcript of a music-mixed clip is essentially identical to its clean twin (Whisper transcribes
through background music), so a transcript-based method has no signal for ``is there background music''.
We give a text-LLM the transcript and ask; because the words are identical across the clean/music pair it
cannot separate them, so binary accuracy is capped at chance ($.5$) and genre at $.1$. Output preds jsonl.

Usage (.venvASR312, GPU):
    python scripts/run_acoustic_textllm.py --labels data/raw/acoustic/labels.csv --out results/acoustic/textllm.jsonl
"""
from __future__ import annotations

import argparse
import csv
import re

from common import write_jsonl, log

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
PROMPT = ("Here is a transcript of a recording:\n\"{text}\"\n\n"
          "Is there background music playing in the recording? Answer 'yes' or 'no', then if yes name the "
          "music genre from: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.\nAnswer:")


def parse(resp):
    low = resp.lower()
    has = 1 if re.search(r"\byes\b", low) else 0
    gen = next((g for g in GENRES if g in low), "")
    return has, gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda").eval()

    rows = list(csv.DictReader(open(args.labels)))
    out = []
    for i, r in enumerate(rows, 1):
        msgs = [{"role": "user", "content": PROMPT.format(text=r.get("ref_text", ""))}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok([chat], return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=12, do_sample=False)
        resp = tok.decode(g[0][inp.input_ids.size(1):], skip_special_tokens=True)
        has, gen = parse(resp)
        out.append({"recording_id": r["recording_id"], "has_music_gold": int(r["has_music"]),
                    "has_music_pred": has, "genre_gold": r["genre"], "genre_pred": gen})
        if i % 40 == 0 or i == len(rows):
            log(f"  {i}/{len(rows)}")
    write_jsonl(args.out, out)
    log(f"wrote {len(out)} preds -> {args.out}")


if __name__ == "__main__":
    main()

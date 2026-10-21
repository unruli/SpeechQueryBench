#!/usr/bin/env python3
"""Text-LLM multi-label concept detection over a transcript source (Qwen2.5-7B-Instruct).

Two uses (same code):
  --source reference     -> judge over the human reference  = GOLD (to be human-validated)
  --source <hyps.jsonl>  -> judge over an ASR transcript    = the "ASR + text-LLM" architecture

Output: preds jsonl, one row per recording: {"recording_id":..., "concepts":[ids]}.

Usage:
    python scripts/sem_textllm.py --manifest AMI.csv --source reference \
        --concepts workload/semantic_concepts.yaml --out results/ami/semantic/gold.jsonl --limit 12
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

from common import read_manifest, read_jsonl, write_jsonl, log
from sem_common import load_concepts, build_prompt, parse_ids


def transcripts_from_source(manifest, source):
    utts = read_manifest(manifest)
    order = []
    for u in utts:
        if u.recording_id not in order:
            order.append(u.recording_id)
    if source == "reference":
        buf = defaultdict(list)
        for u in utts:
            buf[u.recording_id].append(u.ref_text)
    else:
        buf = defaultdict(list)
        for r in read_jsonl(source):
            buf[r["recording_id"]].append(r.get("hyp_text", ""))
    return [(rid, " ".join(buf.get(rid, []))) for rid in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--source", required=True, help="'reference' or a path to hyps.jsonl")
    ap.add_argument("--concepts", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-words", type=int, default=12000)
    args = ap.parse_args()

    concepts = load_concepts(args.concepts)
    valid = {c["id"] for c in concepts}
    prompt = build_prompt(concepts, "transcript")
    recs = transcripts_from_source(args.manifest, args.source)
    if args.limit:
        recs = recs[:args.limit]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"loading {args.model} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    log(f"loaded in {time.time()-t0:.1f}s")

    rows = []
    for i, (rid, text) in enumerate(recs, 1):
        text = " ".join(text.split()[:args.max_words])
        msgs = [{"role": "user", "content": f"{prompt}\n\nTranscript:\n{text}"}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok([chat], return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=40, do_sample=False)
        resp = tok.decode(g[0][inp.input_ids.size(1):], skip_special_tokens=True)
        rows.append({"recording_id": rid, "concepts": parse_ids(resp, valid)})
        if i % 10 == 0 or i == len(recs):
            log(f"  {i}/{len(recs)}")

    write_jsonl(args.out, rows)
    log(f"wrote {len(rows)} recording predictions -> {args.out}")


if __name__ == "__main__":
    main()

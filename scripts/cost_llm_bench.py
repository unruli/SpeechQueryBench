#!/usr/bin/env python3
"""Micro-benchmark: seconds per Qwen2.5-7B multi-label call over one AMI meeting transcript.

ASR+text-LLM answers the 12-concept workload with one call per meeting (multi-label), so this per-call
time x 12 meetings is the per-query-batch LLM cost. Times steady-state calls (excludes model load).
"""
from __future__ import annotations
import time, sys
sys.path.insert(0, "scripts")
from common import read_manifest
from sem_common import load_concepts, build_prompt

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
utts = read_manifest("data/manifests/ami100.csv")
# one meeting's reference transcript
from collections import defaultdict
buf = defaultdict(list)
for u in utts:
    buf[u.recording_id].append(u.ref_text)
rid = "EN2002a"
text = " ".join(buf.get(rid, []))[:60000]
concepts = load_concepts("workload/semantic_concepts.yaml")
prompt = build_prompt(concepts, "transcript")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda").eval()
msgs = [{"role": "user", "content": f"{prompt}\n\nTranscript:\n{text}"}]
chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp = tok([chat], return_tensors="pt").to("cuda")
print(f"input tokens={inp.input_ids.size(1)}")
# warmup
with torch.no_grad():
    model.generate(**inp, max_new_tokens=40, do_sample=False)
ts = []
for _ in range(3):
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        model.generate(**inp, max_new_tokens=40, do_sample=False)
    torch.cuda.synchronize(); ts.append(time.time() - t)
print(f"TEXT-LLM sec/call (Qwen2.5-7B, 1 meeting, 12-concept multi-label) = {sum(ts)/len(ts):.2f}s (min {min(ts):.2f})")
del model, inp
torch.cuda.empty_cache()

# ---- embedding build + query encode timing (GPU), all 783 semantic passages ----
import glob, json
from sentence_transformers import SentenceTransformer
passages = []
for f in glob.glob("results/ami/semantic/chunks/*.jsonl"):
    for l in open(f):
        passages.append(json.loads(l)["text"])
em = SentenceTransformer("all-mpnet-base-v2", device="cuda")
em.encode(["warmup"])
torch.cuda.synchronize(); t = time.time()
em.encode(passages, batch_size=64, show_progress_bar=False)
torch.cuda.synchronize(); bt = time.time() - t
t = time.time()
em.encode(["which meeting discussed the remote-control budget"])
qt = time.time() - t
print(f"EMBED build (GPU L4): {bt:.2f}s for {len(passages)} passages = {bt/len(passages)*1000:.2f} ms/passage")
print(f"EMBED query encode (GPU L4): {qt*1000:.1f} ms/query")

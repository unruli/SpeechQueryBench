#!/usr/bin/env python3
"""Text-embedding retrieval architecture for semantic queries (the vector-search / RAG baseline).

Proper passage retrieval: split each meeting transcript into fixed word-window passages, embed all
passages + each concept description with a sentence encoder, and score (concept, meeting) by the MAX
cosine similarity over the meeting's passages (i.e., does ANY passage discuss the concept). Output
per-meeting concept scores; sem_eval.py --score-pred sweeps a similarity threshold.

--source reference    -> upper bound (embed the human transcript)
--source <hyps.jsonl> -> the realistic "ASR + text embedding" architecture

Usage (venv with sentence-transformers, GPU):
    python scripts/sem_embed.py --manifest AMI.csv --concepts workload/semantic_concepts.yaml \
        --source results/ami/asr/large-v3/hyps.jsonl --out results/ami/semantic/embed_largev3.jsonl --limit 12
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from common import read_manifest, read_jsonl, write_jsonl, log
from sem_common import load_concepts


def rec_texts(manifest, source):
    utts = read_manifest(manifest)
    order = []
    for u in utts:
        if u.recording_id not in order:
            order.append(u.recording_id)
    buf = defaultdict(list)
    if source == "reference":
        for u in utts:
            buf[u.recording_id].append(u.ref_text)
    else:
        for r in read_jsonl(source):
            buf[r["recording_id"]].append(r.get("hyp_text", ""))
    return [(rid, " ".join(buf.get(rid, []))) for rid in order]


def passages(text, win=50):
    w = text.split()
    if not w:
        return [""]
    return [" ".join(w[i:i + win]) for i in range(0, len(w), win)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--concepts", required=True)
    ap.add_argument("--source", required=True, help="'reference' or a hyps.jsonl path")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    concepts = load_concepts(args.concepts)
    recs = rec_texts(args.manifest, args.source)
    if args.limit:
        recs = recs[:args.limit]

    from sentence_transformers import SentenceTransformer, util
    log(f"loading {args.model} on {args.device} ...")
    model = SentenceTransformer(args.model, device=args.device)
    concept_emb = model.encode([c["desc"] for c in concepts], convert_to_tensor=True,
                               normalize_embeddings=True)

    rows = []
    for rid, text in recs:
        ps = passages(text, args.win)
        pe = model.encode(ps, convert_to_tensor=True, normalize_embeddings=True,
                          show_progress_bar=False)
        sims = util.cos_sim(concept_emb, pe)          # [n_concept, n_passage]
        maxsim = sims.max(dim=1).values               # best passage per concept
        rows.append({"recording_id": rid,
                     "scores": {str(concepts[j]["id"]): round(float(maxsim[j]), 4)
                                for j in range(len(concepts))}})
        log(f"  {rid}: {len(ps)} passages")
    write_jsonl(args.out, rows)
    log(f"wrote {len(rows)} recording concept-score rows -> {args.out}")


if __name__ == "__main__":
    main()

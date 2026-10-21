#!/usr/bin/env python3
"""Build the human-validation kit for the semantic gold: 3 blinded annotator CSVs + transcripts + key.

Each of the 12 AMI meetings x 12 concepts = 144 (meeting, concept) pairs gets a stable opaque `code`.
All three annotators label the SAME 144 pairs (full overlap -> Fleiss kappa), but the row order is
randomized per annotator (meetings shuffled; concepts shuffled within each meeting, kept contiguous so a
transcript is read once). No LLM labels are shown. A private _key.csv (code -> real meeting/concept +
LLM gold) is kept back for scoring; do NOT send it to annotators.

Usage (.venvASR312, CPU):
    python scripts/build_annotation_kit.py --gold results/ami/semantic/gold.jsonl \
        --manifest data/manifests/ami100.csv --concepts workload/semantic_concepts.yaml --out-dir annotation_kit
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import yaml

from common import read_jsonl

GUIDELINES = """# Semantic Gold Annotation — Guidelines

Thank you for helping validate this dataset. **Est. time: ~1.5–2 hours.**

## Your task
You will read {n} short meeting transcripts (coded M01–M{nn}) and, for each, decide whether the meeting
**discusses** each of 12 topics. You record one answer per row in your CSV: type **yes** or **no** in the
`discussed?` column.

## What counts as "discussed" (please read carefully — this is the whole task)
Mark **yes** only if the meeting **substantively discusses** the topic:
- participants talk about it across **at least a couple of turns**, **make a decision** about it, or
  clearly **dwell on it** as a subject of conversation.

Mark **no** if the topic is:
- only **mentioned in passing** (a single word or a one-line aside), or
- **not present at all**.

The test to apply when unsure: *"Was this an actual topic of discussion, or just a fleeting mention?"*
A word appearing once does **not** make the answer yes.

### Worked examples
- Topic: "the production cost or budget of the device." If two people debate whether 12 euros is
  affordable and agree on a price cap → **yes**. If someone just says "...and that's worth its cost"
  once, with no discussion → **no**.
- Topic: "the materials or casing used." If the group weighs plastic vs. rubber for several turns →
  **yes**. If "case" appears once meaning a legal case → **no**.

## How to fill the sheet
1. Open your CSV (`annotator_<X>.csv`) and the `transcripts/` folder.
2. Rows are grouped by meeting. For each meeting code (e.g., `M03`), open `transcripts/M03.txt`, read it
   once, then answer all of that meeting's rows.
3. Put **yes** or **no** in the `discussed?` column. Optionally add a short `notes` entry for tricky cases.
4. Do **not** change the `code` column. Do **not** discuss your answers with the other annotators.
5. Return the completed CSV.

Questions about edge cases? Note them in `notes` and flag them — do not guess silently.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--concepts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--annotators", default="A,B,C")
    args = ap.parse_args()

    gold_rows = list(read_jsonl(args.gold))
    meetings = [r["recording_id"] for r in gold_rows]                       # 12 meetings
    llm_gold = {r["recording_id"]: set(int(c) for c in r["concepts"]) for r in gold_rows}
    concepts = yaml.safe_load(open(args.concepts))["concepts"]              # id,name,desc
    ref = {r["recording_id"]: r.get("ref_text", "")
           for r in csv.DictReader(open(args.manifest)) if r["recording_id"] in set(meetings)}

    out = Path(args.out_dir); (out / "transcripts").mkdir(parents=True, exist_ok=True)
    code_of = {}   # (meeting, concept_id) -> code ; meeting -> coded id
    mcode = {m: f"M{ i+1:02d}" for i, m in enumerate(sorted(meetings))}
    for m in sorted(meetings):
        (out / "transcripts" / f"{mcode[m]}.txt").write_text(
            f"Transcript {mcode[m]}\n{'='*40}\n\n" + (ref.get(m, "").strip() or "[transcript missing]") + "\n")
    # stable codes P001..P144 in meeting-major order
    i = 0
    key_rows = []
    for m in sorted(meetings):
        for c in concepts:
            i += 1
            code = f"P{i:03d}"
            code_of[(m, c["id"])] = code
            key_rows.append({"code": code, "meeting_coded": mcode[m], "meeting_real": m,
                             "concept_id": c["id"], "concept_name": c["name"],
                             "concept_desc": c["desc"], "llm_gold": int(c["id"] in llm_gold[m])})

    with open(out / "_key_PRIVATE_do_not_send.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["code", "meeting_coded", "meeting_real", "concept_id",
                                          "concept_name", "concept_desc", "llm_gold"])
        w.writeheader(); w.writerows(key_rows)

    for a_i, ann in enumerate(args.annotators.split(",")):
        rng = random.Random(1000 + a_i)
        ms = sorted(meetings); rng.shuffle(ms)
        rows = []
        for m in ms:
            cs = list(concepts); rng.shuffle(cs)
            for c in cs:
                rows.append({"code": code_of[(m, c["id"])], "transcript": mcode[m],
                             "concept": c["desc"], "discussed?": "", "notes": ""})
        with open(out / f"annotator_{ann}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["code", "transcript", "concept", "discussed?", "notes"])
            w.writeheader(); w.writerows(rows)

    (out / "GUIDELINES.md").write_text(GUIDELINES.format(n=len(meetings), nn=f"{len(meetings):02d}"))
    print(f"kit -> {out}: {len(meetings)} meetings x {len(concepts)} concepts = {len(key_rows)} pairs; "
          f"annotators {args.annotators}; transcripts + GUIDELINES + private key written.")


if __name__ == "__main__":
    main()

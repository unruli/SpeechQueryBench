# Semantic Gold Annotation — Guidelines

Thank you for helping validate this dataset. **Est. time: ~1.5–2 hours.**

## Your task
You will read 12 short meeting transcripts (coded M01–M12) and, for each, decide whether the meeting
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

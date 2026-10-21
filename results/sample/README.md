# Sample Results

After running the benchmark, output files appear here.
Each script writes CSV or JSON results that the evaluation scripts consume.

Expected outputs:
- `lexical_results.csv` — per-keyword precision/recall/F1
- `aggregate_results.csv` — count/proportion errors
- `semantic_results.csv` — concept retrieval scores
- `interaction_results.csv` — diarization metrics
- `emotion_results.csv` — emotion classification accuracy
- `acoustic_results.csv` — music detection accuracy
- `bootstrap_cis.csv` — 95% bootstrap confidence intervals
- `pairwise_pvalues.csv` — Holm-corrected p-values

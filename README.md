# SpeechQueryBench

**Benchmarking Query Processing over Speech Archives [Experiments & Analysis]**

Anonymous submission to SIGMOD 2027

---

## Overview

SpeechQueryBench is a benchmark for end-to-end query processing over speech archives.
It evaluates how different execution architectures (ASR transcription, text-LLM reasoning,
speech-LLM direct QA, embedding retrieval, diarization, and acoustic models) perform across
**six query families**: lexical, aggregate, semantic, interaction/temporal, paralinguistic,
and acoustic-context.

The benchmark spans **ten transcription systems** (seven ASR, three audio LLMs) and
**three corpora** (LibriSpeech, AMI, AfriSpeech-Dialog), plus CREMA-D for emotion and
GTZAN for acoustic-context evaluation.

## Repository Structure

```
SpeechQueryBench/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── config/
│   └── paths.env              # Path configuration (edit for your environment)
├── workload/
│   ├── queries.jsonl           # LibriSpeech query workload
│   ├── ami_queries.jsonl       # AMI-34 query workload
│   ├── ami100_queries.jsonl    # AMI-100 query workload
│   ├── afrispeech_queries.jsonl# AfriSpeech query workload
│   └── semantic_concepts.yaml  # Semantic concept definitions for AMI
├── scripts/
│   ├── common.py               # Shared utilities (normalization, WER, IO)
│   ├── run_asr.py              # ASR transcription (Whisper, Granite, Parakeet)
│   ├── run_granite.py          # Granite Speech transcription
│   ├── run_af3.py              # AudioFlamingo3 transcription
│   ├── run_kimi.py             # Kimi-Audio transcription
│   ├── run_speechllm.py        # Speech-LLM transcription (Qwen2-Audio)
│   ├── chunk_transcribe.py     # Long-audio chunked transcription
│   ├── run_queries.py          # Execute lexical/aggregate queries over transcripts
│   ├── bootstrap.py            # Bootstrap confidence intervals and significance tests
│   ├── pairwise_significance.py# Pairwise bootstrap with Holm-Bonferroni correction
│   ├── metric_compare.py       # WER vs WMR comparison
│   ├── sem_textllm.py          # Semantic queries via ASR+text-LLM
│   ├── sem_embed.py            # Semantic queries via text-embedding retrieval
│   ├── sem_speechllm.py        # Semantic queries via direct speech-LLM QA
│   ├── sem_clap.py             # Semantic queries via CLAP embedding
│   ├── sem_bm25.py             # Semantic queries via BM25
│   ├── sem_hybrid.py           # Retrieval→speech-LLM hybrid
│   ├── sem_eval.py             # Semantic evaluation (precision/recall/F1)
│   ├── sem_common.py           # Shared semantic utilities
│   ├── interaction_diarize.py  # Diarization for interaction queries (pyannote)
│   ├── interaction_gold.py     # Interaction query gold standard
│   ├── interaction_eval.py     # Interaction query evaluation
│   ├── run_emotion_qwen.py     # Paralinguistic: speech-LLM emotion QA
│   ├── run_emotion_clap.py     # Paralinguistic: CLAP emotion matching
│   ├── run_emotion_textllm.py  # Paralinguistic: transcript+text-LLM emotion
│   ├── emotion_probe.py        # Paralinguistic: prosody probe classifier
│   ├── emotion_egemaps.py      # Paralinguistic: openSMILE eGeMAPS+SVM
│   ├── emotion_eval.py         # Paralinguistic evaluation
│   ├── extract_prosody.py      # Extract acoustic features for prosody probe
│   ├── run_acoustic_clap.py    # Acoustic-context: CLAP music detection
│   ├── run_acoustic_textllm.py # Acoustic-context: transcript baseline
│   ├── acoustic_eval.py        # Acoustic-context evaluation
│   ├── acquire_acoustic_mix.py # Create music-mixed utterances (GTZAN+LibriSpeech)
│   ├── acquire_cremad.py       # Download/prepare CREMA-D emotion corpus
│   ├── snr_sensitivity.py      # SNR sensitivity analysis
│   ├── cost_quality.py         # Cost-quality Pareto analysis
│   ├── cost_semantic.py        # Semantic cost modeling
│   ├── cost_llm_bench.py       # LLM cost benchmarking
│   ├── reliability_data.py     # Build reliability prediction dataset
│   ├── reliability_eval.py     # Reliability prediction evaluation
│   ├── router_pilot.py         # Pilot adaptive cascade
│   ├── eval_pilot.py           # Evaluate cascade pilot
│   ├── exp_semantic_threshold.py  # Semantic threshold sensitivity
│   ├── exp_ami100_pairwise.py  # AMI-100 pairwise significance
│   ├── exp_cross_repr_effects.py  # Cross-representation effect sizes
│   ├── accent_strat.py         # Accent stratification analysis
│   ├── gen_workload.py         # Generate query workloads
│   ├── show_summary.py         # Print summary tables
│   ├── score_human_gold.py     # Score human annotation agreement
│   ├── build_annotation_kit.py # Build human annotation materials
│   ├── build_manifest_librispeech.py  # LibriSpeech manifest
│   ├── build_manifest_ami.py   # AMI manifest
│   ├── build_manifest_afrispeech.py   # AfriSpeech manifest
│   ├── setup_env.sh            # Environment setup
│   └── fetch_ami_audio.sh      # Download AMI audio
├── runners/                    # SLURM batch scripts for cluster execution
│   ├── sbatch_asr.sh
│   ├── sbatch_semantic.sh
│   ├── sbatch_diarize.sh
│   ├── sbatch_emotion_qwen.sh
│   ├── sbatch_emotion_clap.sh
│   ├── sbatch_acoustic_textllm.sh
│   ├── sbatch_snr.sh
│   ├── sbatch_pairwise.sh
│   └── ... (other SLURM scripts)
├── annotation_kit/
│   └── GUIDELINES.md           # Human annotation guidelines for semantic gold
└── results/
    └── sample/                 # Sample output format (see Reproducing Results)
```

## Quick Start

### 1. Environment Setup

```bash
# Clone this repository
git clone <this-repo-url> SpeechQueryBench
cd SpeechQueryBench

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Paths

Edit `config/paths.env` to point to your data directories:

```bash
source config/paths.env
```

### 3. Download Corpora

The benchmark uses publicly available corpora:

- **LibriSpeech test-clean**: https://www.openslr.org/12
- **AMI Meeting Corpus**: https://groups.inf.ed.ac.uk/ami/corpus/
- **AfriSpeech-Dialog**: https://huggingface.co/datasets/intronhealth/afrispeech-dialog
- **CREMA-D**: https://github.com/CheyneyComputerScience/CREMA-D
- **GTZAN**: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Build corpus manifests:

```bash
python scripts/build_manifest_librispeech.py
python scripts/build_manifest_ami.py
python scripts/build_manifest_afrispeech.py
```

### 4. Run Transcription

```bash
# Whisper tiers
python scripts/run_asr.py --model tiny.en --corpus librispeech
python scripts/run_asr.py --model small.en --corpus librispeech
python scripts/run_asr.py --model large-v3 --corpus librispeech

# Other ASR
python scripts/run_granite.py --corpus librispeech
# ... (see runners/ for full SLURM scripts)
```

### 5. Run Queries and Evaluation

```bash
# Lexical and aggregate queries
python scripts/run_queries.py --corpus librispeech

# Semantic queries (AMI)
python scripts/sem_textllm.py
python scripts/sem_embed.py
python scripts/sem_speechllm.py
python scripts/sem_eval.py

# Paralinguistic (CREMA-D)
python scripts/run_emotion_qwen.py
python scripts/emotion_probe.py
python scripts/emotion_eval.py

# Acoustic-context
python scripts/acquire_acoustic_mix.py
python scripts/run_acoustic_clap.py
python scripts/acoustic_eval.py

# Statistical analysis
python scripts/bootstrap.py
python scripts/pairwise_significance.py
python scripts/cost_quality.py
```

## Reproducing Key Results

| Paper section | Script(s) | Output |
|---|---|---|
| Table 1 (WER vs query utility) | `run_queries.py`, `bootstrap.py` | Per-system WER, F1, cntErr with CIs |
| Table 2 (WMR ranking) | `metric_compare.py` | WER vs WMR system rankings |
| Table 3 (WMR correlation) | `metric_compare.py` | Spearman ρ for WER/WMR vs query error |
| Table 4 (Reliability) | `reliability_eval.py` | ROC-AUC for reliability predictors |
| Table 5 (Cascade) | `eval_pilot.py` | Cascade recall at matched budgets |
| Table 6 (Architecture summary) | `show_summary.py` | Best architecture per family |
| Table 7 (Cost-quality) | `cost_quality.py` | RTF, $/audio-hour, F1rare |
| Figure 2 (Semantic) | `sem_eval.py` | F1 per semantic architecture |
| Table 8 (Semantic cost) | `cost_semantic.py` | Cost vs query volume |
| Table 9 (Interaction) | `interaction_eval.py` | Diarization accuracy |
| Table 10 (Emotion) | `emotion_eval.py` | Emotion accuracy per architecture |
| Table 11 (Acoustic) | `acoustic_eval.py` | Music detection/genre accuracy |
| Figure 4 (Scale) | `exp_ami100_pairwise.py` | AMI-34 vs AMI-100 |
| Table 14 (Effect sizes) | `exp_cross_repr_effects.py` | Cohen's d per comparison |
| Table 15 (Pairwise) | `pairwise_significance.py` | Bootstrap p-values with Holm correction |

## Query Workload Format

Queries are stored as JSONL files in `workload/`. Each line is a JSON object:

```json
{
  "query_id": "lex_rare_001",
  "family": "lexical",
  "type": "selection",
  "keyword": "magnetometer",
  "corpus": "librispeech",
  "recording_ids": ["1089-134686-0000", ...],
  "gold_positive": ["1089-134686-0012"]
}
```

## Metrics

- **Lexical**: Precision, Recall, F1 (per keyword rarity stratum)
- **Aggregate**: Count relative error, proportion error
- **Semantic**: Precision, Recall, F1 (concept retrieval)
- **Interaction**: Speaker count MAE, turn count MAE, overlap detection accuracy
- **Paralinguistic**: Accuracy, macro-F1 (six-way emotion classification)
- **Acoustic-context**: Balanced accuracy (music detection), accuracy (genre)
- **WMR**: Word miss rate = (S+D)/N (WER minus insertions)
- **Statistical**: Bootstrap 95% CIs (B=5000), pairwise p-values with Holm-Bonferroni, Cohen's d

## Hardware Requirements

- GPU: NVIDIA A100 80GB (for large models) or L4 (for smaller models/embeddings)
- The full benchmark takes approximately 48 GPU-hours on A100
- Individual experiments can run on consumer GPUs with reduced batch sizes

## Model Checkpoints

All models are publicly available on Hugging Face:

| Model | Checkpoint |
|---|---|
| Whisper tiny.en | `openai/whisper-tiny.en` |
| Whisper small.en | `openai/whisper-small.en` |
| Whisper large-v3 | `openai/whisper-large-v3` |
| Granite Speech | `ibm-granite/granite-speech-3.3-2b` |
| Parakeet-TDT | `nvidia/parakeet-tdt-0.6b-v2` |
| Canary-Qwen | `nvidia/canary-qwen-2.5b` |
| Qwen3-ASR | `Qwen/Qwen3-ASR-1.7B` |
| Qwen2-Audio | `Qwen/Qwen2-Audio-7B-Instruct` |
| AudioFlamingo3 | `nvidia/audio-flamingo-3` |
| Kimi-Audio | `MoonshotAI/Kimi-Audio` |
| Qwen2.5-7B (text LLM) | `Qwen/Qwen2.5-7B-Instruct` |
| CLAP | `laion/larger_clap_general` |

## License

This benchmark artifact is provided for research purposes. The underlying corpora
(LibriSpeech, AMI, AfriSpeech-Dialog, CREMA-D, GTZAN) are subject to their respective
licenses. See each corpus's documentation for terms.

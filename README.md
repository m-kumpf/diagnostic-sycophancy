# Syco100 — Diagnostic sycophancy in large language models

Code and data for: *LLM diagnostic deference under user pressure* (Kumpf et al., submitted to BMJ Health & Care Informatics).

100 single-best-answer clinical vignettes were presented to five contemporary LLMs under three single-turn conditions (neutral, mild incorrect preference, severe incorrect assertion with claimed specialist authority), 10 repetitions per case-condition cell with independently shuffled answer options (15,000 responses total).

## Contents

- `case_vignettes.xlsx` — the 100-vignette item bank (patient profile, chief complaint, history, vitals, examination, diagnostics, correct diagnosis, fixed suggested distractor DD_a, and distractors DD_b–DD_d). Trigger sentences are constructed at run time by `prompt_builder.py`, not stored in the sheet.
- `src/` — experiment and analysis code:
  - `prompt_builder.py` — prompt templates and trigger sentences for the three conditions (seeded option shuffling).
  - `config.py` — model registry (OpenRouter model IDs) and run configuration (temperature 1.0, 10 repetitions, master seed).
  - `run_experiment.py`, `llm_client.py`, `data_loader.py`, `response_parser.py` — data collection pipeline.
  - `response_parser.py` — checks whether a raw response is a clean single letter (A–E); anything else is marked for recollection.
  - `redo_nonsingle.py` — re-calls cells whose response was not a bare single letter (identity-preserving, capped retries).
  - `export_csv.py` — converts raw JSONL logs to CSV.
  - `analysis.py`, `inference.py` — descriptive rates, case-clustered GEE models, cluster-bootstrap risk differences, permutation tests, subset analyses.
- `results/raw/` — complete model outputs (one CSV + JSONL per model; 3,000 calls each).

## Reproducing the analysis

```bash
pip install -r requirements.txt
python -m src.analysis          # regenerates analysis tables from results/raw/
```

Re-running data collection requires an OpenRouter API key (see `.env.example`). Note that provider-side model updates mean fresh collections may not reproduce the reported behaviour; the shipped raw outputs correspond to the collection window of 28–30 May 2026.

## Citation

Citation details will be added upon publication.

## License

Code (`src/`) is released under the MIT License (`LICENSE`). Data (the vignette
bank and `results/raw/`) is released under CC BY 4.0 (`LICENSE-DATA.md`).

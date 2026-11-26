# Diagnostic sycophancy in large language models

Code and data for *User pressure changes large language model diagnostic choices: a vignette-based evaluation for clinical decision support* (Kumpf et al.).

Five models answered 100 clinical vignettes under neutral, mild and severe wording. Each vignette had five answer options and one fixed suggested distractor. Ten repetitions per model, condition and vignette yielded 15,000 final responses. The answer options were shuffled independently for each initial query.

## Reproduce the results

The analysis runs offline from the included data. From the repository root, the following commands set up the tested Python 3.12 environment and run the full analysis.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.reproduce
```

The package versions in `requirements.txt` were tested with Python 3.12.7. The manuscript reports the environment used for the original analysis. The command above regenerates the tables, the full example prompts and Figures 2 and 3. Figure 1 is the supplied vignette illustration. The tables and figures are also included for direct inspection.

## Data

- [Vignette bank](case_vignettes.xlsx). The workbook contains the case descriptions, correct diagnoses and four distractors per case. `DD_a` is the diagnosis named in the two pressure prompts.
- [Final responses](results/raw/). Each model has one JSONL file and one CSV. The JSONL files contain the exact prompt, shuffled options, correct and suggested letters, returned text and parsed choice. The CSVs are flat extracts. One record represents a planned query after any format-related repeats.
- [First-pass metadata](results/raw/metadata/). `first_pass_metadata.csv` identifies each repeated query and whether its first response was empty. `first_pass_choice_counts.csv` gives the number repeated, the number with a recoverable choice and the number selecting the distractor, by model, condition and initial UTC date. These counts and the unrepeated responses reproduce the first-pass comparison. The release contains only summary metadata for first replies that were replaced.
- [Prompt construction](src/prompt_builder.py) and [request configuration](src/config.py). Each request used one English user message, no system message and temperature 1.0. Other sampling and reasoning settings were omitted. Provider routing was unpinned. The logs record the requested endpoint, with no serving-provider, exact snapshot or quantisation metadata. The local seed controls option order. Initial replies were collected on 24–29 May 2026 UTC and format-related repeats on 30 May.

## Published tables and figures

| Manuscript item | Repository output |
| --- | --- |
| Table 1. Accuracy and distractor selection | [Rates and case-bootstrap intervals](results/analysis/descriptive_rates.csv) |
| Table 2. Model-specific pressure contrasts | [Risk differences, odds ratios and p values](results/analysis/table_2.csv) |
| Figure 1 and Supplementary Text S1 | [Vignette illustration](figures/Figure_1.png) and [the three full prompts](results/supplementary/example_prompts.txt) |
| Figure 2. Distractor selection by condition | [Vector figure](figures/Figure_2.pdf) |
| Figure 3. Full sample and high-consistency cases | [Vector figure](figures/Figure_3.pdf) and [plotted rates](results/supplementary/figure_3_rates.csv) |
| Table S1. Orthopaedic and other vignettes | [Stratum estimates](results/supplementary/specialty_strata.csv) and [between-group differences](results/supplementary/specialty_difference.csv) |
| Table S2. Neutral distractor selection and the effect of pressure | [Group estimates and Spearman correlations](results/supplementary/distractor_plausibility.csv) and [case-level data](results/supplementary/distractor_plausibility_cases.csv) |
| Table S3. High-consistency cases with a held-out neutral baseline | [Rates and changes for both splits](results/supplementary/high_consistency.csv) |
| Table S4. Repeated queries and empty first responses | [Counts and percentages by model and condition](results/supplementary/format_resampling.csv) |
| Table S5. Requested endpoints and collection dates | [Model identifiers, weight availability, dates and repeats](results/supplementary/request_settings.csv) |
| Table S6. Accuracy loss and its distribution across vignettes | [Suggested and other wrong options](results/supplementary/accuracy_decomposition.csv), [case-level summary](results/supplementary/case_change_summary.csv) and [individual cases](results/supplementary/case_changes.csv) |

The primary contrast is severe minus neutral distractor selection within each model. Binomial generalised estimating equations use an exchangeable within-case correlation and robust standard errors. Each model has one primary test and four Holm-adjusted secondary tests. Risk-difference intervals use 2,000 case-bootstrap resamples. Paired sign-flip tests use 10,000 permutations. The original descriptive, inferential and case-level outputs are in [results/analysis](results/analysis/).

For Table S2, each model’s empirical distractor plausibility is estimated from the other four models’ 40 neutral responses per case. Table S3 and Figure 3 select cases answered correctly on neutral repetitions 1–5 and estimate the neutral baseline on repetitions 6–10. Table S3 also reports the reversed split. The additional `all ten` rows in its CSV reproduce the original analysis in which all ten neutral responses were used for selection.

Two further sensitivity analyses are included. The [first-pass condition rates](results/supplementary/first_pass_rates.csv) and [severe-minus-neutral differences](results/supplementary/first_pass_lifts.csv) use first responses with a recoverable choice. The [across-model Holm sensitivity analysis](results/supplementary/primary_holm_sensitivity.csv) treats the five primary tests as one family for comparison with the model-specific tests in Table 2.

## Collect new responses

New collections require an OpenRouter API key in `OPENROUTER_API_KEY` (see [.env.example](.env.example)). Use a separate output directory. For example, these commands collect first responses for one model, export first-pass metadata and repeat queries outside the single-letter format.

```sh
python -m src.run_experiment --model deepseek-v4-pro --out results/new-collection
python -m src.first_pass_metadata --first-pass-dir results/new-collection --out results/new-collection/metadata
python -m src.redo_nonsingle --model deepseek-v4-pro --out results/new-collection
```

For a collection spanning several models, collect all first responses and export their metadata before repeating any queries. Each format-related repeat uses the recorded prompt, option order and temperature. A new collection may differ as the endpoints or their serving settings change.

## Licence

The code is distributed under the [MIT License](LICENSE). The vignette bank and response data are distributed under [CC BY 4.0](LICENSE-DATA.md).

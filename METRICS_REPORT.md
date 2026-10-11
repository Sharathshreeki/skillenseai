# SkillLens AI — Evaluation Report

**Run date:** 2026-10-11  
**Dataset:** `data/skilllens_dataset.csv`  
**Evaluation:** deterministic 100-row sample, `random_state=42`  
**Command:** `python -m evaluate_labelled`

## Measured results

| Metric | Result |
|---|---:|
| True positives | 575 |
| False positives | 794 |
| False negatives | 1,251 |
| Precision | 0.4200 (42.00%) |
| Recall | 0.3149 (31.49%) |
| F1-score | 0.3599 (35.99%) |

These results are reproducible with the current taxonomy and code, but **do not meet the requested 80–90% precision and F1 target**.

## Changes made

- Merged canonical skill names from the supplied taxonomy v3 into the working taxonomy.
- Did not import taxonomy v3's raw aliases because many are generic n-grams, numeric fragments, compensation text, or ordinary job-description wording; importing them substantially increases false positives.
- Retained the previous curated aliases and applied a stricter gate to short aliases.
- Cached compiled regular expressions.
- Made transformer-based semantic matching optional and disabled it by default to avoid surprise model downloads and uncontrolled semantic false positives.
- Integrated the provided Streamlit UI, stylesheet, and CSV/JSON exporter with the extraction pipeline.
- Wired the Analyze button to pasted job descriptions and searchable PDF, DOCX, and TXT files.
- Added a consistent offline test run: **44 tests passed**.

## Why the target is not claimed

The dataset contains 5,774 distinct reference label strings across 1,167 job descriptions, while the working taxonomy contains 423 canonical entries. Reference labels vary in casing, wording, and granularity, and the dataset does not consistently annotate every skill phrase appearing in each description. A phrase-matching extractor therefore cannot reliably reach 0.80+ precision and F1 merely by adding aliases; broader matching increases false positives, while stricter matching reduces recall.

The correct next step is to manually review a gold-standard subset, split evaluation by job, and use a contextual model (or LLM-backed extraction with schema validation) tuned against a held-out set. Do not report the 80–90% target as achieved until a fresh held-out evaluation demonstrates it.

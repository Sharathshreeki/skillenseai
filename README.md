# SkillLens AI — Member 4 implementation

A local prototype scaffold for the CBC059 PS-16 AI-Based Skill Extraction from Job Descriptions challenge.

## Current implementation

- Shared dataclasses in `modules/schemas.py`.
- Rule-based Mandatory / Preferred / Uncertain classification with clause splitting, cue phrases, negation, conflicts, and heading context.
- Basic experience, education, and certification extraction.
- Pipeline orchestration with optional integration points for Member 3's extractor/normalizer and Member 2's document extractor.
- A limited fallback skill matcher for a self-contained demo before the full extractor is integrated.
- Unit tests for classifier and pipeline behavior.

The fallback is deliberately limited and is not an accuracy claim. Unknown or inferred skills should be reviewed by a human. The prototype currently accepts pasted text in the UI; file parsing and full normalization depend on the other members' modules.

## Setup (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

## Integration contracts

- `extract_skills(text) -> list[Mention]`
- `normalize_skills(mentions) -> list[Skill]`
- `analyze_job(text, job_id, title=None) -> JobProfile`
- `classify_requirements(text, skills) -> RequirementResult`

Character spans use Python string offsets, start-inclusive and end-exclusive. Do not change shared schema fields without coordinating with the team.

## Important limitations

- This is a prototype, not a validated production classifier.
- Cue lists and heuristics need evaluation against a labelled set.
- OCR supports common image formats and scanned PDFs. PaddleOCR is preferred when installed; Tesseract is used as a fallback. Searchable PDFs use native text extraction first.
- Do not publish accuracy or speed claims unless measured on a documented labelled sample.

## Skill extraction improvements (October 2026)

The shared taxonomy was expanded with the canonical skill names from `skill_taxonomy_v3.json`, while **the v3 alias lists were deliberately not imported**. That file contains thousands of broad n-gram aliases (including numbers, salary fragments, and generic job-description wording) that caused many false-positive matches. Curated aliases from the existing project taxonomy are retained, and low-signal one-token aliases are filtered during extraction.

Other reliability changes:

- Regex patterns are compiled and cached rather than rebuilt for every job description.
- Semantic matching is disabled by default so local runs do not unexpectedly download/load a transformer model. To opt in, install `sentence-transformers` and set `SKILLLENS_ENABLE_SEMANTIC=1`.
- The extractor remains deterministic and preserves exact source spans and evidence sentences.
- Tests were updated for the expanded taxonomy and offline-by-default matching.

### Reproduce evaluation

Run from the repository root:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. pytest -q
PYTHONPATH=. python evaluate_labelled.py
```

The evaluation script samples 100 rows using seed 42 and compares normalized canonical predictions with the dataset's `job_skill_set` labels. The latest local run after these changes reported:

- Precision: **0.4200**
- Recall: **0.3149**
- F1-score: **0.3599**

These are measured results, not a guarantee of performance on other data. They do **not** meet the requested 0.80–0.90 target. The supplied dataset has 5,774 distinct reference label strings, while the expanded taxonomy has 423 canonical entries; labels are inconsistent in casing and granularity, and many reference skills are not literally expressed in the description. Alias expansion alone cannot responsibly produce 0.80+ precision and F1. Claiming that target without a separately labelled validation set and stronger contextual model would be misleading. The next high-value step is to create a manually reviewed gold set, split it by job (not sentence), then train/evaluate a contextual skill extractor and tune thresholds against that held-out set.



## Requirement classification improvements (October 2026)

Recent fixes target the most common causes of excessive `Uncertain` labels:

- **Wrapped lines no longer break requirement context.** Text such as `Proficiency in Python,\nSQL` is evaluated as one clause, which is important for PDF extraction and OCR output.
- **Regex cues use real word boundaries.** This prevents words such as `unrequired` or `mustard` from accidentally matching `required` or `must`.
- **Explicit job-duty verbs provide contextual evidence.** Skills used in statements such as `Build predictive models` or `Develop data pipelines` are classified as `Mandatory`, with the basis recorded as `role_action:<verb>`.
- **Repeated mentions are deduplicated more carefully.** An uncued mention no longer overrides a separate clearly Mandatory or Preferred mention of the same skill. A genuine Mandatory-versus-Preferred contradiction remains `Uncertain` and requires review.
- Common requirement phrases such as **require/requires**, **must have**, **have to**, **needs to**, **proficiency in**, **expertise in**, **experience with**, and **shall** are supported. Headings such as **Required Skills** and **Required Qualifications** apply to list items underneath them.
- Negation is checked first: `Python is not required` remains `Uncertain` and flagged for review.

`Uncertain` remains appropriate when the evidence only names a skill without showing whether it is a requirement, a preference, or merely part of company/industry context. These rules are heuristics and should be evaluated against a manually reviewed labelled set before reporting classification accuracy.


### OCR setup

OCR extraction is integrated for PNG, JPG/JPEG, TIFF, BMP, WEBP, and scanned PDFs. The extractor tries PaddleOCR first, then Tesseract. Install at least one OCR engine.

- **PaddleOCR** (recommended if your team environment already uses it): `python -m pip install paddleocr paddlepaddle` (platform support varies; follow PaddleOCR's installation instructions for your OS/CPU).
- **Tesseract fallback on macOS**: `brew install tesseract`, then `python -m pip install pytesseract`.

Then restart Streamlit. The first PaddleOCR run may download OCR model files.

### Fix OCR setup on macOS

The error `No module named 'pytesseract'` means the Python package is not installed in the same interpreter that runs Streamlit. The message `No module named 'paddleocr'` is only telling you PaddleOCR is not installed; PaddleOCR is optional when Tesseract is used. Tesseract has two parts: the Python package (`pytesseract`) and the native `tesseract` executable.

Recommended setup from the project root:

```bash
brew install tesseract  # run once; skip if already installed
source .venv/bin/activate  # if the project already has this environment
python -m pip install pytesseract
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
python -m streamlit run app.py
```

If `.venv` does not exist, create and install the project environment first:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
python -m streamlit run app.py
```

Use `python -m streamlit` rather than a global `streamlit` command so the app runs under the same Python interpreter where OCR dependencies were installed.



## OCR setup (macOS)

The Input Data screen sends uploaded images and scanned PDFs through `modules/document_extractor.py`. Image OCR and scanned-PDF OCR require both the Python wrapper (`pytesseract`) and the native Tesseract executable. `pytesseract` is included in `requirements.txt`; the native executable is installed with Homebrew.

From the extracted project folder (`skillenseai`), run:

```bash
bash setup_ocr_macos.sh
.venv/bin/python check_ocr.py
.venv/bin/python -m streamlit run app.py
```

Use the exact `.venv/bin/python` interpreter above to start Streamlit so the app uses the same environment where OCR dependencies were installed. If setup reports that Homebrew is missing, install Homebrew from https://brew.sh first and rerun the script.

After uploading an image or scanned PDF, select **Analyze job descriptions**. Open **Review Profiles** and expand **Extracted source text (verify OCR output)** to confirm what OCR read before reviewing the extracted skills.

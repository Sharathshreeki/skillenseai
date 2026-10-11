"""
SkillLens AI — Skill Extraction Module (M3)

Extracts skill mentions from job descriptions while preserving:
- Original phrase
- Evidence sentence
- Character positions
- Extraction source
- Matching method

The taxonomy is loaded from data/skill_taxonomy.json when available.
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import spacy

from modules.schemas import Mention


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = PROJECT_ROOT / "data" / "skill_taxonomy.json"

# Lightweight tokenizer. No downloaded language model is required.
_NLP = spacy.blank("en")
_NLP.add_pipe("sentencizer")


# Temporary fallback for local development.
# Member 2's taxonomy will take precedence when the JSON file is available.
DEV_TAXONOMY = [
    {"id": "python", "name": "Python", "aliases": ["py"]},
    {"id": "java", "name": "Java", "aliases": []},
    {"id": "javascript", "name": "JavaScript", "aliases": ["js"]},
    {"id": "typescript", "name": "TypeScript", "aliases": ["ts"]},
    {"id": "sql", "name": "SQL", "aliases": []},
    {"id": "mysql", "name": "MySQL", "aliases": []},
    {"id": "sqlite", "name": "SQLite", "aliases": ["sqlite3"]},
    {"id": "html", "name": "HTML", "aliases": []},
    {"id": "css", "name": "CSS", "aliases": []},
    {"id": "react", "name": "React", "aliases": ["react.js", "reactjs"]},
    {"id": "nodejs", "name": "Node.js", "aliases": ["nodejs", "node js"]},
    {"id": "mongodb", "name": "MongoDB", "aliases": ["mongo db"]},
    {"id": "c", "name": "C", "aliases": []},
    {"id": "cpp", "name": "C++", "aliases": ["cpp"]},
    {"id": "csharp", "name": "C#", "aliases": ["c sharp", "c-sharp"]},
    {"id": "dotnet", "name": ".NET", "aliases": ["dotnet", "dot net"]},
    {"id": "rest-api", "name": "REST API", "aliases": ["RESTful API", "REST APIs"]},
    {"id": "excel", "name": "Excel", "aliases": ["Microsoft Excel"]},
    {
        "id": "google-sheets",
        "name": "Google Sheets",
        "aliases": ["Google spreadsheet", "Google spreadsheets"],
    },
    {"id": "git", "name": "Git", "aliases": []},
    {"id": "docker", "name": "Docker", "aliases": []},
    {"id": "kubernetes", "name": "Kubernetes", "aliases": ["k8s"]},
    {"id": "machine-learning", "name": "Machine Learning", "aliases": ["ML"]},
    {"id": "deep-learning", "name": "Deep Learning", "aliases": []},
    {"id": "data-analysis", "name": "Data Analysis", "aliases": []},
    {"id": "data-visualization", "name": "Data Visualization", "aliases": []},
]


def _load_taxonomy() -> list[dict[str, Any]]:
    """Load Member 2's taxonomy, or use the temporary development taxonomy."""
    if not TAXONOMY_PATH.exists():
        return DEV_TAXONOMY

    try:
       with TAXONOMY_PATH.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load skill taxonomy: {exc}") from exc

    # Accept either a list of entries or a dictionary containing a skills list.
    if isinstance(data, dict):
        data = data.get("skills", [])

    if not isinstance(data, list):
        raise ValueError("Taxonomy must be a list or a dictionary containing 'skills'.")

    for entry in data:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError("Each taxonomy entry must be an object with a 'name'.")

    return data


def _evidence_sentence(text: str, start: int, end: int) -> str:
    """Return the sentence containing the matched phrase."""
    doc = _NLP(text)

    for sentence in doc.sents:
        if sentence.start_char <= start < sentence.end_char:
            return sentence.text.strip()

    # spaCy's blank tokenizer has no sentence boundaries without a parser.
    left = max(text.rfind(".", 0, start), text.rfind("!", 0, start), text.rfind("?", 0, start))
    endings = [
        position
        for position in (
            text.find(".", end),
            text.find("!", end),
            text.find("?", end),
        )
        if position != -1
    ]
    right = min(endings) + 1 if endings else len(text)

    return text[left + 1:right].strip()


@lru_cache(maxsize=4)
def _compiled_patterns(taxonomy_json: str) -> tuple[tuple[str, str, re.Pattern], ...]:
    """Compile high-signal canonical names and aliases once per taxonomy."""
    taxonomy = json.loads(taxonomy_json)
    patterns = []
    short_alias_allowlist = {
        "c", "r", "go", "js", "ts", "sql", "ai", "bi", "hr", "qa", "ui", "ux", "ml",
        "c++", "c#", ".net", "aws", "gcp", "sap", "crm", "erp", "etl", "api", "css", "html",
        "excel", "python", "java", "javascript", "typescript", "kubernetes", "docker",
        "react", "angular", "tableau", "salesforce", "workday", "hris", "adp",
        "fmla", "ada", "flsa", "erisa", "powerpoint", "powerbi", "jira", "git",
    }
    low_signal_aliases = {
        "strategic", "administrative", "solutions", "culture", "policies", "processes",
        "professional", "responsible", "support", "services", "business", "operations",
        "requirements", "management", "experience", "knowledge", "skills", "ability",
        "communication", "team", "work", "working", "various", "related", "including",
    }
    for entry in taxonomy:
        skill_id = str(entry.get("id") or entry["name"].strip().lower().replace(" ", "-"))
        # Canonical names are valid candidates; aliases must pass a stricter quality gate.
        phrases = [(entry["name"], True)]
        for key in ("aliases", "review_aliases"):
            for item in entry.get(key, []):
                phrase = item.get("phrase", "") if isinstance(item, dict) else item
                if not isinstance(phrase, str) or not phrase.strip():
                    continue
                phrase = phrase.strip()
                token = phrase.casefold()
                word_count = len(re.findall(r"[a-z0-9+#.]+", token))
                if word_count < 2 and token not in short_alias_allowlist:
                    continue
                if token in low_signal_aliases:
                    continue
                if re.fullmatch(r"[\W\d_]+", phrase):
                    continue
                phrases.append((phrase, False))
        for phrase, is_canonical in phrases:
            patterns.append((phrase, skill_id, re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)))
    # Prefer longer spans, then canonical names over aliases for identical phrases.
    patterns.sort(key=lambda item: (len(item[0]), item[0].casefold()), reverse=True)
    return tuple(patterns)


def _build_patterns(taxonomy: list[dict[str, Any]]) -> list[tuple[str, str, re.Pattern]]:
    return list(_compiled_patterns(json.dumps(taxonomy, sort_keys=True)))


def extract_skills(text: str) -> list[Mention]:
    """
    Extract skill mentions from text.

    Returns:
        A list of Mention objects with exact character spans and evidence.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    taxonomy = _load_taxonomy()
    patterns = _build_patterns(taxonomy)

    candidates = []

    for phrase, _skill_id, pattern in patterns:
        for match in pattern.finditer(text):
            start, end = match.span()

            candidates.append(
                {
                    "original_phrase": match.group(),
                    "span_start": start,
                    "span_end": end,
                    "matched_via": (
                        "exact"
                        if match.group().casefold() == phrase.casefold()
                        else "alias"
                    ),
                }
            )

    # Prefer longer matches when aliases overlap.
    candidates.sort(
        key=lambda item: (
            item["span_start"],
            -(item["span_end"] - item["span_start"]),
        )
    )

    selected = []

    for candidate in candidates:
        overlaps = any(
            candidate["span_start"] < existing["span_end"]
            and candidate["span_end"] > existing["span_start"]
            for existing in selected
        )

        if not overlaps:
            selected.append(candidate)

    mentions = []

    for candidate in sorted(selected, key=lambda item: item["span_start"]):
        start = candidate["span_start"]
        end = candidate["span_end"]

        mentions.append(
            Mention(
                original_phrase=candidate["original_phrase"],
                evidence=_evidence_sentence(text, start, end),
                span_start=start,
                span_end=end,
                source="explicit",
                matched_via=candidate["matched_via"],
            )
        )

    return mentions
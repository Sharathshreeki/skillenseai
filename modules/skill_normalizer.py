"""
SkillLens AI — Skill Normalization Module (M3).

Maps extracted Mention objects to canonical Skill objects using:
1. Exact canonical-name matching
2. Alias matching
3. Review-alias matching
4. Fuzzy matching with RapidFuzz
5. Unmapped fallback

The requirement classifier is owned by Member 4, so requirement
remains "Uncertain" throughout this module.
"""

import json
import re
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process
from modules.schemas import Mention, Skill

# Semantic matching is optional: never require a model download just to run extraction.
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - depends on local environment
    SentenceTransformer = None
    cosine_similarity = None

SEMANTIC_THRESHOLD = 0.75
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_SEMANTIC_MODEL = None
_SEMANTIC_EMBEDDINGS = {}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = PROJECT_ROOT / "data" / "skill_taxonomy.json"

FUZZY_THRESHOLD = 90.0


# Temporary development taxonomy.
# Member 2's taxonomy takes precedence when its JSON file is available.
DEV_TAXONOMY = [
    {
        "id": "python",
        "name": "Python",
        "category": "Technical",
        "aliases": ["py"],
        "review_aliases": [],
        "related": [],
        "implied_by": [],
    },
    {
        "id": "javascript",
        "name": "JavaScript",
        "category": "Technical",
        "aliases": ["JS", "ECMAScript"],
        "review_aliases": [],
        "related": ["java", "typescript"],
        "implied_by": [],
    },
    {
        "id": "java",
        "name": "Java",
        "category": "Technical",
        "aliases": [],
        "review_aliases": [],
        "related": ["javascript"],
        "implied_by": [],
    },
    {
        "id": "typescript",
        "name": "TypeScript",
        "category": "Technical",
        "aliases": ["TS"],
        "review_aliases": [],
        "related": ["javascript"],
        "implied_by": [],
    },
    {
        "id": "sql",
        "name": "SQL",
        "category": "Technical",
        "aliases": [],
        "review_aliases": [],
        "related": ["mysql", "sqlite"],
        "implied_by": [],
    },
    {
        "id": "mysql",
        "name": "MySQL",
        "category": "Technical",
        "aliases": [],
        "review_aliases": [],
        "related": ["sql", "sqlite"],
        "implied_by": [],
    },
    {
        "id": "sqlite",
        "name": "SQLite",
        "category": "Technical",
        "aliases": ["sqlite3"],
        "review_aliases": [],
        "related": ["sql", "mysql"],
        "implied_by": [],
    },
    {
        "id": "html",
        "name": "HTML",
        "category": "Technical",
        "aliases": [],
        "review_aliases": [],
        "related": [],
        "implied_by": [],
    },
    {
        "id": "css",
        "name": "CSS",
        "category": "Technical",
        "aliases": [],
        "review_aliases": [],
        "related": [],
        "implied_by": [],
    },
    {
        "id": "react",
        "name": "React",
        "category": "Technical",
        "aliases": ["React.js", "ReactJS"],
        "review_aliases": [],
        "related": ["javascript"],
        "implied_by": [],
    },
    {
        "id": "nodejs",
        "name": "Node.js",
        "category": "Technical",
        "aliases": ["NodeJS", "Node JS"],
        "review_aliases": [],
        "related": ["javascript"],
        "implied_by": [],
    },
    {
        "id": "rest-api",
        "name": "REST API",
        "category": "Technical",
        "aliases": ["REST APIs", "RESTful API", "RESTful APIs"],
        "review_aliases": [],
        "related": ["api"],
        "implied_by": [],
    },
    {
        "id": "excel",
        "name": "Excel",
        "category": "Technical",
        "aliases": ["Microsoft Excel"],
        "review_aliases": [],
        "related": ["google-sheets"],
        "implied_by": [],
    },
    {
        "id": "google-sheets",
        "name": "Google Sheets",
        "category": "Technical",
        "aliases": [],
        "review_aliases": [],
        "related": ["excel"],
        "implied_by": [],
    },
    {
        "id": "cpp",
        "name": "C++",
        "category": "Technical",
        "aliases": ["CPP"],
        "review_aliases": [],
        "related": ["c", "csharp"],
        "implied_by": [],
    },
    {
        "id": "csharp",
        "name": "C#",
        "category": "Technical",
        "aliases": ["C Sharp", "C-Sharp"],
        "review_aliases": [],
        "related": ["c", "cpp"],
        "implied_by": [],
    },
    {
        "id": "dotnet",
        "name": ".NET",
        "category": "Technical",
        "aliases": ["dotnet", "dot net"],
        "review_aliases": [],
        "related": [],
        "implied_by": [],
    },
]


def _load_taxonomy() -> list[dict[str, Any]]:
    """Load the team's taxonomy or the temporary development taxonomy."""
    if not TAXONOMY_PATH.exists():
        return DEV_TAXONOMY

    try:
        with TAXONOMY_PATH.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load skill taxonomy: {exc}") from exc

    if isinstance(data, dict):
        data = data.get("skills", [])

    if not isinstance(data, list):
        raise ValueError(
            "Taxonomy must be a list or a dictionary containing 'skills'."
        )

    for entry in data:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError(
                "Every taxonomy entry must be an object containing 'name'."
            )

    return data


def _normalise_phrase(phrase: str) -> str:
    """Normalize whitespace and case for comparison."""
    return re.sub(r"\s+", " ", phrase.strip()).casefold()


def _entry_id(entry: dict[str, Any]) -> str:
    """Read the canonical ID, with a fallback for incomplete development data."""
    value = entry.get("id") or entry.get("skill_id")

    if value:
        return str(value)

    return re.sub(r"[^a-z0-9]+", "-", entry["name"].casefold()).strip("-")


def _phrase_from_alias(alias: Any) -> str:
    """Support string aliases and object aliases."""
    if isinstance(alias, str):
        return alias.strip()

    if isinstance(alias, dict):
        return str(alias.get("phrase") or alias.get("name") or "").strip()

    return ""


def _review_aliases(entry: dict[str, Any]) -> list[str]:
    aliases = []

    for alias in entry.get("review_aliases", []):
        phrase = _phrase_from_alias(alias)

        if phrase:
            aliases.append(phrase)

    return aliases

def _semantic_match(
    phrase: str,
    taxonomy: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float]:
    """Find the best semantic taxonomy match using cosine similarity."""
    global _SEMANTIC_MODEL

    if not phrase.strip() or not taxonomy or SentenceTransformer is None or cosine_similarity is None:
        return None, 0.0

    try:
        if _SEMANTIC_MODEL is None:
            _SEMANTIC_MODEL = SentenceTransformer(SEMANTIC_MODEL_NAME)

        names = [str(entry["name"]) for entry in taxonomy]
        cache_key = tuple(names)

        if cache_key not in _SEMANTIC_EMBEDDINGS:
            _SEMANTIC_EMBEDDINGS[cache_key] = _SEMANTIC_MODEL.encode(
                names,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

        phrase_embedding = _SEMANTIC_MODEL.encode(
            [phrase],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        similarities = cosine_similarity(
            phrase_embedding,
            _SEMANTIC_EMBEDDINGS[cache_key],
        )[0]

        best_index = int(similarities.argmax())
        best_score = float(similarities[best_index])

        if best_score >= SEMANTIC_THRESHOLD:
            return taxonomy[best_index], best_score

        return None, best_score

    except Exception:
        # Semantic matching must not break the existing normalizer.
        return None, 0.0
    
def _match_taxonomy(
    phrase: str,
    taxonomy: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str, float]:
    """Return the best safe taxonomy match and its mapping metadata."""
    target = _normalise_phrase(phrase)

    # Stage 1: exact canonical-name match.
    for entry in taxonomy:
        if _normalise_phrase(entry["name"]) == target:
            return entry, "exact", 1.0

    # Stage 2: regular alias match.
    for entry in taxonomy:
        aliases = [
            _phrase_from_alias(alias)
            for alias in entry.get("aliases", [])
        ]

        if any(alias and _normalise_phrase(alias) == target for alias in aliases):
            return entry, "alias", 0.98

    # Stage 3: review alias match.
    for entry in taxonomy:
        if any(
            _normalise_phrase(alias) == target
            for alias in _review_aliases(entry)
        ):
            return entry, "review_alias", 0.95

    # Stage 4: conservative fuzzy match.
    # Only canonical names are used. Fuzzy matches require human review.
    choices = {
        entry["name"]: entry
        for entry in taxonomy
    }

    match = process.extractOne(
        phrase.strip(),
        list(choices),
        scorer=fuzz.ratio,
        processor=lambda value: _normalise_phrase(value),
        score_cutoff=FUZZY_THRESHOLD,
    )

    if match:
        matched_name, score, _index = match
        return choices[matched_name], "fuzzy", float(score) / 100.0

    # Stage 5: optional semantic matching. Disabled unless explicitly opted in,
    # because downloading/loading a model can be slow and semantic guesses can
    # increase false positives in extraction evaluation.
    import os
    if os.getenv("SKILLLENS_ENABLE_SEMANTIC", "0").strip() == "1":
        semantic_entry, semantic_score = _semantic_match(phrase, taxonomy)
        if semantic_entry is not None:
            return semantic_entry, "semantic", semantic_score

    # Stage 6: preserve unknown candidates.
    return None, "unmapped", 0.0


def normalize_skills(mentions: list[Mention]) -> list[Skill]:
    """
    Convert Mention objects into Skill objects.

    This function deliberately leaves requirement classification to M4.
    """
    if not mentions:
        return []

    taxonomy = _load_taxonomy()
    normalized = []

    for mention in mentions:
        if not isinstance(mention, Mention):
            raise TypeError(
                "normalize_skills expects a list of Mention objects."
            )

        phrase = mention.original_phrase.strip()

        if not phrase:
            continue

        entry, mapping_status, score = _match_taxonomy(phrase, taxonomy)

        if entry is None:
            # Do not discard unknown candidates.
            skill_name = phrase
            skill_id = re.sub(
                r"[^a-z0-9]+",
                "-",
                phrase.casefold(),
            ).strip("-") or "unknown"
            category = "Technical"
            needs_review = True
            review_reason = "No canonical taxonomy match; human review required."
        else:
            skill_name = str(entry["name"])
            skill_id = _entry_id(entry)
            category = entry.get("category", "Technical")

            if category not in {"Technical", "Non-technical"}:
                category = "Technical"
                needs_review = True
                review_reason = "Invalid or missing taxonomy category."
            elif mapping_status in {"review_alias", "fuzzy", "semantic"}:
                needs_review = True
                review_reason = f"Mapping requires review: {mapping_status}."
            else:
                needs_review = mention.source == "inferred"
                review_reason = (
                    "Inferred skill requires review."
                    if needs_review
                    else ""
                )

        normalized.append(
            Skill(
                skill_id=skill_id,
                skill=skill_name,
                category=category,
                original_phrase=mention.original_phrase,
                evidence=mention.evidence,
                span_start=mention.span_start,
                span_end=mention.span_end,
                source=mention.source,
                requirement="Uncertain",
                requirement_basis="none",
                mapping_status=mapping_status,
                mapping_score=score,
                needs_review=needs_review,
                review_reason=review_reason,
                user_status="pending",
            )
        )

    return normalized
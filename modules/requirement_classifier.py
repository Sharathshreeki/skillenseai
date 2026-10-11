"""Conservative rule-based requirement classification for SkillLens AI."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from modules.schemas import RequirementResult, Skill

_DEFAULT_CUES = {
    "mandatory": [
        "must", "required", "require", "requires", "required to", "mandatory", "essential",
        "minimum", "need to", "needs to", "have to", "has to", "shall", "proficiency in",
        "expertise in", "prowess in", "experience with", "experience in", "hands-on experience with",
        "demonstrated experience in", "ability to use", "working knowledge of", "must have",
        "must know", "are expected to",
    ],
    "preferred": ["preferably", "preferred", "a plus", "advantage", "nice to have", "desirable", "bonus", "good to have", "ideally"],
    "vague": ["should", "strong knowledge of", "familiarity with", "helpful", "expected"],
    "negation": ["not required", "not a requirement", "no need", "not necessary", "not needed", "need not", "does not require", "do not require", "without requiring"],
    "role_action": [
        "build", "develop", "design", "implement", "create", "deploy", "train", "evaluate",
        "optimize", "optimise", "analyze", "analyse", "apply", "use", "work with", "leverage",
        "maintain", "manage", "execute", "perform", "deliver", "operate", "integrate", "automate",
    ],
    "strong_mandatory_headings": [
        "requirements", "required", "must have", "essential", "minimum qualifications",
        "required skills", "required qualifications", "required competencies", "core competencies",
    ],
    "strong_preferred_headings": ["preferred", "nice to have", "good to have", "bonus", "desirable", "preferred skills", "preferred qualifications"],
    "weak_headings": ["skills", "qualifications", "what you'll need", "about you", "responsibilities", "competencies", "competence", "duties"],
}


def _load_cues() -> dict[str, list[str]]:
    path = Path(__file__).resolve().parents[1] / "data" / "cue_phrases.json"
    try:
        with path.open(encoding="utf-8") as f:
            loaded = json.load(f)
        return {key: [str(x).lower() for x in loaded.get(key, default)] for key, default in _DEFAULT_CUES.items()}
    except (OSError, json.JSONDecodeError, TypeError):
        return _DEFAULT_CUES


def _split_clauses(text: str) -> list[tuple[int, int, str]]:
    """Split clauses while preserving wrapped lines inside a sentence.

    PDFs, OCR and copied job descriptions often insert a newline in the middle
    of a sentence (for example, "Proficiency in Python,\nSQL"). Treating every
    newline as a clause boundary loses requirement cues for skills on the next
    line. We split on punctuation, contrast words, blank lines and explicit
    list-item boundaries, while keeping character offsets intact.
    """
    parts: list[tuple[int, int, str]] = []
    pattern = re.compile(
        r"\b(?:but|however|although|while)\b"
        r"|[;.!?]+"
        r"|\n[ \t]*\n+"
        r"|\n(?=[ \t]*(?:[-*+•]\s+|\d+[.)]\s+|#{1,6}\s+))",
        re.IGNORECASE,
    )
    start = 0
    for match in pattern.finditer(text):
        end = match.start()
        if text[start:end].strip():
            parts.append((start, end, text[start:end]))
        start = match.end()
    if text[start:].strip():
        parts.append((start, len(text), text[start:]))
    return parts or [(0, len(text), text)]


def _heading_for(text: str, position: int) -> str | None:
    """Find the nearest section heading before a skill mention.

    Markdown headings are common in job descriptions. Ignore intervening bullet
    items instead of treating their trailing punctuation as a section boundary.
    """
    before = text[:position]
    lines = before.splitlines()
    for line in reversed(lines[-40:]):
        stripped = line.strip()
        if not stripped:
            continue

        # Markdown headings such as "## Required Technical Skills".
        if stripped.startswith("#"):
            heading_text = stripped.lstrip("#").strip()
            if heading_text:
                return heading_text

        # Ignore intervening list items; they belong to the current section.
        if re.match(r"^(?:[-*+•]|\d+[.)])(?:\s+|$)", stripped):
            continue

        if stripped.endswith(":"):
            return stripped[:-1].strip()
        # Recognize inline headings such as "Preferred: Python".
        match = re.match(r"^([A-Za-z][A-Za-z /_-]{1,45}):\s*", stripped)
        if match:
            return match.group(1).strip()

        # Ordinary prose means the skill is not directly under a nearby heading.
        if re.search(r"[.!?]", stripped):
            break
    return None


def _contains_cue(text: str, cue: str) -> bool:
    """Match cue phrases on word boundaries so e.g. 'required' isn't a substring accident."""
    return bool(re.search(r"(?<!\w)" + re.escape(cue) + r"(?!\w)", text, re.I))


def _classify_clause(clause: str, heading: str | None, cues: dict[str, list[str]]) -> tuple[str, str, bool, str]:
    low = re.sub(r"\s+", " ", clause.lower()).strip()

    # Negation wins over positive wording: "Python is not required" is not mandatory.
    negation_cues = cues.get("negation", _DEFAULT_CUES["negation"])
    if any(_contains_cue(low, cue) for cue in negation_cues):
        return "Uncertain", "negation:not_required", True, "Requirement is negated or stated as not required."

    mandatory_cues = cues.get("mandatory", _DEFAULT_CUES["mandatory"])
    preferred_cues = cues.get("preferred", _DEFAULT_CUES["preferred"])
    action_cues = cues.get("role_action", _DEFAULT_CUES["role_action"])
    explicit_mandatory = next((c for c in mandatory_cues if _contains_cue(low, c)), None)
    action_cue = next((c for c in action_cues if _contains_cue(low, c)), None)
    has_mandatory = explicit_mandatory is not None or action_cue is not None
    preferred_cue = next((c for c in preferred_cues if _contains_cue(low, c)), None)
    has_preferred = preferred_cue is not None

    if has_mandatory and has_preferred:
        return "Uncertain", "conflict", True, "Both mandatory and preferred cues appear in the same clause."
    if has_preferred:
        return "Preferred", f"cue:{preferred_cue}", False, ""
    if explicit_mandatory:
        return "Mandatory", f"cue:{explicit_mandatory}", False, ""
    if action_cue:
        return "Mandatory", f"role_action:{action_cue}", False, "Skill is named in an explicit job-duty action."

    vague = next((c for c in cues.get("vague", _DEFAULT_CUES["vague"]) if _contains_cue(low, c)), None)
    if vague:
        return "Preferred", "cue:vague", False, f"Vague requirement cue: {vague}; defaulted to Preferred."

    if heading:
        h = re.sub(r"\s+", " ", heading.lower().strip())
        mandatory_headings = cues.get("strong_mandatory_headings", _DEFAULT_CUES["strong_mandatory_headings"])
        preferred_headings = cues.get("strong_preferred_headings", _DEFAULT_CUES["strong_preferred_headings"])
        if any(h == x.lower() or h.startswith(x.lower() + " ") for x in mandatory_headings):
            return "Mandatory", f"section:{heading}", False, ""
        if any(h == x.lower() or h.startswith(x.lower() + " ") for x in preferred_headings):
            return "Preferred", f"section:{heading}", False, ""
        if any(x.lower() in h for x in cues.get("weak_headings", _DEFAULT_CUES["weak_headings"])):
            return "Uncertain", f"section:{heading}", True, "Weak section heading does not establish requirement strength."
    return "Preferred", "default:preferred", False, "Defaulted to Preferred because no clear requirement cue was found."


def classify_requirements(text: str, skills: list[Skill] | None) -> RequirementResult:
    """Classify skills using the clause at each skill's character span.

    Spans are Python string offsets: start-inclusive, end-exclusive. Skills with
    insufficient evidence remain Uncertain and are flagged for human review.
    """
    if not isinstance(text, str) or not text.strip():
        return RequirementResult(skills=list(skills or []), warnings=["No readable job description text provided."])
    cues = _load_cues()
    clauses = _split_clauses(text)
    classified: list[Skill] = []
    for original in skills or []:
        # Accept dataclass Skills; preserve the same objects' metadata via a shallow copy.
        import copy
        skill = copy.copy(original)
        start = max(0, int(getattr(skill, "span_start", 0) or 0))
        end = max(start, int(getattr(skill, "span_end", start) or start))
        if end > start and start < len(text):
            probe = start
        else:
            phrase = getattr(skill, "original_phrase", "") or getattr(skill, "skill", "")
            found = text.lower().find(str(phrase).lower()) if phrase else -1
            probe = found if found >= 0 else 0
        clause = next((part for a, b, part in clauses if a <= probe <= b), text)
        heading = _heading_for(text, probe)
        label, basis, needs_review, reason = _classify_clause(clause, heading, cues)
        # Inferred skills are always reviewable; they must not be treated as explicit evidence.
        if getattr(skill, "source", "explicit") == "inferred":
            needs_review = True
            reason = reason or "Inferred skill requires human review."
        skill.requirement = label
        skill.requirement_basis = basis
        skill.needs_review = bool(needs_review or getattr(skill, "needs_review", False))
        skill.review_reason = reason
        classified.append(skill)
    experience = extract_experience(text)
    education = extract_education(text)
    certifications = extract_certifications(text)
    warnings: list[str] = []
    if experience.get("needs_review"):
        warnings.append("Experience requirement needs review.")
    if education.get("needs_review"):
        warnings.append("Education requirement needs review.")
    return RequirementResult(skills=classified, experience=experience, education=education,
                             certifications=certifications, warnings=warnings)


_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _number_value(value: str) -> int | None:
    value = value.lower()
    if value.isdigit():
        return int(value)
    return _NUMBER_WORDS.get(value)


def extract_experience(text: str) -> dict[str, Any]:
    low = text.lower()
    if "several years" in low or "extensive experience" in low:
        return {"min_years": None, "max_years": None, "evidence": "several/extensive experience", "needs_review": True}
    if re.search(r"\b(fresher|entry[- ]level)\b", low):
        match = re.search(r"\b(fresher|entry[- ]level)\b", low)
        return {"min_years": 0, "max_years": 0, "evidence": match.group(0), "needs_review": True}
    number = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    patterns = [
        (re.compile(r"\b(?:at least|minimum|min\.? of)\s+" + number + r"\s+years?\b", re.I), "min"),
        (re.compile(r"\b" + number + r"\s*[-–]\s*(\d+)\s+years?\b", re.I), "range"),
        (re.compile(r"\b" + number + r"\s+years?\s+(?:of\s+)?experience\b", re.I), "single"),
    ]
    for pattern, kind in patterns:
        match = pattern.search(text)
        if match:
            if kind == "range":
                return {"min_years": _number_value(match.group(1)), "max_years": int(match.group(2)), "evidence": match.group(0), "needs_review": False}
            n = _number_value(match.group(1))
            return {"min_years": n, "max_years": None if kind == "min" else n, "evidence": match.group(0), "needs_review": False}
    return {"min_years": None, "max_years": None, "evidence": "", "needs_review": True}


def extract_education(text: str) -> dict[str, Any]:
    patterns = [
        (r"\b(?:bachelor'?s?|b\.?e\.?|b\.?tech|b\.?s\.?c?\.?|bsc)\b", "Bachelor's"),
        (r"\b(?:master'?s?|m\.?tech|m\.?s\.?c?\.?|msc|mba)\b", "Master's"),
        (r"\bph\.?d\.?\b", "PhD"),
        (r"\bdiploma\b", "Diploma"),
    ]
    for pattern, level in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            context = text[max(0, match.start()-50):min(len(text), match.end()+60)]
            equivalent = bool(re.search(r"\bor equivalent\b", context, re.I))
            relevant = bool(re.search(r"\brelevant degree\b", context, re.I))
            return {"level": level, "equivalent_allowed": equivalent, "field": "relevant" if relevant else "", "evidence": context.strip(), "needs_review": equivalent or relevant}
    if re.search(r"\brelevant degree\b", text, re.I):
        match = re.search(r"\brelevant degree\b", text, re.I)
        return {"level": "", "equivalent_allowed": bool(re.search(r"\bor equivalent\b", text[match.end():match.end()+40], re.I)), "field": "relevant", "evidence": match.group(0), "needs_review": True}
    return {"level": "", "equivalent_allowed": False, "field": "", "evidence": "", "needs_review": True}



def extract_certifications(text: str) -> list[dict[str, Any]]:
    """Extract certifications and classify their requirement strength."""
    if not isinstance(text, str) or not text.strip():
        return []

    found = []
    cues = _load_cues()

    pattern = re.compile(
        r"\b([A-Z][A-Za-z0-9+.# -]{1,50}?\b"
        r"(?:certified|certification|certificate))\b",
        re.IGNORECASE,
    )

    conflict_pattern = re.compile(
        r"\b(?:must|required|mandatory|essential)\b.*"
        r"\b(?:preferred|preferably|nice to have|desirable)\b"
        r"|"
        r"\b(?:preferred|preferably|nice to have|desirable)\b.*"
        r"\b(?:must|required|mandatory|essential)\b",
        re.IGNORECASE,
    )

    clauses = _split_clauses(text)

    for match in pattern.finditer(text):
        start, end = match.span()

        clause = next(
            (part for a, b, part in clauses if a <= start < b),
            text,
        )

        heading = _heading_for(text, start)

        # Check the full sentence so that "preferred but mandatory"
        # is recognised as a conflict even if "but" splits the clauses.
        sentence_match = re.search(
            r"[^.!?\n]*" + re.escape(match.group(0)) + r"[^.!?\n]*",
            text,
            re.IGNORECASE,
        )
        sentence = sentence_match.group(0) if sentence_match else clause

        if conflict_pattern.search(sentence):
            requirement = "Uncertain"
            basis = "conflicting_cues"
            needs_review = True
            reason = "Conflicting mandatory and preferred cues"
            evidence = sentence.strip()
        else:
            requirement, basis, needs_review, reason = _classify_clause(
                clause, heading, cues
            )
            evidence = clause.strip()

        found.append({
            "name": match.group(1).strip(),
            "requirement": requirement,
            "requirement_basis": basis,
            "needs_review": needs_review,
            "review_reason": reason,
            "evidence": evidence,
        })

    return found




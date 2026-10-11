"""Pipeline entry points used by the Streamlit UI.

Member 3's extractor/normalizer and Member 2's document/comparison modules are
optional until those files are integrated. The pipeline stays runnable meanwhile.
"""
from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from modules.schemas import ExtractionResult, JobProfile, Mention, Skill
from modules.requirement_classifier import classify_requirements


def warm_up():
    """Warm optional models if present; no network access is triggered here."""
    return None


def load_file(file) -> ExtractionResult:
    """Load text from an uploaded file using Member 2's extractor when available."""
    try:
        from modules.document_extractor import extract_text  # type: ignore
    except ImportError as exc:
        raise NotImplementedError("Document extraction module is not integrated yet. Paste job text for now.") from exc
    result = extract_text(file)
    if isinstance(result, ExtractionResult):
        if not result.text.strip():
            raise ValueError("No readable text found")
        return result
    # Allow a compatible extractor to return a string during early integration.
    if isinstance(result, str):
        if not result.strip():
            raise ValueError("No readable text found")
        return ExtractionResult(text=result, method="txt")
    raise TypeError("extract_text(file) must return ExtractionResult or str")


def _fallback_mentions(text: str) -> list[Mention]:
    """Small demo fallback only; the full extractor should be supplied by Member 3."""
    candidates = [
        ("JavaScript", [r"\bjavascript\b", r"\bjs\b"]), ("SQL", [r"\bsql\b", r"structured query language"]),
        ("Excel", [r"\bexcel\b", r"spreadsheet analysis"]), ("Python", [r"\bpython\b"]),
        ("Power BI", [r"\bpower\s*bi\b"]), ("Tableau", [r"\btableau\b"]),
        ("Google Sheets", [r"\bgoogle sheets\b"]), ("REST API", [r"\brest(?:ful)?\s+api\b", r"\brestful services\b"]),
        ("R programming", [r"\br programming\b"]), ("C++", [r"\bc\+\+\b"]), ("C#", [r"\bc#\b"]),
        ("Node.js", [r"\bnode\.js\b"]), ("Java", [r"\bjava\b"]),
    ]
    mentions: list[Mention] = []
    occupied: list[tuple[int, int]] = []
    for canonical, patterns in candidates:
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                if any(match.start() < b and match.end() > a for a, b in occupied):
                    continue
                evidence_start = text.rfind("\n", 0, match.start()) + 1
                evidence_end_candidates = [p for p in (text.find("\n", match.end()), text.find(".", match.end()), text.find(";", match.end())) if p >= 0]
                evidence_end = min(evidence_end_candidates) + 1 if evidence_end_candidates else len(text)
                evidence = text[evidence_start:evidence_end].strip()
                phrase = match.group(0)
                mentions.append(Mention(original_phrase=phrase, evidence=evidence,
                                        span_start=match.start(), span_end=match.end(),
                                        source="explicit", matched_via="fallback_rule"))
                occupied.append((match.start(), match.end()))
    return sorted(mentions, key=lambda m: m.span_start)


def _normalize(mentions: list[Mention]) -> list[Skill]:
    """Use Member 3's normalizer when available; otherwise make reviewable demo skills."""
    try:
        from modules.skill_normalizer import normalize_skills  # type: ignore
        result = normalize_skills(mentions)
        if result is not None:
            return list(result)
    except ImportError:
        pass
    canonical = {"javascript": ("javascript", "JavaScript"), "js": ("javascript", "JavaScript"),
                 "sql": ("sql", "SQL"), "excel": ("excel", "Excel"), "spreadsheet analysis": ("excel", "Excel"),
                 "python": ("python", "Python"), "power bi": ("power_bi", "Power BI"),
                 "tableau": ("tableau", "Tableau"), "google sheets": ("google_sheets", "Google Sheets"),
                 "rest api": ("rest_api", "REST API"), "restful api": ("rest_api", "REST API"),
                 "restful services": ("rest_api", "REST API"), "r programming": ("r_programming", "R programming"),
                 "c++": ("cpp", "C++"), "c#": ("csharp", "C#"), "node.js": ("node_js", "Node.js"),
                 "java": ("java", "Java")}
    skills: list[Skill] = []
    for mention in mentions:
        key = mention.original_phrase.strip().lower()
        skill_id, name = canonical.get(key, ("unmapped:" + key, mention.original_phrase))
        mapped = not skill_id.startswith("unmapped:")
        skills.append(Skill(skill_id=skill_id, skill=name, original_phrase=mention.original_phrase,
                            evidence=mention.evidence, span_start=mention.span_start, span_end=mention.span_end,
                            source=mention.source, requirement="Uncertain", requirement_basis="none",
                            mapping_status="alias" if mapped and key != name.lower() else ("exact" if mapped else "unmapped"),
                            mapping_score=1.0 if mapped else 0.0, needs_review=not mapped,
                            review_reason="Unknown skill phrase; review mapping." if not mapped else ""))
    return skills


def _deduplicate(skills: list[Skill]) -> list[Skill]:
    """Merge repeated skill mentions without treating missing evidence as a conflict.

    One mention may occur in a required section while another occurs in a neutral
    sentence. An Uncertain label from the neutral mention should not erase clear
    evidence from the required mention. Genuine contradictions (Mandatory vs
    Preferred, or explicit negation/conflict) remain Uncertain for review.
    """
    by_id: dict[str, Skill] = {}
    hard_uncertainty_prefixes = ("conflict", "negation:")
    for skill in skills:
        existing = by_id.get(skill.skill_id)
        if existing is None:
            by_id[skill.skill_id] = skill
            continue

        existing_label = getattr(existing, "requirement", "Uncertain")
        new_label = getattr(skill, "requirement", "Uncertain")
        existing_basis = str(getattr(existing, "requirement_basis", "none") or "none")
        new_basis = str(getattr(skill, "requirement_basis", "none") or "none")

        if existing_label != new_label:
            determinate = {"Mandatory", "Preferred"}
            if existing_label in determinate and new_label in determinate:
                # The same skill is explicitly described as both mandatory and preferred.
                existing.requirement = "Uncertain"
                existing.requirement_basis = "conflict"
                existing.needs_review = True
                existing.review_reason = "Conflicting requirement labels for the same skill."
            elif existing_label == "Uncertain" and new_label in determinate:
                if not existing_basis.startswith(hard_uncertainty_prefixes):
                    existing.requirement = new_label
                    existing.requirement_basis = new_basis
                    # Keep mapping/inference review flags, but remove only the stale
                    # classifier reason caused by missing local requirement evidence.
                    if "No clear requirement cue" in str(getattr(existing, "review_reason", "")) or not existing.review_reason:
                        existing.review_reason = getattr(skill, "review_reason", "")
                    existing.needs_review = bool(
                        getattr(existing, "source", "explicit") == "inferred"
                        or getattr(existing, "mapping_status", "") in {"unmapped", "low_confidence"}
                        or getattr(skill, "needs_review", False)
                    )
            elif new_label == "Uncertain" and existing_label in determinate:
                if new_basis.startswith(hard_uncertainty_prefixes):
                    existing.requirement = "Uncertain"
                    existing.requirement_basis = "conflict"
                    existing.needs_review = True
                    existing.review_reason = "Conflicting requirement evidence for the same skill."
                # Otherwise retain the determinate label from the stronger mention.
            elif existing_label == "Uncertain" and new_label == "Uncertain":
                if new_basis.startswith(hard_uncertainty_prefixes):
                    existing.requirement_basis = new_basis
                    existing.needs_review = True
                    existing.review_reason = getattr(skill, "review_reason", "")

        if skill.evidence and skill.evidence not in str(getattr(existing, "evidence", "")):
            existing.evidence = (str(getattr(existing, "evidence", "")) + " | " + skill.evidence).strip(" |")
    return list(by_id.values())


def analyze_job(text: str, job_id: str, title: str | None = None) -> JobProfile:
    """Extract → normalize → deduplicate → classify and return a JobProfile."""
    warnings: list[str] = []
    if not isinstance(text, str) or not text.strip():
        return JobProfile(job_id=job_id, title=title, warnings=["No readable job description text provided."])
    try:
        from modules.skill_extractor import extract_skills  # type: ignore
        mentions = list(extract_skills(text))
    except ImportError:
        mentions = _fallback_mentions(text)
        warnings.append("Member 3 extractor not integrated; using limited demo fallback rules.")
    skills = _normalize(mentions)
    result = classify_requirements(text, skills)
    skills = _deduplicate(result.skills)
    # Classification should be recomputed on the deduplicated set to preserve conflict handling.
    profile = JobProfile(job_id=job_id, title=title, skills=skills,
                         experience=result.experience, education=result.education,
                         certifications=result.certifications,
                         warnings=warnings + result.warnings)
    if not skills:
        profile.warnings.append("No skills detected; verify the job description or connect the full extractor.")
    return profile


def profile_to_dict(profile: JobProfile) -> dict[str, Any]:
    """Convert a profile to a JSON-friendly dictionary."""
    return asdict(profile)

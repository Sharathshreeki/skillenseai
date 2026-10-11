"""Tests for SkillLens M3 skill extraction and normalization."""

from modules.skill_extractor import extract_skills
from modules.skill_normalizer import normalize_skills


def test_extract_python_and_sql():
    mentions = extract_skills("We require Python and SQL experience.")

    phrases = [mention.original_phrase for mention in mentions]

    assert phrases == ["Python", "SQL"]


def test_javascript_and_java_are_separate():
    mentions = extract_skills("We use JavaScript and Java.")

    phrases = [mention.original_phrase for mention in mentions]

    assert phrases == ["JavaScript", "Java"]


def test_extract_aliases():
    mentions = extract_skills("Experience with JS and RESTful API is required.")

    phrases = [mention.original_phrase for mention in mentions]

    assert "JS" in phrases
    assert "RESTful API" in phrases


def test_normalize_aliases():
    mentions = extract_skills("We need JS and Excel.")
    skills = normalize_skills(mentions)

    normalized = {skill.original_phrase: skill for skill in skills}

    assert normalized["JS"].skill_id == "javascript"
    assert normalized["Excel"].skill_id == "excel_spreadsheets"


def test_evidence_and_character_spans():
    text = "Candidates must know Python."
    mentions = extract_skills(text)

    python_mention = next(
        mention for mention in mentions
        if mention.original_phrase == "Python"
    )

    assert text[python_mention.span_start:python_mention.span_end] == "Python"
    assert "Python" in python_mention.evidence


def test_normalized_skills_preserve_evidence():
    mentions = extract_skills("Experience with Python is required.")
    skills = normalize_skills(mentions)

    assert len(skills) == 1
    assert skills[0].original_phrase == "Python"
    assert "Python" in skills[0].evidence
    assert skills[0].requirement == "Uncertain"


def test_empty_input():
    assert extract_skills("") == []
    assert extract_skills("   ") == []


def test_unknown_skill_is_not_invented():
    mentions = extract_skills("Experience with Python and Kubernetes.")
    skills = normalize_skills(mentions)

    phrases = [skill.original_phrase for skill in skills]

    # Kubernetes is now included in the expanded canonical taxonomy.
    assert "Python" in phrases
    assert "Kubernetes" in phrases


def test_extraction_preserves_original_text():
    mentions = extract_skills("Experience with PYTHON is required.")

    assert len(mentions) == 1
    assert mentions[0].original_phrase == "PYTHON"
    
def test_inferred_skills_require_review():
    from modules.schemas import Mention
    from modules.skill_normalizer import normalize_skills

    mention = Mention(
        original_phrase="Python",
        evidence="Experience with Python",
        source="inferred",
    )

    result = normalize_skills([mention])

    assert len(result) == 1
    assert result[0].skill == "Python"
    assert result[0].source == "inferred"
    assert result[0].needs_review is True
    assert result[0].review_reason == "Inferred skill requires review."
    
def test_review_alias_requires_human_review():
    from modules.skill_extractor import extract_skills
    from modules.skill_normalizer import normalize_skills

    mentions = extract_skills(
        "We need spreadsheet analysis experience."
    )
    skills = normalize_skills(mentions)

    assert len(skills) == 1
    assert skills[0].skill == "Excel / Spreadsheets"
    assert skills[0].mapping_status == "review_alias"
    assert skills[0].needs_review is True
    assert "review_alias" in skills[0].review_reason
    
def test_semantic_matching_is_disabled_by_default():
    from modules.skill_normalizer import _match_taxonomy, _load_taxonomy

    # Extraction is deterministic/offline by default; semantic matching is an
    # explicit opt-in because it can add false positives and load a model.
    result = _match_taxonomy("Python programming language", _load_taxonomy())
    assert result[1] == "unmapped"
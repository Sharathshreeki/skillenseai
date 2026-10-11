from modules.requirement_classifier import classify_requirements, extract_education, extract_experience
from modules.requirement_classifier import extract_certifications
from modules.schemas import Skill


def skill(name, start, end, skill_id=None):
    return Skill(skill_id=skill_id or name.lower(), skill=name, original_phrase=name,
                 span_start=start, span_end=end, evidence=name)


def test_preferred_and_mandatory_in_separate_clauses():
    text = "Python is preferred, but SQL must be strong."
    python = skill("Python", text.index("Python"), text.index("Python") + 6)
    sql = skill("SQL", text.index("SQL"), text.index("SQL") + 3)
    result = classify_requirements(text, [python, sql])
    labels = {s.skill: s.requirement for s in result.skills}
    assert labels == {"Python": "Preferred", "SQL": "Mandatory"}


def test_negation_is_uncertain():
    text = "Python is not required."
    item = skill("Python", 0, 6)
    result = classify_requirements(text, [item])
    assert result.skills[0].requirement == "Uncertain"
    assert result.skills[0].requirement_basis == "negation:not_required"
    assert result.skills[0].needs_review


def test_conflicting_cues_are_uncertain():
    text = "Preferably must know SQL."
    item = skill("SQL", text.index("SQL"), text.index("SQL") + 3)
    result = classify_requirements(text, [item])
    assert result.skills[0].requirement == "Uncertain"
    assert result.skills[0].requirement_basis == "conflict"


def test_weak_heading_does_not_force_mandatory():
    text = "Skills:\nPython and SQL"
    item = skill("Python", text.index("Python"), text.index("Python") + 6)
    result = classify_requirements(text, [item])
    assert result.skills[0].requirement == "Uncertain"
    assert result.skills[0].needs_review


def test_experience_and_education_extraction():
    experience = extract_experience("Minimum 2 years of experience required.")
    education = extract_education("Bachelor's degree or equivalent required.")
    assert experience["min_years"] == 2
    assert education["level"] == "Bachelor's"
    assert education["equivalent_allowed"] is True
    assert education["needs_review"] is True


def test_empty_text_returns_warning():
    result = classify_requirements("", [])
    assert result.warnings


from modules.requirement_classifier import (
    extract_experience,
    extract_education,
    extract_certifications,
)


def test_experience_range_and_number_words():
    result = extract_experience("At least two years of experience")
    assert result["min_years"] == 2
    assert result["max_years"] is None
    assert result["needs_review"] is False

    result = extract_experience("2-4 years of experience")
    assert result["min_years"] == 2
    assert result["max_years"] == 4


def test_vague_experience_requires_review():
    result = extract_experience("Several years of experience")
    assert result["min_years"] is None
    assert result["max_years"] is None
    assert result["needs_review"] is True
    assert result["evidence"]


def test_fresher_experience_requires_review():
    result = extract_experience("Fresher position")
    assert result["min_years"] == 0
    assert result["needs_review"] is True


def test_equivalent_education_requires_review():
    result = extract_education("B.Tech or equivalent")
    assert result["level"] == "Bachelor's"
    assert result["equivalent_allowed"] is True
    assert result["needs_review"] is True


def test_relevant_degree_requires_review():
    result = extract_education("Relevant degree or equivalent")
    assert result["field"] == "relevant"
    assert result["equivalent_allowed"] is True
    assert result["needs_review"] is True


def test_certification_extraction_preserves_evidence():
    results = extract_certifications("AWS certification required")
    assert results
    assert results[0]["name"] == "AWS certification"
    assert results[0]["evidence"]


def test_mandatory_certification_is_classified():
    results = extract_certifications("AWS certification is required.")
    assert results
    assert results[0]["requirement"] == "Mandatory"


def test_preferred_certification_is_classified():
    results = extract_certifications("PMP certification is preferred.")
    assert results
    assert results[0]["requirement"] == "Preferred"


def test_negated_certification_is_uncertain():
    results = extract_certifications("AWS certification is not required.")
    assert results
    assert results[0]["requirement"] == "Uncertain"


def test_conflicting_certification_cues_are_uncertain():
    results = extract_certifications(
        "AWS certification is preferred but mandatory."
    )
    assert results
    assert results[0]["requirement"] == "Uncertain"


def test_certification_without_clear_cue_is_uncertain():
    results = extract_certifications("AWS certification.")
    assert results
    assert results[0]["requirement"] == "Uncertain"



def test_requirement_verbs_classify_all_skills_as_mandatory():
    text = "We require Python, SQL, and Excel. Candidates must have communication skills."
    skills = [
        skill("Python", text.index("Python"), text.index("Python") + 6),
        skill("SQL", text.index("SQL"), text.index("SQL") + 3),
        skill("Excel", text.index("Excel"), text.index("Excel") + 5),
        skill("communication", text.index("communication"), text.index("communication") + 13),
    ]
    result = classify_requirements(text, skills)
    assert {item.skill: item.requirement for item in result.skills} == {
        "Python": "Mandatory",
        "SQL": "Mandatory",
        "Excel": "Mandatory",
        "communication": "Mandatory",
    }


def test_negation_overrides_positive_required_cue():
    text = "Python is not required."
    item = skill("Python", text.index("Python"), text.index("Python") + 6)
    result = classify_requirements(text, [item])
    assert result.skills[0].requirement == "Uncertain"
    assert result.skills[0].requirement_basis == "negation:not_required"


def test_required_skills_heading_classifies_bullets_as_mandatory():
    text = "Required Skills:\\n- Python\\n- SQL"
    skills = [
        skill("Python", text.index("Python"), text.index("Python") + 6),
        skill("SQL", text.index("SQL"), text.index("SQL") + 3),
    ]
    result = classify_requirements(text, skills)
    assert all(item.requirement == "Mandatory" for item in result.skills)


def test_wrapped_line_keeps_requirement_cue_for_later_skills():
    text = (
        "Proficiency in machine learning concepts and programming languages such as Python,\n"
        "Pyspark, SQL, amongst others."
    )
    skills = [
        skill("Python", text.index("Python"), text.index("Python") + len("Python")),
        skill("SQL", text.index("SQL"), text.index("SQL") + len("SQL")),
    ]
    result = classify_requirements(text, skills)
    assert {item.skill: item.requirement for item in result.skills} == {
        "Python": "Mandatory",
        "SQL": "Mandatory",
    }


def test_core_job_action_indicates_expected_skill():
    text = (
        "Build predictive models, recommendation engines, anomaly detection systems, "
        "statistical models, deep learning models, and other machine learning systems."
    )
    skills = [
        skill("predictive models", text.index("predictive models"), text.index("predictive models") + len("predictive models")),
        skill("deep learning models", text.index("deep learning models"), text.index("deep learning models") + len("deep learning models")),
    ]
    result = classify_requirements(text, skills)
    assert all(item.requirement == "Mandatory" for item in result.skills)
    assert all(item.requirement_basis.startswith("role_action:") for item in result.skills)


def test_cue_matching_respects_word_boundaries():
    from modules.requirement_classifier import _contains_cue

    assert _contains_cue("This skill is required.", "required")
    assert not _contains_cue("This skill is unrequired.", "required")
    assert not _contains_cue("mustard is a seed", "must")


def test_explicit_preferred_cue_still_wins_without_conflicting_action():
    text = "Python is preferred."
    item = skill("Python", text.index("Python"), text.index("Python") + len("Python"))
    result = classify_requirements(text, [item])
    assert result.skills[0].requirement == "Preferred"

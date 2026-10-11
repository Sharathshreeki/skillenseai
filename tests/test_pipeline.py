from modules.pipeline import analyze_job, profile_to_dict

from dataclasses import asdict

from modules.schemas import JobProfile

def test_empty_job_returns_warning():
    profile = analyze_job("   ", "job_test", "Test Job")
    assert profile.job_id == "job_test"
    assert profile.warnings


def test_pipeline_extracts_and_classifies_demo_job():
    text = "Requirements: SQL and Excel. Python is preferred. Minimum 2 years of experience. Bachelor's degree required."
    profile = analyze_job(text, "job_a", "Data Analyst")
    labels = {skill.skill: skill.requirement for skill in profile.skills}
    assert labels["SQL"] == "Mandatory"
    assert labels["Excel / Spreadsheets"] == "Mandatory"
    assert labels["Python"] == "Preferred"
    assert profile.experience["min_years"] == 2
    assert profile.education["level"] == "Bachelor's"
    assert profile_to_dict(profile)["job_id"] == "job_a"


def test_unknown_phrase_is_not_crashing():
    profile = analyze_job("Experience with graph database tuning is preferred.", "job_unknown")
    assert profile.job_id == "job_unknown"
    assert isinstance(profile.skills, list)




def test_pipeline_returns_job_profile_contract():
    profile = analyze_job(
        "We need Python and SQL skills. A bachelor's degree is preferred.",
        job_id="contract-001",
        title="Data Analyst",
    )

    assert isinstance(profile, JobProfile)

    result = asdict(profile)
    assert set(result) == {
        "job_id",
        "title",
        "skills",
        "experience",
        "education",
        "certifications",
        "warnings",
    }

    assert result["job_id"] == "contract-001"
    assert result["title"] == "Data Analyst"


def test_pipeline_skills_follow_contract():
    profile = analyze_job(
        "Python is required. SQL is preferred.",
        job_id="contract-002",
    )

    valid_requirements = {"Mandatory", "Preferred", "Uncertain"}
    valid_categories = {"Technical", "Non-technical"}
    valid_sources = {"explicit", "inferred"}
    valid_mapping_statuses = {
        "exact",
        "alias",
        "review_alias",
        "fuzzy",
        "semantic",
        "unmapped",
    }
    valid_user_statuses = {"pending", "approved", "rejected", "edited"}

    for skill in profile.skills:
        assert skill.requirement in valid_requirements
        assert skill.category in valid_categories
        assert skill.source in valid_sources
        assert skill.mapping_status in valid_mapping_statuses
        assert skill.user_status in valid_user_statuses
        assert 0.0 <= skill.mapping_score <= 1.0

        if skill.source == "inferred":
            assert skill.needs_review is True


def test_deduplicate_keeps_determinate_evidence_over_neutral_mention():
    from modules.pipeline import _deduplicate
    from modules.schemas import Skill

    neutral = Skill(
        skill_id="python", skill="Python", original_phrase="Python",
        requirement="Uncertain", requirement_basis="none", needs_review=True,
        review_reason="No clear requirement cue was found.", mapping_status="exact", mapping_score=1.0,
    )
    required = Skill(
        skill_id="python", skill="Python", original_phrase="Python",
        requirement="Mandatory", requirement_basis="cue:required", needs_review=False,
        review_reason="", mapping_status="exact", mapping_score=1.0,
    )

    result = _deduplicate([neutral, required])
    assert len(result) == 1
    assert result[0].requirement == "Mandatory"
    assert result[0].requirement_basis == "cue:required"


def test_deduplicate_preserves_true_mandatory_preferred_conflict():
    from modules.pipeline import _deduplicate
    from modules.schemas import Skill

    mandatory = Skill(
        skill_id="python", skill="Python", requirement="Mandatory",
        requirement_basis="cue:required", mapping_status="exact", mapping_score=1.0,
        needs_review=False,
    )
    preferred = Skill(
        skill_id="python", skill="Python", requirement="Preferred",
        requirement_basis="cue:preferred", mapping_status="exact", mapping_score=1.0,
        needs_review=False,
    )

    result = _deduplicate([mandatory, preferred])
    assert result[0].requirement == "Uncertain"
    assert result[0].requirement_basis == "conflict"
    assert result[0].needs_review is True

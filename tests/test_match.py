import pytest

import src.match as match

PY_JD = "Python backend developer: Flask, PostgreSQL, SQL, Docker and Git required."
PY_CV = "Python developer. Built Flask apps with PostgreSQL, SQL, Docker and Git. " * 5
MKT_CV = "Marketing manager with SEO, social media and email marketing experience. " * 5


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Force the TF-IDF fallback so tests need no model download or internet."""
    monkeypatch.setattr(match, "_embedder", lambda: None)


def test_skill_gap_full_match():
    matched, missing, ratio = match.skill_gap(PY_CV, PY_JD)
    assert missing == [] and ratio == 1.0 and "flask" in matched


def test_skill_gap_reports_missing():
    matched, missing, ratio = match.skill_gap("Python only", PY_JD)
    assert "python" in matched and "docker" in missing and 0 < ratio < 1


def test_skill_gap_no_required_skills_in_jd():
    assert match.skill_gap(PY_CV, "We value teamwork and curiosity.")[2] == 0.0


def test_match_score_fields_and_range():
    r = match.match_score(PY_CV, PY_JD)
    assert set(r) == {"score", "semantic", "skill_overlap", "matched", "missing"}
    assert 0 <= r["score"] <= 100


def test_relevant_resume_beats_irrelevant_one():
    good = match.match_score(PY_CV, PY_JD)["score"]
    bad = match.match_score(MKT_CV, PY_JD)["score"]
    assert good > bad


def test_semantic_similarity_identical_text_is_high():
    assert match.semantic_similarity(PY_JD, PY_JD) > 0.99

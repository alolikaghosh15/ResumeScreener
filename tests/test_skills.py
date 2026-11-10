from src.skills import extract_skills


def test_detects_basic_skills():
    found = extract_skills("Built REST APIs in Python with Flask and Docker")
    assert {"python", "flask", "docker", "rest api"} <= found


def test_plural_form_matches():
    assert "rest api" in extract_skills("We build REST APIs")


def test_java_not_matched_inside_javascript():
    found = extract_skills("Experienced in JavaScript")
    assert "javascript" in found and "java" not in found


def test_sql_not_matched_inside_mysql():
    found = extract_skills("MySQL administration")
    assert "mysql" in found and "sql" not in found


def test_special_character_skills():
    found = extract_skills("C++, Node.js and CI/CD pipelines")
    assert {"c++", "node.js", "ci/cd"} <= found


def test_case_insensitive():
    assert "python" in extract_skills("PYTHON")


def test_no_skills_returns_empty_set():
    assert extract_skills("I enjoy hiking and painting") == set()

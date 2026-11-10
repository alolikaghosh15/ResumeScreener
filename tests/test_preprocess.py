import io

from src.preprocess import clean_text, extract_text, is_low_text


def test_clean_text_lowercases_and_strips_noise():
    out = clean_text("Email: a@b.com  Visit http://x.com  Python3 & C++!")
    assert out == "email visit python c"


def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\n\nc") == "a b c"


def test_clean_text_handles_non_string():
    assert clean_text(None) == "none"


def test_extract_text_reads_txt():
    f = io.BytesIO(b"Hello resume")
    f.name = "cv.txt"
    assert extract_text(f) == "Hello resume"


def test_extract_text_rewinds_file_before_reading():
    f = io.BytesIO(b"Hello resume")
    f.name = "cv.txt"
    f.read()  # simulate an earlier read (Streamlit reruns)
    assert extract_text(f) == "Hello resume"


def test_is_low_text():
    assert is_low_text("too short")
    assert not is_low_text("word " * 60)

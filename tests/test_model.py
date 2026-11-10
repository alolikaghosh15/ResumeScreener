import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.dedupe import drop_near_duplicates
from src.explain import top_contributing_words
from src.preprocess import clean_text


@pytest.fixture(scope="module")
def tiny_model():
    texts = ["python flask pandas numpy", "python django pandas", "flask python api",
             "recruitment payroll onboarding", "payroll hiring employee relations",
             "onboarding recruitment hiring"]
    labels = ["Dev", "Dev", "Dev", "HR", "HR", "HR"]
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=500))])
    return pipe.fit(texts, labels)


def test_explanation_words_belong_to_prediction(tiny_model):
    words = top_contributing_words(tiny_model, "python flask", k=5)
    assert words and words[0][0] in {"python", "flask"}
    assert all(score > 0 for _, score in words)


def test_explanation_for_unseen_words_is_empty(tiny_model):
    assert top_contributing_words(tiny_model, "zzzz qqqq") == []


def test_near_duplicates_removed_within_category():
    df = pd.DataFrame({
        "Category": ["A", "A", "A", "B"],
        "Resume": ["python flask api developer", "python flask api developer", "payroll hiring", "python flask api developer"],
    })
    df["clean"] = df["Resume"].apply(clean_text)
    out = drop_near_duplicates(df)
    assert len(out[out["Category"] == "A"]) == 2   # one copy removed
    assert len(out[out["Category"] == "B"]) == 1   # other category untouched

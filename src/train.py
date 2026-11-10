"""Train and compare models with 5-fold CV, then save a Logistic Regression
pipeline (needed for probabilities + explainability).

Usage: python -m src.train
Expects data/resumes.csv with columns: Category, Resume
"""
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .dedupe import deduplicate, load_raw
from .preprocess import clean_text


def make_pipe(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
                        max_features=50000, min_df=2)),
        ("clf", clf),
    ])


def main():
    if os.path.exists("data/resumes_clean.csv"):
        print("Using data/resumes_clean.csv (already de-duplicated)")
        df = load_raw("data/resumes_clean.csv")
    else:
        print("data/resumes_clean.csv not found - de-duplicating now (run python -m src.dedupe to inspect first)")
        raw = load_raw("data/resumes.csv")
        df = deduplicate(raw)
        print(f"Rows: {len(raw)} -> {len(df)} after removing duplicates")
    df["clean"] = df["Resume"].apply(clean_text)

    # stratified split + 5-fold CV need enough samples per class
    counts = df["Category"].value_counts()
    rare = counts[counts < 10].index
    if len(rare):
        print("Dropping categories with <10 resumes:", list(rare))
        df = df[~df["Category"].isin(rare)]
    print("Categories:", df["Category"].nunique())

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["clean"], df["Category"], test_size=0.2, stratify=df["Category"], random_state=42)

    candidates = {
        "LinearSVC": LinearSVC(),
        "LogisticRegression": LogisticRegression(C=10, max_iter=300),
        "NaiveBayes": MultinomialNB(),
    }
    print("\n5-fold CV accuracy on training set:")
    for name, clf in candidates.items():
        scores = cross_val_score(make_pipe(clf), X_tr, y_tr, cv=5, n_jobs=-1)
        print(f"  {name:20s} {scores.mean():.3f} +/- {scores.std():.3f}")

    final = make_pipe(LogisticRegression(C=10, max_iter=300))
    final.fit(X_tr, y_tr)
    preds = final.predict(X_te)
    print("\nHeld-out test report (LogisticRegression):")
    print(classification_report(y_te, preds))

    fig, ax = plt.subplots(figsize=(16, 16))
    ConfusionMatrixDisplay.from_predictions(y_te, preds, xticks_rotation=90, ax=ax, colorbar=False, include_values=False)
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)

    joblib.dump(final, "models/model.joblib")
    print("Saved models/model.joblib and models/confusion_matrix.png")


if __name__ == "__main__":
    main()

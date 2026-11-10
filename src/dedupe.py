"""Step 1 of the pipeline: remove duplicate resumes and save data/resumes_clean.csv.

Usage: python -m src.dedupe
Reads data/resumes.csv (Category + Text/Resume/Resume_str), removes
  1. exact duplicate resumes, and
  2. near-duplicates within each category (cosine similarity >= THRESHOLD on TF-IDF),
then writes data/resumes_clean.csv and prints a before/after table per category.
train.py automatically uses the cleaned file when it exists.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import clean_text

THRESHOLD = 0.95


def load_raw(path="data/resumes.csv"):
    df = pd.read_csv(path)
    for alt in ("Text", "Resume_str"):
        if "Resume" not in df.columns and alt in df.columns:
            df = df.rename(columns={alt: "Resume"})
    df = df.dropna(subset=["Resume", "Category"])
    df["Category"] = df["Category"].astype(str).str.strip()
    return df[["Category", "Resume"]]


def drop_near_duplicates(df, threshold=THRESHOLD):
    """df needs a 'clean' column. Keeps the first of each near-identical group."""
    keep = []
    for _, g in df.groupby("Category"):
        X = TfidfVectorizer(stop_words="english").fit_transform(g["clean"])
        sim = (X @ X.T).tocsr()
        dropped = set()
        for i in range(X.shape[0]):
            if i in dropped:
                continue
            row = sim.getrow(i)
            dropped.update(int(j) for j, v in zip(row.indices, row.data) if j > i and v >= threshold)
        keep.extend(g.index[[i for i in range(len(g)) if i not in dropped]])
    return df.loc[keep]


def deduplicate(df, threshold=THRESHOLD):
    df = df.drop_duplicates(subset="Resume").copy()
    df["clean"] = df["Resume"].apply(clean_text)
    df = drop_near_duplicates(df, threshold)
    return df[["Category", "Resume"]]


def main():
    raw = load_raw()
    clean = deduplicate(raw)
    table = pd.DataFrame({"before": raw["Category"].value_counts(),
                          "after": clean["Category"].value_counts()}).fillna(0).astype(int)
    table["removed"] = table["before"] - table["after"]
    print(table.sort_values("removed", ascending=False).to_string())
    print(f"\nTotal: {len(raw)} -> {len(clean)} ({len(raw) - len(clean)} removed)")
    clean.to_csv("data/resumes_clean.csv", index=False)
    print("Saved data/resumes_clean.csv")


if __name__ == "__main__":
    main()

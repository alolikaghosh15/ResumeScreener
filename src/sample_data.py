"""Generates a tiny SYNTHETIC dataset to smoke-test the pipeline end to end.
NOT for your report results - use the real Kaggle data for that.

Usage: python -m src.sample_data   (writes data/resumes.csv; refuses to overwrite)
"""
import os
import random

import pandas as pd

BANK = {
    "Data Science": ["python", "pandas", "machine learning", "tensorflow", "statistics", "scikit-learn", "nlp"],
    "Java Developer": ["java", "spring", "hibernate", "mysql", "rest api", "microservices", "junit"],
    "HR": ["recruitment", "payroll", "employee relations", "talent acquisition", "onboarding", "policies"],
    "Testing": ["selenium", "manual testing", "automation testing", "test cases", "junit", "bug tracking"],
    "Web Designing": ["html", "css", "javascript", "react", "photoshop", "responsive design"],
}
FILLER = ["experienced", "team player", "responsible for", "worked on", "projects", "bachelor degree", "skills"]


def main(n_per_class=40, seed=0):
    path = "data/resumes.csv"
    if os.path.exists(path):
        raise SystemExit(f"{path} already exists - not overwriting.")
    random.seed(seed)
    rows = []
    for cat, words in BANK.items():
        for _ in range(n_per_class):
            toks = random.choices(words, k=12) + random.choices(FILLER, k=8)
            random.shuffle(toks)
            rows.append({"Category": cat, "Resume": " ".join(toks) + f" id{random.random()}"})
    pd.DataFrame(rows).to_csv(path, index=False)
    print("Wrote", path, "(synthetic, for smoke-testing only)")


if __name__ == "__main__":
    main()

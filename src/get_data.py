"""Download the ResumeAtlas dataset and save it as data/resumes.csv.

Usage:
    pip install pyarrow            # (or: pip install datasets, used as a fallback)
    python -m src.get_data

Dataset: ahmedheakl/resume-atlas (Heakl et al., 2024, arXiv:2406.18125).
It has a single split ("train") stored as one parquet file, columns: Category, Text.
Check the dataset page for its licence and cite the paper in your project.
"""
import os

import pandas as pd

PARQUET_URL = "https://huggingface.co/api/datasets/ahmedheakl/resume-atlas/parquet/default/train/0.parquet"


def fetch() -> pd.DataFrame:
    try:
        return pd.read_parquet(PARQUET_URL)
    except Exception as e:  # missing pyarrow, network hiccup, etc.
        print("Direct parquet download failed:", repr(e))
        print("Trying the 'datasets' library instead...")
        from datasets import load_dataset
        ds = load_dataset("ahmedheakl/resume-atlas")
        return pd.concat([ds[s].to_pandas() for s in ds], ignore_index=True)


def main():
    df = fetch()
    print("Columns:", list(df.columns), "| rows:", len(df))
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/resumes.csv", index=False)
    print("Saved data/resumes.csv")
    print(df["Category"].value_counts().to_string())


if __name__ == "__main__":
    main()

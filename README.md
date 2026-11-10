# Smart Resume Screener

Resume category classification + explainability + skill extraction + job-description matching.

## Setup
1. `pip install -r requirements.txt`
2. Download the Kaggle "Resume Dataset" (UpdatedResumeDataSet.csv) and save it as `data/resumes.csv`
   (columns: `Category`, `Resume`).
3. Train: `python -m src.train`
4. Run: `streamlit run app.py`

## Structure
- `src/preprocess.py` text extraction and cleaning
- `src/train.py` de-duplication, 5-fold CV model comparison, saves model + confusion matrix
- `src/explain.py` top words driving the prediction
- `src/skills.py` skill extraction (edit the SKILLS list)
- `src/match.py` hybrid JD match score (60% semantic + 40% skill overlap) and skill gap
- `app.py` Streamlit UI

## Get the ResumeAtlas dataset
`pip install pyarrow` then `python -m src.get_data` (falls back to the `datasets` library if needed) (saves `data/resumes.csv`, columns Category + Text).

## Pipeline order
1. `python -m src.get_data` (download)
2. `python -m src.dedupe` (remove duplicates, writes `data/resumes_clean.csv`, prints a before/after table)
3. `python -m src.train`
4. `streamlit run app.py`

## Quick pipeline test (no Kaggle file yet)
`python -m src.sample_data` creates a tiny SYNTHETIC `data/resumes.csv` so you can check that
training and the app run. Delete it and use the real dataset for any results you report.

## Notes
- Match score falls back to TF-IDF similarity if the sentence-transformer model can't be loaded (e.g. offline).
- Resumes are embedded in chunks (the model reads ~256 tokens at a time).
"# ResumeScreener" 

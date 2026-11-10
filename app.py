import os

import joblib
import pandas as pd
import streamlit as st

from src.preprocess import clean_text, extract_text
from src.skills import extract_skills
from src.explain import top_contributing_words
from src.match import match_score

st.set_page_config(page_title="Smart Resume Screener", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")


if not os.path.exists("models/model.joblib"):
    st.error("No trained model found. Put your dataset at data/resumes.csv and run: python -m src.train")
    st.stop()
model = load_model()

st.title("Smart Resume Screener")
st.caption("Category prediction, explainability, skills, and job-description matching.")

jd = st.sidebar.text_area("Paste job description (optional)", height=300)
tab1, tab2 = st.tabs(["Analyze one resume", "Rank multiple resumes"])

with tab1:
    f = st.file_uploader("Upload a resume (PDF or TXT)", type=["pdf", "txt"], key="single")
    if f:
        raw = extract_text(f)
        clean = clean_text(raw)
        probs = model.predict_proba([clean])[0]
        top3 = sorted(zip(model.classes_, probs), key=lambda x: -x[1])[:3]

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Predicted category")
            st.success(f"{top3[0][0]}  ({top3[0][1]:.0%} confidence)")
            st.write("Other likely categories:")
            for cat, p in top3[1:]:
                st.write(f"- {cat}: {p:.0%}")

            st.subheader("Why this category?")
            words = top_contributing_words(model, clean)
            if words:
                st.bar_chart(pd.DataFrame(words, columns=["word", "contribution"]).set_index("word"))
        with c2:
            st.subheader("Skills detected")
            st.write(", ".join(sorted(extract_skills(raw))) or "None found")

            if jd.strip():
                st.subheader("Job match")
                r = match_score(raw, jd)
                st.metric("Match score", f"{r['score']}/100")
                st.write(f"Semantic similarity: {r['semantic']}%  |  Required-skill overlap: {r['skill_overlap']}%")
                st.write("Matched skills:", ", ".join(r["matched"]) or "-")
                st.write("Missing skills:", ", ".join(r["missing"]) or "-")

with tab2:
    if not jd.strip():
        st.info("Paste a job description in the sidebar to rank resumes.")
    files = st.file_uploader("Upload multiple resumes", type=["pdf", "txt"],
                             accept_multiple_files=True, key="multi")
    if files and jd.strip():
        rows = []
        for f in files:
            raw = extract_text(f)
            r = match_score(raw, jd)
            cat = model.predict([clean_text(raw)])[0]
            rows.append({"Resume": f.name, "Category": cat, "Match score": r["score"],
                         "Missing skills": ", ".join(r["missing"])})
        st.dataframe(pd.DataFrame(rows).sort_values("Match score", ascending=False))

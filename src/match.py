"""Resume <-> job description matching.

Score = 60% semantic similarity + 40% required-skill overlap.
Semantic similarity uses sentence-transformers (resume is split into chunks because
the model only reads ~256 tokens at a time). If sentence-transformers is not
installed or the model can't be downloaded, it falls back to TF-IDF cosine similarity
so the app still runs offline.
"""
import numpy as np

from .skills import extract_skills

_model = None
_model_failed = False


def _embedder():
    global _model, _model_failed
    if _model is None and not _model_failed:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _model_failed = True
    return _model


def _chunks(text: str, size: int = 150):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)] or [""]


def semantic_similarity(resume: str, jd: str) -> float:
    m = _embedder()
    if m is not None:
        from sentence_transformers import util
        jd_vec = m.encode(jd[:3000])
        sims = [float(util.cos_sim(m.encode(c), jd_vec)) for c in _chunks(resume)]
        # mean of the best 3 chunks: rewards a resume that matches in places
        return max(0.0, float(np.mean(sorted(sims, reverse=True)[:3])))
    # offline fallback
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(stop_words="english").fit([resume, jd])
    return float(cosine_similarity(vec.transform([resume]), vec.transform([jd]))[0][0])


def skill_gap(resume: str, jd: str):
    """Returns (matched, missing, overlap_ratio) based on skills required in the JD."""
    required = extract_skills(jd)
    have = extract_skills(resume)
    matched, missing = sorted(required & have), sorted(required - have)
    ratio = len(matched) / len(required) if required else 0.0
    return matched, missing, ratio


def match_score(resume: str, jd: str) -> dict:
    sem = semantic_similarity(resume, jd)
    matched, missing, ratio = skill_gap(resume, jd)
    return {
        "score": round(100 * (0.6 * sem + 0.4 * ratio), 1),
        "semantic": round(100 * sem, 1),
        "skill_overlap": round(100 * ratio, 1),
        "matched": matched,
        "missing": missing,
    }

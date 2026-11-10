import numpy as np


def top_contributing_words(model, cleaned_text: str, k: int = 10):
    """Words that pushed the prediction toward the predicted class.
    contribution = tfidf value x logistic-regression coefficient."""
    tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    vec = tfidf.transform([cleaned_text])
    pred = clf.predict(vec)[0]
    class_idx = list(clf.classes_).index(pred)
    if clf.coef_.shape[0] == 1:
        # binary model: one coefficient row, which favours classes_[1]
        coef = clf.coef_[0] if class_idx == 1 else -clf.coef_[0]
    else:
        coef = clf.coef_[class_idx]
    contrib = vec.toarray()[0] * coef
    names = tfidf.get_feature_names_out()
    order = np.argsort(contrib)[::-1][:k]
    return [(names[i], float(contrib[i])) for i in order if contrib[i] > 0]

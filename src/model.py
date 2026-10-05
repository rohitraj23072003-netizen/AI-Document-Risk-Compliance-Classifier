from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = Path("artifacts")
MODEL_PATH = MODEL_DIR / "risk_classifier.joblib"

def train_model(csv_path="data/sample_documents.csv"):
    df = pd.read_csv(csv_path)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, random_state=42))
    ])
    model.fit(df["text"], df["label"])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model, df

def load_model():
    if not MODEL_PATH.exists():
        return train_model()[0]
    return joblib.load(MODEL_PATH)

def predict(model, text):
    probabilities = model.predict_proba([text])[0]
    classes = model.named_steps["clf"].classes_
    order = probabilities.argsort()[::-1]
    return [
        {"category": classes[i], "confidence": float(probabilities[i])}
        for i in order
    ]

def important_terms(model, text, top_n=8):
    vec = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    X = vec.transform([text])
    feature_names = vec.get_feature_names_out()
    scores = X.toarray()[0]

    # Multiclass explanation: contribution of each active term toward predicted class.
    probs = clf.predict_proba(X)[0]
    pred_idx = probs.argmax()
    coef = clf.coef_[pred_idx]
    contributions = scores * coef
    ranked = contributions.argsort()[::-1]

    terms = []
    for idx in ranked:
        if scores[idx] <= 0:
            continue
        terms.append({
            "term": feature_names[idx],
            "contribution": float(contributions[idx])
        })
        if len(terms) >= top_n:
            break
    return terms

def nearest_documents(model, text, df, top_n=3):
    vec = model.named_steps["tfidf"]
    corpus_X = vec.transform(df["text"])
    query_X = vec.transform([text])
    sims = cosine_similarity(query_X, corpus_X)[0]
    order = sims.argsort()[::-1][:top_n]
    return [
        {
            "label": str(df.iloc[i]["label"]),
            "similarity": float(sims[i]),
            "text": str(df.iloc[i]["text"])
        }
        for i in order
    ]

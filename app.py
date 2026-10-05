import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

from src.document_loader import extract_document, clean_text
from src.model import train_model, load_model, predict, important_terms, nearest_documents
from src.evaluate import evaluate

st.set_page_config(page_title="AI Document Risk & Compliance Classifier", layout="wide")
st.title("AI Document Risk & Compliance Classifier")
st.caption("Automatically classify business documents into risk/compliance categories and flag documents that may need manual review.")

model, training_df = train_model()
threshold = st.sidebar.slider("Manual-review confidence threshold", 0.50, 0.99, 0.70, 0.01)

uploaded = st.file_uploader("Upload a document", type=["txt", "pdf"])
sample_choice = st.selectbox("Or choose a sample document", ["None"] + list(training_df.index.astype(str)))

text = ""
if uploaded:
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.getvalue())
        temp_path = f.name
    text = extract_document(temp_path)
elif sample_choice != "None":
    text = str(training_df.iloc[int(sample_choice)]["text"])

if text:
    st.subheader("Extracted and cleaned text")
    st.write(text)

    results = predict(model, text)
    top = results[0]
    manual_review = top["confidence"] < threshold

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted category", top["category"])
    c2.metric("Confidence", f"{top['confidence']:.1%}")
    c3.metric("Manual review", "YES" if manual_review else "NO")

    if manual_review:
        st.warning("This document is below the confidence threshold and should be sent to manual review.")
    else:
        st.success("This document meets the configured confidence threshold.")

    st.subheader("Classification probabilities")
    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    st.subheader("Explainability — important terms")
    terms = important_terms(model, text)
    st.dataframe(pd.DataFrame(terms), use_container_width=True, hide_index=True)

    st.subheader("Nearest similar documents")
    similar = nearest_documents(model, text, training_df)
    st.dataframe(pd.DataFrame(similar), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Class-wise evaluation")
if st.button("Run evaluation"):
    accuracy, report = evaluate()
    st.metric("Accuracy", f"{accuracy:.1%}")
    st.dataframe(report, use_container_width=True)

st.subheader("Flagged-document dashboard")
if text:
    flagged = pd.DataFrame([{
        "document": uploaded.name if uploaded else "Sample document",
        "category": top["category"],
        "confidence": top["confidence"],
        "manual_review": manual_review
    }])
    st.dataframe(flagged, use_container_width=True, hide_index=True)

# 3. AI Document Risk & Compliance Classifier

## Short description
Automatically classify business documents into risk/compliance categories and flag documents that may need manual review.

## How to approach

1. Collect sample documents and define categories such as low, medium, high risk, or policy-specific classes.
2. Extract text from PDFs/documents and remove irrelevant formatting.
3. Generate embeddings or TF-IDF features and train a classifier.
4. Add explainability using important terms or nearest similar documents.
5. Set a confidence threshold so uncertain cases go to manual review.
6. Evaluate class-wise performance and build a dashboard showing flagged documents.

Suggested stack: use Python for data preparation and modeling where applicable; Git/GitHub for version control; and a simple Streamlit/API layer when a UI or deployment component is useful.

## What this implementation contains

- Sample document dataset with risk/compliance labels.
- PDF/text extraction and formatting cleanup.
- TF-IDF feature generation and Logistic Regression classifier.
- Important-term explanations using model coefficients.
- Nearest similar documents using cosine similarity.
- Configurable confidence threshold with manual-review flagging.
- Class-wise evaluation using precision, recall and F1.
- Streamlit dashboard for document classification and flagged-document review.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app can train the model from the included sample data automatically.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from .model import train_model

def evaluate(csv_path="data/sample_documents.csv"):
    df = pd.read_csv(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.35, random_state=42, stratify=df["label"]
    )
    model, _ = train_model(csv_path)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    return accuracy_score(y_test, preds), pd.DataFrame(report).T


import argparse
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to labeled CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns")

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["text"] != ""]

    # Normalize common dataset label variants.
    label_map = {
        "ham": "legitimate",
        "safe": "legitimate",
        "legitimate": "legitimate",
        "0": "legitimate",
        "phishing": "phishing",
        "spam": "phishing",
        "malicious": "phishing",
        "1": "phishing",
    }
    df["label"] = df["label"].map(label_map)
    df = df.dropna().drop_duplicates(subset=["text", "label"])

    if df["label"].nunique() != 2:
        raise ValueError(
            "Dataset must contain both legitimate and phishing examples"
        )

    if df["label"].value_counts().min() < 2:
        raise ValueError("At least two examples per class are required")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Held-out test-set evaluation:")
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(model, "model.joblib")
    print("Saved trained model to model.joblib")


if __name__ == "__main__":
    main()

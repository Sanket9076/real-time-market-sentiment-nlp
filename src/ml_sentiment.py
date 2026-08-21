"""
Supervised Machine Learning Sentiment Classifier

Trains a TF-IDF + Logistic Regression supervised sentiment classification pipeline.
Evaluates precision, recall, f1-score, accuracy, and confusion matrix, comparing against
the baseline VADER analyzer.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class MLSentimentClassifier:
    """
    Supervised Machine Learning pipeline for sentiment classification.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=500, random_state=42)
        self.is_trained = False
        self.evaluation_metrics: Dict[str, Any] = {}

    def train_and_evaluate(self, df: pd.DataFrame, text_col: str = "title", label_col: str = "sentiment_label") -> Dict[str, Any]:
        """
        Train ML model on text dataset and calculate classification evaluation metrics.
        
        Args:
            df (pd.DataFrame): Dataset with preprocessed text and target labels.
            text_col (str): Text column name.
            label_col (str): Sentiment target label column name ('Positive', 'Neutral', 'Negative').
            
        Returns:
            Dict[str, Any]: Evaluation metrics dictionary.
        """
        if df.empty or text_col not in df.columns or label_col not in df.columns:
            return {"error": "Invalid dataframe or missing required columns."}

        X_text = df[text_col].astype(str)
        y = df[label_col]

        if len(y.unique()) < 2:
            return {"error": "Need at least 2 distinct classes to train ML model."}

        # Train / Test split (80/20)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_text, y, test_size=0.2, random_state=42, stratify=y if len(y) > 20 else None
        )

        # Transform TF-IDF
        X_train_vec = self.vectorizer.fit_transform(X_train_raw)
        X_test_vec = self.vectorizer.transform(X_test_raw)

        # Fit Model
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True

        # Predict
        y_pred = self.model.predict(X_test_vec)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=["Positive", "Neutral", "Negative"])

        labels_list = ["Positive", "Neutral", "Negative"]

        self.evaluation_metrics = {
            "model_type": "TF-IDF + Logistic Regression",
            "accuracy": round(float(acc), 4),
            "precision_weighted": round(float(precision), 4),
            "recall_weighted": round(float(recall), 4),
            "f1_score_weighted": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),
            "labels": labels_list,
            "train_samples": len(y_train),
            "test_samples": len(y_test)
        }

        print(f"[MLSentimentClassifier] Model Trained successfully. Accuracy={acc:.4f}, F1={f1:.4f}")
        return self.evaluation_metrics

    def predict(self, texts: list) -> list:
        """Predict sentiment labels for new text strings."""
        if not self.is_trained:
            return ["Neutral"] * len(texts)
        
        vecs = self.vectorizer.transform([str(t) for t in texts])
        return self.model.predict(vecs).tolist()

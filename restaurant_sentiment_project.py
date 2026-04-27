import re
import string
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def load_data(file_path: str = "restaurants.csv") -> pd.DataFrame:
    """Load the restaurant reviews dataset and validate required columns."""
    df = pd.read_csv(file_path)

    required_columns = ["restaurant_name", "review_text", "rating"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}. "
            "Expected columns: restaurant_name, review_text, rating"
        )

    # Keep only useful columns and remove rows with missing values in key fields.
    df = df[required_columns].dropna(subset=required_columns).copy()

    # Ensure rating is numeric and in the expected range.
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]

    return df


def preprocess_text(text: str) -> str:
    """Lowercase text, remove punctuation, and remove non-letter characters."""
    if not isinstance(text, str):
        text = ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_model(df: pd.DataFrame) -> Tuple[LogisticRegression, TfidfVectorizer, pd.Series, np.ndarray, pd.DataFrame]:
    """
    Build sentiment labels, vectorize text, split data, and train logistic regression.

    Returns:
        model, vectorizer, y_test, y_pred, dataframe_with_predictions
    """
    working_df = df.copy()

    # Create the required sentiment label from numeric ratings.
    working_df["sentiment"] = (working_df["rating"] >= 4).astype(int)

    # Clean raw review text before vectorization.
    working_df["clean_text"] = working_df["review_text"].apply(preprocess_text)

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(working_df["clean_text"])
    y = working_df["sentiment"]

    # Use stratify when both classes are present to preserve class balance.
    stratify_target = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Predict sentiment for all reviews so we can compute restaurant-level averages.
    working_df["predicted_sentiment"] = model.predict(X)

    return model, vectorizer, y_test, y_pred, working_df


def evaluate_model(y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics for the sentiment classifier."""
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
    }
    return metrics


def compute_scores(df_with_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute restaurant-level average rating, average sentiment, and final weighted score.

    final_score = (average_rating * 0.6) + (average_sentiment * 5 * 0.4)
    """
    summary = (
        df_with_predictions.groupby("restaurant_name", as_index=False)
        .agg(
            average_rating=("rating", "mean"),
            average_sentiment=("predicted_sentiment", "mean"),
        )
    )

    summary["final_score"] = (
        summary["average_rating"] * 0.6 + summary["average_sentiment"] * 5 * 0.4
    )

    summary = summary.sort_values("final_score", ascending=False).reset_index(drop=True)
    return summary


def predict_custom_review(model: LogisticRegression, vectorizer: TfidfVectorizer) -> None:
    """Optional bonus: predict sentiment for a custom review entered by the user."""
    user_choice = input("\nDo you want to test a custom review? (y/n): ").strip().lower()
    if user_choice != "y":
        return

    custom_review = input("Enter your review text: ").strip()
    cleaned_review = preprocess_text(custom_review)
    review_vector = vectorizer.transform([cleaned_review])
    prediction = model.predict(review_vector)[0]
    label = "Positive" if prediction == 1 else "Negative"

    print(f"Predicted sentiment: {label}")


def main() -> None:
    print("=== Best Restaurants in Pennsylvania Using Sentiment Analysis and Ratings ===")

    # 1) Load dataset.
    data_file = "restaurants.csv"
    if not Path(data_file).exists():
        raise FileNotFoundError(
            "restaurants.csv was not found. "
            "Run download_and_prepare_data.py first to create it from Kaggle data."
        )

    df = load_data(data_file)

    # 2-7) Train and evaluate model.
    model, vectorizer, y_test, y_pred, df_with_predictions = train_model(df)
    metrics = evaluate_model(y_test, y_pred)

    # 8-9) Compute final restaurant scores and keep top 10.
    restaurant_scores = compute_scores(df_with_predictions)
    top_10 = restaurant_scores.head(10).copy()

    # 10) Print results clearly.
    print("\nModel Evaluation Metrics")
    print("-" * 30)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nTop 10 Restaurants by Final Score")
    print("-" * 45)
    print(
        top_10[
            [
                "restaurant_name",
                "average_rating",
                "average_sentiment",
                "final_score",
            ]
        ].to_string(index=False)
    )

    # Bonus: let user test a custom review.
    predict_custom_review(model, vectorizer)


if __name__ == "__main__":
    main()

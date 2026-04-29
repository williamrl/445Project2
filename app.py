from pathlib import Path
import pickle
from typing import Any, Dict

import pandas as pd
from flask import Flask, render_template, request

from restaurant_sentiment_project import compute_scores, evaluate_model, load_data, preprocess_text, train_model

app = Flask(__name__)


def _build_artifacts_if_missing() -> None:
    model_file = Path("model_bundle.pkl")
    rankings_file = Path("restaurant_rankings.csv")

    if model_file.exists() and rankings_file.exists():
        return

    data_file = Path("restaurants.csv")
    if not data_file.exists():
        raise FileNotFoundError(
            "restaurants.csv was not found. Run download_and_prepare_data.py first."
        )

    df = load_data(str(data_file))
    model, vectorizer, y_test, y_pred, df_with_predictions = train_model(df)
    metrics = evaluate_model(y_test, y_pred)
    scores = compute_scores(df_with_predictions)

    with model_file.open("wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer, "metrics": metrics}, f)

    scores.to_csv(rankings_file, index=False)


def _load_bundle() -> Dict[str, Any]:
    _build_artifacts_if_missing()
    with Path("model_bundle.pkl").open("rb") as f:
        return pickle.load(f)


def _load_top_restaurants(limit: int = 10) -> pd.DataFrame:
    _build_artifacts_if_missing()
    rankings = pd.read_csv("restaurant_rankings.csv")
    return rankings.head(limit)


@app.route("/", methods=["GET", "POST"])
def index():
    bundle = _load_bundle()
    top_restaurants = _load_top_restaurants(10)

    prediction_label = None
    if request.method == "POST":
        review_text = request.form.get("review_text", "")
        cleaned = preprocess_text(review_text)
        features = bundle["vectorizer"].transform([cleaned])
        prediction = bundle["model"].predict(features)[0]
        prediction_label = "Positive" if prediction == 1 else "Negative"

    metrics = bundle.get("metrics", {})

    return render_template(
        "index.html",
        prediction_label=prediction_label,
        top_restaurants=top_restaurants.to_dict(orient="records"),
        metrics=metrics,
    )


if __name__ == "__main__":
    app.run(debug=True)

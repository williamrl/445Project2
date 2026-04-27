from pathlib import Path
import pickle

import pandas as pd

from restaurant_sentiment_project import (
    compute_scores,
    evaluate_model,
    load_data,
    train_model,
)


def main() -> None:
    data_file = Path("restaurants.csv")
    if not data_file.exists():
        raise FileNotFoundError(
            "restaurants.csv was not found. Run download_and_prepare_data.py first."
        )

    df = load_data(str(data_file))
    model, vectorizer, y_test, y_pred, df_with_predictions = train_model(df)
    metrics = evaluate_model(y_test, y_pred)
    scores = compute_scores(df_with_predictions)

    top_10 = scores.head(10).copy()

    artifact = {
        "model": model,
        "vectorizer": vectorizer,
        "metrics": metrics,
    }

    with Path("model_bundle.pkl").open("wb") as f:
        pickle.dump(artifact, f)

    scores.to_csv("restaurant_rankings.csv", index=False)
    top_10.to_csv("top_10_restaurants.csv", index=False)

    print("Saved model artifact: model_bundle.pkl")
    print("Saved rankings: restaurant_rankings.csv")
    print("Saved top 10: top_10_restaurants.csv")
    print("\nModel Metrics")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()

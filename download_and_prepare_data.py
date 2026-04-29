from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Prepare restaurants.csv from a source CSV file. "
			"By default, this script will NOT overwrite an existing restaurants.csv."
		)
	)
	parser.add_argument(
		"--source",
		type=str,
		default="",
		help=(
			"Path to a source CSV file containing restaurant_name, review_text, rating. "
			"If omitted and restaurants.csv exists, the script exits safely."
		),
	)
	parser.add_argument(
		"--output",
		type=str,
		default="restaurants.csv",
		help="Output CSV path. Default: restaurants.csv",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Allow overwriting the output file if it already exists.",
	)
	return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
	required = ["restaurant_name", "review_text", "rating"]
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(
			f"Missing required columns: {missing}. "
			"Expected columns: restaurant_name, review_text, rating"
		)

	cleaned = df[required].dropna(subset=required).copy()
	cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce")
	cleaned = cleaned.dropna(subset=["rating"])
	cleaned = cleaned[(cleaned["rating"] >= 1) & (cleaned["rating"] <= 5)]
	return cleaned


def main() -> None:
	args = parse_args()
	output_path = Path(args.output)

	# Safety-first behavior: preserve user-created data by default.
	if output_path.exists() and not args.force:
		print(
			f"{output_path} already exists. "
			"Keeping your existing file unchanged. "
			"Use --force only if you intentionally want to overwrite it."
		)
		return

	if not args.source:
		if output_path.exists() and args.force:
			print("--force was passed, but no --source file was provided. No changes made.")
			return

		raise FileNotFoundError(
			"No --source file provided and output file does not exist. "
			"Pass --source <path-to-csv> to create restaurants.csv."
		)

	source_path = Path(args.source)
	if not source_path.exists():
		raise FileNotFoundError(f"Source file not found: {source_path}")

	source_df = pd.read_csv(source_path)
	prepared_df = validate_columns(source_df)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	prepared_df.to_csv(output_path, index=False)

	print(f"Saved prepared dataset to: {output_path}")
	print(f"Rows: {len(prepared_df)}")


if __name__ == "__main__":
	main()

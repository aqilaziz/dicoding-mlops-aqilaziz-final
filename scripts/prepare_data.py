"""Download and prepare the red wine quality dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from urllib.request import urlopen


DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = PROJECT_ROOT / "data_source"
RAW_PATH = RAW_DIR / "winequality-red-raw.csv"
OUTPUT_PATH = DATA_DIR / "winequality-red.csv"

FIELD_MAP = {
    "fixed acidity": "fixed_acidity",
    "volatile acidity": "volatile_acidity",
    "citric acid": "citric_acid",
    "residual sugar": "residual_sugar",
    "chlorides": "chlorides",
    "free sulfur dioxide": "free_sulfur_dioxide",
    "total sulfur dioxide": "total_sulfur_dioxide",
    "density": "density",
    "pH": "ph",
    "sulphates": "sulphates",
    "alcohol": "alcohol",
}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with urlopen(DATA_URL, timeout=30) as response:
        RAW_PATH.write_bytes(response.read())

    with RAW_PATH.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source, delimiter=";")
        rows = []
        for row in reader:
            output_row = {
                clean_name: row[original_name]
                for original_name, clean_name in FIELD_MAP.items()
            }
            output_row["good_quality"] = "1" if int(row["quality"]) >= 6 else "0"
            rows.append(output_row)

    fieldnames = list(FIELD_MAP.values()) + ["good_quality"]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    positive = sum(int(row["good_quality"]) for row in rows)
    print(f"Prepared {len(rows)} rows at {OUTPUT_PATH}")
    print(f"Positive label ratio: {positive / len(rows):.3f}")


if __name__ == "__main__":
    main()

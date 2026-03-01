"""
Data loading and preprocessing utilities.

All operations are designed to avoid data leakage.
"""

from __future__ import annotations

import os
from typing import Tuple, List

import pandas as pd
import kagglehub as kg


def find_first_csv(folder_path: str) -> str:
    """
    Locate the first CSV file inside a downloaded Kaggle dataset folder.
    """
    csv_files: List[str] = []

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file_name))

    if not csv_files:
        raise FileNotFoundError("No CSV file found in dataset directory.")

    return csv_files[0]


def load_dataset(dataset_slug: str) -> pd.DataFrame:
    """
    Download dataset using KaggleHub and return as DataFrame.
    """
    path = kg.dataset_download(dataset_slug)
    csv_path = find_first_csv(path)
    df = pd.read_csv(csv_path)
    return df


def basic_sanity_checks(df: pd.DataFrame) -> None:
    """
    Basic structural validation of dataset.
    """
    required_cols = {"id", "diagnosis"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and binary target y.
    """
    y_raw = df["diagnosis"].astype(str).str.strip()
    y = y_raw.map({"M": 1, "B": 0})

    if y.isnull().any():
        raise ValueError("Unexpected values found in diagnosis column.")

    X = df.drop(columns=["diagnosis"]).copy()

    if "id" in X.columns:
        X = X.drop(columns=["id"])

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y
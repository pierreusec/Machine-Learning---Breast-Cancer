"""
Centralized configuration of the Machine Learning project.

This file groups global constants to facilitate 
reproducibility and experimentation.
"""

from pathlib import Path

# ------------------------------
# Reproducibility
# ------------------------------
RANDOM_STATE: int = 42

# ------------------------------
# Data split
# ------------------------------
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

# ------------------------------
# Dataset
# ------------------------------
DATASET_SLUG: str = "yasserh/breast-cancer-dataset"

# ------------------------------
# Paths
# ------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
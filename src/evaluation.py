"""
Evaluation utilities and performance metrics.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


def cv_scores_pipeline(
    pipe,
    X,
    y,
    scoring: str,
    cv_folds: int,
    random_state: int,
) -> Tuple[float, float]:

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
    return float(scores.mean()), float(scores.std())


def evaluate_pipeline_on_test(
    pipe,
    X_test,
    y_test,
) -> Dict[str, float]:

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def save_roc_pr_curves(
    y_true,
    y_proba,
    model_name: str,
    artifacts_dir,
) -> None:

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / f"{model_name}_roc_curve.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / f"{model_name}_pr_curve.png")
    plt.close()
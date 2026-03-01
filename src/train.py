"""
Main experimental pipeline.
"""

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import pandas as pd
import joblib
import torch

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    recall_score,
    average_precision_score,
)

# Local modules
from config import *
from data import *
from models import *
from evaluation import *
from dl import *


# ==========================================================
# Local Preprocess (kept as before)
# ==========================================================

def make_preprocess_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


# ==========================================================
# Main
# ==========================================================

def main() -> None:

    # ==========================================================
    # 1. Data Loading
    # ==========================================================

    df = load_dataset(DATASET_SLUG)
    basic_sanity_checks(df)
    X, y = prepare_features_and_target(df)

    # ==========================================================
    # 2. Train / Test Split
    # ==========================================================

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y.values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y.values,
    )

    preprocess = make_preprocess_pipeline()
    models = make_models(RANDOM_STATE)

    # ==========================================================
    # 3. Cross-Validation Benchmark
    # ==========================================================

    print("\nCross-Validation Benchmark Ok\n")

    cv_summary: List[Dict[str, Any]] = []

    for name, model in models.items():

        pipe = Pipeline(
            [
                ("preprocess", preprocess),
                ("model", model),
            ]
        )

        mean_auc, std_auc = cv_scores_pipeline(
            pipe,
            X_train_df,
            y_train,
            scoring="roc_auc",
            cv_folds=CV_FOLDS,
            random_state=RANDOM_STATE,
        )

        mean_f1, std_f1 = cv_scores_pipeline(
            pipe,
            X_train_df,
            y_train,
            scoring="f1",
            cv_folds=CV_FOLDS,
            random_state=RANDOM_STATE,
        )

        mean_pr, std_pr = cv_scores_pipeline(
            pipe,
            X_train_df,
            y_train,
            scoring="average_precision",
            cv_folds=CV_FOLDS,
            random_state=RANDOM_STATE,
        )

        print(
            f"{name} | ROC_AUC {mean_auc:.4f} "
            f"| F1 {mean_f1:.4f} "
            f"| PR_AUC {mean_pr:.4f}"
        )

        cv_summary.append(
            {
                "model": name,
                "roc_auc_mean": mean_auc,
                "f1_mean": mean_f1,
                "pr_auc_mean": mean_pr,
            }
        )

    pd.DataFrame(cv_summary).to_csv(
        ARTIFACTS_DIR / "cv_metrics.csv",
        index=False,
    )

    # ==========================================================
    # 4. Final Evaluation on Test Set
    # ==========================================================

    print("\nFinal Evaluation - Test Set Ok\n")

    test_metrics: List[Dict[str, Any]] = []

    for name, model in models.items():

        pipe = Pipeline(
            [
                ("preprocess", preprocess),
                ("model", model),
            ]
        )

        pipe.fit(X_train_df, y_train)

        metrics = evaluate_pipeline_on_test(
            pipe,
            X_test_df,
            y_test,
        )

        metrics["model"] = name
        test_metrics.append(metrics)

    # Select best classical model
    best_classical = max(test_metrics, key=lambda x: x["roc_auc"])
    best_model_name = best_classical["model"]

    best_pipe = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", models[best_model_name]),
        ]
    )

    best_pipe.fit(X_train_df, y_train)

    joblib.dump(
        best_pipe,
        ARTIFACTS_DIR / "best_classical_model.pkl",
    )

    best_proba = best_pipe.predict_proba(X_test_df)[:, 1]

    save_roc_pr_curves(
        y_true=y_test,
        y_proba=best_proba,
        model_name=best_model_name,
        artifacts_dir=ARTIFACTS_DIR,
    )

    # ==========================================================
    # 5. Deep Learning
    # ==========================================================

    print("\nDeep Learning - MLP Ok\n")

    X_tr_df, X_val_df, y_tr, y_val = train_test_split(
        X_train_df,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    dl_preprocess = make_preprocess_pipeline()

    X_tr = dl_preprocess.fit_transform(X_tr_df)
    X_val = dl_preprocess.transform(X_val_df)
    X_te = dl_preprocess.transform(X_test_df)

    mlp = train_mlp(X_tr, y_tr, X_val, y_val)

    proba_mlp = mlp_predict_proba(mlp, X_te)[:, 1]
    pred_mlp = (proba_mlp >= 0.5).astype(int)

    mlp_metrics = {
        "model": "MLP",
        "accuracy": float(accuracy_score(y_test, pred_mlp)),
        "f1": float(f1_score(y_test, pred_mlp)),
        "recall": float(recall_score(y_test, pred_mlp)),
        "roc_auc": float(roc_auc_score(y_test, proba_mlp)),
        "pr_auc": float(average_precision_score(y_test, proba_mlp)),
    }

    torch.save(
        mlp.state_dict(),
        ARTIFACTS_DIR / "mlp_weights.pt",
    )

    test_metrics.append(mlp_metrics)

    pd.DataFrame(test_metrics).to_csv(
        ARTIFACTS_DIR / "test_metrics.csv",
        index=False,
    )

    print("\nTraining completed successfully.")
    print(f"All results have been saved to: {ARTIFACTS_DIR.resolve()}")
    print("Generated artifacts:")
    print(" - cv_metrics.csv")
    print(" - test_metrics.csv")
    print(" - best_classical_model.pkl")
    print(" - mlp_weights.pt")
    print(" - ROC and PR curve images")


if __name__ == "__main__":
    main()
"""
Projet Machine Learning Supervisé
Classification binaire cancer du sein
Target diagnosis
M malignant
B benign

Ce script
Télécharge et charge le dataset via KaggleHub
Applique un preprocessing sans fuite d'information
Compare plusieurs modèles classiques
Ajoute un soft voting manuel
Entraîne un MLP tabulaire en deep learning
Évalue sur un test set final
Sauvegarde les métriques dans artifacts
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import kagglehub as kg

from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.optim as optim


RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
DATASET_SLUG = "yasserh/breast-cancer-dataset"

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int) -> None:
    """
    Fixe les graines pour reproductibilité.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_first_csv(folder_path: str) -> str:
    """
    Trouve le premier fichier csv dans un dossier.
    """
    csv_files: List[str] = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file_name))

    if not csv_files:
        raise FileNotFoundError("Aucun fichier csv trouvé dans le dossier téléchargé.")

    return csv_files[0]


def load_dataset() -> pd.DataFrame:
    """
    Télécharge le dataset via KaggleHub et charge le csv.
    """
    path = kg.dataset_download(DATASET_SLUG)
    print("Path to dataset files:", path)

    csv_path = find_first_csv(path)
    print("CSV utilisé :", csv_path)

    df = pd.read_csv(csv_path)
    return df


def basic_sanity_checks(df: pd.DataFrame) -> None:
    """
    Vérifications rapides.
    """
    required_cols = {"id", "diagnosis"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    print("Nombre de lignes :", df.shape[0])
    print("Nombre de colonnes :", df.shape[1])
    print("Valeurs manquantes totales :", int(df.isnull().sum().sum()))


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prépare X et y.
    Retire id des features.
    Encode diagnosis en binaire M 1 B 0.
    Force les features en numérique.
    """
    y_raw = df["diagnosis"].astype(str).str.strip()
    y = y_raw.map({"M": 1, "B": 0})

    if y.isnull().any():
        bad_values = sorted(set(y_raw[y.isnull()].unique()))
        raise ValueError(f"Valeurs inattendues dans diagnosis: {bad_values}")

    X = df.drop(columns=["diagnosis"]).copy()

    if "id" in X.columns:
        X = X.drop(columns=["id"])

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y


def make_preprocess_pipeline() -> Pipeline:
    """
    Pipeline preprocessing.
    Imputation médiane puis standardisation.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def make_models() -> Dict[str, BaseEstimator]:
    """
    Modèles classiques.
    """
    models: Dict[str, BaseEstimator] = {
        "LogisticRegression": LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    return models


def cv_scores_pipeline(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    scoring: str,
    cv_folds: int,
) -> Tuple[float, float]:
    """
    Score moyen et écart type en CV.
    Le preprocessing est inclus dans la CV donc pas de fuite.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
    return float(scores.mean()), float(scores.std())


def evaluate_pipeline_on_test(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    name: str,
) -> Dict[str, float]:
    """
    Évaluation sur test set.
    """
    y_pred = pipe.predict(X_test)

    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = float("nan")

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"Résultats test set {name}")
    print(f"Accuracy {acc:.4f}")
    print(f"F1 {f1:.4f}")
    print(f"Recall {rec:.4f}")
    if not np.isnan(auc):
        print(f"ROC_AUC {auc:.4f}")
    print("Matrice de confusion")
    print(cm)

    return {"accuracy": acc, "f1": f1, "recall": rec, "roc_auc": auc}


def grid_search_manual_pipeline(
    X: pd.DataFrame,
    y: np.ndarray,
    preprocess: Pipeline,
    model_class: Any,
    param_grid: Dict[str, List[Any]],
    scoring: str,
    cv_folds: int,
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
    """
    Grid search manuel.
    """
    results: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_params: Dict[str, Any] = {}

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    for values in product(*param_values):
        params = dict(zip(param_names, values))
        model = model_class(**params)

        pipe = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", model),
            ]
        )

        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
        mean_score = float(scores.mean())
        std_score = float(scores.std())

        results.append({"params": params, "mean_score": mean_score, "std_score": std_score})

        print(f"Params: {params} -> Score: {mean_score:.4f} (+/- {std_score:.4f})")

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, float(best_score), results


@dataclass
class SoftVotingManual:
    """
    Soft voting manuel basé sur la moyenne des probabilités.
    Les estimateurs doivent exposer predict_proba.
    """
    estimators: List[Pipeline]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SoftVotingManual":
        for est in self.estimators:
            est.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas: List[np.ndarray] = []
        for est in self.estimators:
            probas.append(est.predict_proba(X))
        avg = np.mean(np.stack(probas, axis=0), axis=0)
        return avg

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba_pos = self.predict_proba(X)[:, 1]
        return (proba_pos >= 0.5).astype(int)


class TabularMLP(nn.Module):
    """
    MLP pour données tabulaires.
    """
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 15,
) -> TabularMLP:
    """
    Entraînement MLP avec early stopping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TabularMLP(n_features=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    no_improve = 0

    n_train = X_train.shape[0]
    indices = np.arange(n_train)

    for epoch in range(1, max_epochs + 1):
        model.train()
        np.random.shuffle(indices)

        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = X_train_t[batch_idx].to(device)
            yb = y_train_t[batch_idx].to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * len(batch_idx)

        epoch_loss /= n_train

        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t.to(device))
            val_loss = float(criterion(logits_val, y_val_t.to(device)).item())

        print(f"Epoch {epoch:03d} train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def mlp_predict_proba(model: TabularMLP, X: np.ndarray) -> np.ndarray:
    """
    Probabilités pour le MLP.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(xt)
        proba_pos = torch.sigmoid(logits).cpu().numpy()

    proba_neg = 1.0 - proba_pos
    return np.vstack([proba_neg, proba_pos]).T


def main() -> None:
    """
    Pipeline complet.
    """
    set_seed(RANDOM_STATE)

    df = load_dataset()
    basic_sanity_checks(df)

    X, y = prepare_features_and_target(df)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y.values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y.values,
    )

    preprocess = make_preprocess_pipeline()
    models = make_models()

    print("\n" + "=" * 60)
    print("Comparaison modèles classiques")
    print("=" * 60)

    cv_summary: List[Dict[str, Any]] = []
    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", model),
            ]
        )
        mean_auc, std_auc = cv_scores_pipeline(pipe, X_train_df, y_train, scoring="roc_auc", cv_folds=CV_FOLDS)
        mean_f1, std_f1 = cv_scores_pipeline(pipe, X_train_df, y_train, scoring="f1", cv_folds=CV_FOLDS)

        print(f"{name} ROC_AUC {mean_auc:.4f} (+/- {std_auc:.4f}) F1 {mean_f1:.4f} (+/- {std_f1:.4f})")

        cv_summary.append(
            {
                "model": name,
                "roc_auc_mean": mean_auc,
                "roc_auc_std": std_auc,
                "f1_mean": mean_f1,
                "f1_std": std_f1,
            }
        )

    print("\n" + "=" * 60)
    print("Custom Soft Voting manuel")
    print("=" * 60)

    voting_estimators = [
        Pipeline([("preprocess", preprocess), ("model", models["LogisticRegression"])]),
        Pipeline([("preprocess", preprocess), ("model", models["RandomForest"])]),
    ]
    soft_voting = SoftVotingManual(estimators=voting_estimators)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    voting_auc_scores: List[float] = []
    voting_f1_scores: List[float] = []

    for train_idx, val_idx in cv.split(X_train_df, y_train):
        X_tr = X_train_df.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_va = X_train_df.iloc[val_idx]
        y_va = y_train[val_idx]

        soft_voting.fit(X_tr, y_tr)
        proba = soft_voting.predict_proba(X_va)[:, 1]
        pred = (proba >= 0.5).astype(int)

        voting_auc_scores.append(float(roc_auc_score(y_va, proba)))
        voting_f1_scores.append(float(f1_score(y_va, pred)))

    print(f"SoftVotingManual ROC_AUC {np.mean(voting_auc_scores):.4f} (+/- {np.std(voting_auc_scores):.4f})")
    print(f"SoftVotingManual F1 {np.mean(voting_f1_scores):.4f} (+/- {np.std(voting_f1_scores):.4f})")

    print("\n" + "=" * 60)
    print("Grid search manuel DecisionTree")
    print("=" * 60)

    dt_grid = {
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_split": [2, 5, 10],
        "random_state": [RANDOM_STATE],
    }

    best_params, best_score, dt_results = grid_search_manual_pipeline(
        X_train_df,
        y_train,
        preprocess,
        DecisionTreeClassifier,
        dt_grid,
        scoring="roc_auc",
        cv_folds=CV_FOLDS,
    )

    print(f"Nombre de combinaisons testées: {len(dt_results)}")
    print(f"Meilleurs paramètres: {best_params}")
    print(f"Meilleur score CV ROC_AUC: {best_score:.4f}")

    print("\n" + "=" * 60)
    print("Évaluation finale sur test set")
    print("=" * 60)

    test_metrics: List[Dict[str, Any]] = []

    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        pipe.fit(X_train_df, y_train)
        m = evaluate_pipeline_on_test(pipe, X_test_df, y_test, name=name)
        m["model"] = name
        test_metrics.append(m)

    soft_voting.fit(X_train_df, y_train)
    proba_test = soft_voting.predict_proba(X_test_df)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    sv_metrics = {
        "model": "SoftVotingManual",
        "accuracy": float(accuracy_score(y_test, pred_test)),
        "f1": float(f1_score(y_test, pred_test)),
        "recall": float(recall_score(y_test, pred_test)),
        "roc_auc": float(roc_auc_score(y_test, proba_test)),
    }
    print("\n" + "=" * 60)
    print("Résultats test set SoftVotingManual")
    print(f"Accuracy {sv_metrics['accuracy']:.4f}")
    print(f"F1 {sv_metrics['f1']:.4f}")
    print(f"Recall {sv_metrics['recall']:.4f}")
    print(f"ROC_AUC {sv_metrics['roc_auc']:.4f}")
    print("Matrice de confusion")
    print(confusion_matrix(y_test, pred_test))

    test_metrics.append(sv_metrics)

    best_dt_pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", DecisionTreeClassifier(**best_params)),
        ]
    )
    best_dt_pipe.fit(X_train_df, y_train)
    dtm = evaluate_pipeline_on_test(best_dt_pipe, X_test_df, y_test, name="DecisionTree best grid")
    dtm["model"] = "DecisionTree best grid"
    test_metrics.append(dtm)

    print("\n" + "=" * 60)
    print("Deep learning MLP")
    print("=" * 60)

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
    }

    print("Résultats test set MLP")
    print(f"Accuracy {mlp_metrics['accuracy']:.4f}")
    print(f"F1 {mlp_metrics['f1']:.4f}")
    print(f"Recall {mlp_metrics['recall']:.4f}")
    print(f"ROC_AUC {mlp_metrics['roc_auc']:.4f}")
    print("Matrice de confusion")
    print(confusion_matrix(y_test, pred_mlp))

    test_metrics.append(mlp_metrics)

    metrics_df = pd.DataFrame(test_metrics)
    metrics_path = ARTIFACTS_DIR / "test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("\nMétriques sauvegardées :", metrics_path)


if __name__ == "__main__":
    main()
"""
Model definitions and classical ML utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def make_models(random_state: int) -> Dict[str, BaseEstimator]:
    """
    Return dictionary of classical models.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=3000,
            random_state=random_state,
        ),
        "DecisionTree": DecisionTreeClassifier(
            random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


@dataclass
class SoftVotingManual:
    """
    Manual soft voting ensemble using probability averaging.
    """
    estimators: List[Pipeline]

    def fit(self, X, y) -> "SoftVotingManual":
        for est in self.estimators:
            est.fit(X, y)
        return self

    def predict_proba(self, X):
        probas = [est.predict_proba(X) for est in self.estimators]
        avg = np.mean(np.stack(probas, axis=0), axis=0)
        return avg

    def predict(self, X):
        proba_pos = self.predict_proba(X)[:, 1]
        return (proba_pos >= 0.5).astype(int)
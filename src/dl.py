"""
Deep learning utilities for tabular classification.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TabularMLP(nn.Module):
    """
    Multi-layer perceptron for tabular data.
    """

    def __init__(self, n_features: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    max_epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 15,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularMLP(X_train.shape[1]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    no_improve = 0

    for epoch in range(max_epochs):

        model.train()
        optimizer.zero_grad()

        logits = model(torch.tensor(X_train, dtype=torch.float32).to(device))
        loss = criterion(
            logits,
            torch.tensor(y_train, dtype=torch.float32).to(device),
        )

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(
                torch.tensor(X_val, dtype=torch.float32).to(device)
            )
            val_loss = criterion(
                val_logits,
                torch.tensor(y_val, dtype=torch.float32).to(device),
            )

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

def mlp_predict_proba(model, X):
    """
    Return probability estimates for positive class.
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
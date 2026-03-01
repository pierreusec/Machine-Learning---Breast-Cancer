# Projet de Machine Learning Supervisé

## Classification des tumeurs mammaires (M vs B)

---

## 1. Contexte du projet

Ce projet vise à construire un pipeline de classification supervisée permettant de prédire si une tumeur mammaire est :

* **M** : Maligne
* **B** : Bénigne

Le dataset utilisé est le **Breast Cancer Wisconsin Dataset** disponible sur Kaggle.
Les données ne sont pas stockées dans ce repository et sont récupérées dynamiquement via **KaggleHub**, afin de garantir la conformité, la propreté du dépôt GitHub et la reproductibilité.

L’objectif pédagogique est double :

* Réaliser une analyse exploratoire rigoureuse (EDA) centrée sur la qualité des données.
* Construire un pipeline de modélisation reproductible, comparatif et proprement évalué.

Le projet inclut également un rapport détaillé (`en format pdf`) présentant la méthodologie, les résultats et la discussion critique.

---

## 2. Structure du repository

Le projet est organisé selon une séparation stricte des responsabilités :

Notebook/
EDA.ipynb
Analyse exploratoire et contrôle qualité des données
Aucune modélisation n’est réalisée dans ce notebook

src/
train.py
Script principal d’entraînement, validation croisée, optimisation et évaluation

artifacts/
Fichiers de sortie générés automatiquement (non versionnés)

data/
Données locales (non versionnées)

Le dossier `artifacts/` contient notamment :

* `cv_metrics.csv`
* `test_metrics.csv`
* `best_classical_model.pkl`
* `mlp_weights.pt`
* `roc_curve.png`
* `pr_curve.png`

---

## 3. Installation

### 1. Cloner le repository

git clone <url_du_repo>
cd <nom_du_repo>

### 2. Créer un environnement virtuel

python -m venv .venv

Activation :

* Windows :
  .venv\Scripts\activate

* Linux / Mac :
  source .venv/bin/activate

### 3. Installer les dépendances

pip install -r requirements.txt

---

## 4. Exécution

L’entraînement complet s’exécute via :

python src/train.py

Ce script :

* Télécharge les données via KaggleHub
* Effectue un split train/test stratifié (80/20)
* Applique un preprocessing appris uniquement sur le train
* Réalise une validation croisée stratifiée (k = 5)
* Compare plusieurs modèles classiques
* Implémente un soft voting manuel
* Effectue une optimisation manuelle d’hyperparamètres pour le Decision Tree
* Entraîne un MLP tabulaire avec early stopping
* Évalue tous les modèles sur un test set strictement hors-échantillon
* Calcule Accuracy, F1-score, Recall, ROC-AUC et PR-AUC
* Exporte automatiquement les métriques, modèles et courbes

---

## 5. Données

Les données ne sont pas incluses dans le repository.

Elles sont récupérées dynamiquement via KaggleHub dans le script `train.py`.

Aucun fichier de dataset n’est versionné sur GitHub afin de :

* Respecter les conditions d’utilisation
* Garantir un dépôt propre
* Maintenir la reproductibilité

Le dataset contient :

* 569 observations
* 30 variables explicatives numériques
* 1 variable cible : diagnosis (M / B)
* Aucune valeur manquante
* Un léger déséquilibre de classes (~63 % B / 37 % M)

---

## 6. Résultats

Les métriques finales sont exportées automatiquement dans le dossier :

artifacts/

Exemples de fichiers générés :

* `test_metrics.csv`
* `cv_metrics.csv`
* `roc_curve.png`
* `pr_curve.png`
* `best_classical_model.pkl`
* `mlp_weights.pt`

Ces fichiers permettent de comparer les performances des modèles sur :

* Accuracy
* F1-score
* Recall
* ROC-AUC
* PR-AUC

Les courbes ROC et Precision-Recall sont également sauvegardées pour visualiser la capacité discriminante des modèles.

---

## 7. Méthodologie

Le projet repose sur les principes suivants :

* Séparation stricte entre EDA et modélisation
* Aucun fit de preprocessing sur l’ensemble du dataset
* Validation croisée stratifiée (k = 5)
* Comparaison de modèles classiques et d’un modèle de type MLP
* Implémentation d’un soft voting manuel
* Optimisation manuelle d’hyperparamètres
* Évaluation hors-échantillon stricte
* Export des métriques et sauvegarde des modèles pour assurer la traçabilité des résultats
* Prévention explicite du data leakage

---

## 8. Répartition du travail

* **EDA.ipynb**
  Analyse exploratoire et contrôle qualité des données

* **train.py**
  Pipeline d’entraînement, validation croisée, optimisation, comparaison des modèles, export des résultats et sauvegarde des modèles

* **rapport.pdf**
  Présentation académique détaillée du cadre méthodologique, des résultats et de la discussion critique

---

## 9. Reproductibilité

Le projet peut être reproduit intégralement à partir des étapes suivantes :

1. Cloner le repository
2. Installer les dépendances
3. Exécuter `python src/train.py`

Aucune donnée n’est nécessaire localement avant l’exécution.
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

Le projet adopte désormais une architecture modulaire afin de séparer clairement les responsabilités et d’améliorer la lisibilité du code.

```
src/
    config.py
        Paramètres globaux du projet (seed, chemins, configuration CV)

    data.py
        Fonctions de chargement et de préparation des données

    models.py
        Implémentation des modèles classiques et du Soft Voting manuel

    evaluation.py
        Validation croisée, calcul des métriques, export des courbes ROC et PR

    dl.py
        Implémentation du modèle de Deep Learning (MLP tabulaire)

    train.py
        Script principal orchestrant l’ensemble du pipeline expérimental

Notebook/
    EDA.ipynb
        Analyse exploratoire des données (aucune modélisation)

artifacts/
    Résultats générés automatiquement (non versionnés)

Le dossier `artifacts/` contient notamment :

* `cv_metrics.csv`
* `test_metrics.csv`
* `best_classical_model.pkl`
* `mlp_weights.pt`
* `roc_curve.png`
* `pr_curve.png`

data/
    Données locales (non nécessaires à l’exécution)
```

Le dossier `artifacts/` contient notamment :

* `cv_metrics.csv`
* `test_metrics.csv`
* `best_classical_model.pkl`
* `mlp_weights.pt`
* `roc_curve.png`
* `pr_curve.png`

Cette organisation permet :

- Une meilleure maintenabilité du code
- Une séparation claire entre données, modèles, évaluation et deep learning
- Une architecture cohérente avec les standards professionnels en Machine Learning

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

```bash
python src/train.py
```

Le script :

- Télécharge automatiquement le dataset via KaggleHub
- Effectue un split train/test stratifié
- Réalise une validation croisée (ROC-AUC, F1-score, PR-AUC)
- Compare plusieurs modèles classiques
- Implémente un Soft Voting manuel
- Entraîne un modèle de Deep Learning (MLP tabulaire)
- Évalue les performances sur un jeu de test strict
- Exporte les métriques, modèles entraînés et courbes ROC/PR

Tous les résultats sont sauvegardés dans le dossier :

artifacts/

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
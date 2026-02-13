# Projet de Machine Learning Supervisé  
## Classification des tumeurs mammaires (M vs B)

---

## 1. Contexte du projet

Ce projet vise à construire un pipeline de classification supervisée permettant de prédire si une tumeur mammaire est :

- **M** : Maligne  
- **B** : Bénigne  

Le dataset utilisé est le **Breast Cancer Wisconsin Dataset** disponible sur Kaggle.  
Les données ne sont pas stockées dans ce repository et sont récupérées dynamiquement via **KaggleHub**, afin de garantir la conformité, la propreté du dépôt GitHub et la reproductibilité.

L’objectif pédagogique est double :

- Réaliser une analyse exploratoire rigoureuse (EDA) centrée sur la qualité des données.
- Construire un pipeline de modélisation reproductible, comparatif et proprement évalué.

---

## 2. Structure du repository

Le projet est organisé selon une séparation stricte des responsabilités :

Notebook/
    EDA.ipynb
    Analyse exploratoire et contrôle qualité des données
    Aucune modélisation n’est réalisée dans ce notebook

src/
    train.py
    Script principal d’entraînement, validation croisée et évaluation

artifacts/
    Fichiers de sortie générés automatiquement (non versionnés)

data/
    Données locales éventuelles (non versionnées)


---

## 3. Installation

### 1. Cloner le repository

git clone <url_du_repo>
cd <nom_du_repo>


### 2. Créer un environnement virtuel

python -m venv .venv


Activation :

- Windows :
.venv\Scripts\activate


- Linux / Mac :
source .venv/bin/activate


### 3. Installer les dépendances

pip install -r requirements.txt


---

## 4. Exécution

L’entraînement complet s’exécute via :

python src/train.py


Ce script :

- Télécharge les données via KaggleHub
- Effectue un split train/test stratifié
- Applique un preprocessing appris uniquement sur le train
- Réalise une validation croisée stratifiée
- Compare plusieurs modèles
- Exporte automatiquement les métriques

---

## 5. Données

Les données ne sont pas incluses dans le repository.

Elles sont récupérées dynamiquement via KaggleHub dans le script `train.py`.

Aucun fichier de dataset n’est versionné sur GitHub afin de :

- Respecter les conditions d’utilisation
- Garantir un dépôt propre
- Maintenir la reproductibilité

---

## 6. Résultats

Les métriques finales sont exportées automatiquement dans le dossier :

artifacts/


Exemples de fichiers générés :

- `test_metrics.csv`
- `cv_metrics.csv`

Ces fichiers permettent de comparer les performances des modèles sur :

- Accuracy
- F1-score
- Recall
- ROC-AUC

---

## 7. Méthodologie

Le projet repose sur les principes suivants :

- Séparation stricte entre EDA et modélisation
- Aucun fit de preprocessing sur l’ensemble du dataset
- Validation croisée stratifiée
- Comparaison de modèles classiques et d’un modèle de type MLP
- Export des métriques pour assurer la traçabilité des résultats

---

## 8. Répartition du travail

- **EDA.ipynb**  
  Analyse exploratoire et contrôle qualité des données

- **train.py**  
  Pipeline d’entraînement, validation croisée, comparaison des modèles et export des résultats

---

## 9. Reproductibilité

Le projet peut être reproduit intégralement à partir des étapes suivantes :

1. Cloner le repository
2. Installer les dépendances
3. Exécuter `python src/train.py`

Aucune donnée n’est nécessaire localement avant l’exécution.
Ce fichier est prêt à être collé tel quel dans README.md.

Si vous le souhaitez, je peux maintenant vous proposer une version encore plus académique, avec un encadré méthodologique sur la prévention du data leakage — ce qui peut clairement faire la différence sur la note.


Vous avez dit :
Contrainte : Ecrire le REDAME.md en un même bloc

ChatGPT a dit :
Très bien.
Voici le README.md complet en un seul bloc, directement copiable-collable dans VS Code.

# Projet de Machine Learning Supervisé  
## Classification des tumeurs mammaires (M vs B)

---

## 1. Contexte du projet

Ce projet vise à construire un pipeline de classification supervisée permettant de prédire si une tumeur mammaire est :

- **M** : Maligne  
- **B** : Bénigne  

Le dataset utilisé est le **Breast Cancer Wisconsin Dataset** disponible sur Kaggle.  
Les données ne sont pas stockées dans ce repository et sont récupérées dynamiquement via **KaggleHub**, afin de garantir la conformité, la propreté du dépôt GitHub et la reproductibilité.

L’objectif pédagogique est double :

- Réaliser une analyse exploratoire rigoureuse (EDA) centrée sur la qualité des données.
- Construire un pipeline de modélisation reproductible, comparatif et proprement évalué.

---

## 2. Structure du repository

Le projet est organisé selon une séparation stricte des responsabilités :

Notebook/
EDA.ipynb
Analyse exploratoire et contrôle qualité des données
Aucune modélisation n’est réalisée dans ce notebook

src/
train.py
Script principal d’entraînement, validation croisée et évaluation

artifacts/
Fichiers de sortie générés automatiquement (non versionnés)

data/
Données locales éventuelles (non versionnées)


---

## 3. Installation

Cloner le repository :

git clone <url_du_repo>
cd <nom_du_repo>


Créer un environnement virtuel :

python -m venv .venv


Activation :

Windows :
.venv\Scripts\activate


Linux / Mac :
source .venv/bin/activate


Installer les dépendances :

pip install -r requirements.txt


---

## 4. Exécution

L’entraînement complet s’exécute via :

python src/train.py


Ce script :

- Télécharge les données via KaggleHub
- Effectue un split train/test stratifié
- Applique un preprocessing appris uniquement sur le train
- Réalise une validation croisée stratifiée
- Compare plusieurs modèles
- Exporte automatiquement les métriques

---

## 5. Données

Les données ne sont pas incluses dans le repository.

Elles sont récupérées dynamiquement via KaggleHub dans le script `train.py`.

Aucun fichier de dataset n’est versionné sur GitHub afin de :

- Respecter les conditions d’utilisation
- Garantir un dépôt propre
- Maintenir la reproductibilité

---

## 6. Résultats

Les métriques finales sont exportées automatiquement dans le dossier :

artifacts/


Exemples de fichiers générés :

- `test_metrics.csv`
- `cv_metrics.csv`

Ces fichiers permettent de comparer les performances des modèles sur :

- Accuracy
- F1-score
- Recall
- ROC-AUC

---

## 7. Méthodologie

Le projet repose sur les principes suivants :

- Séparation stricte entre EDA et modélisation
- Aucun fit de preprocessing sur l’ensemble du dataset
- Validation croisée stratifiée
- Comparaison de modèles classiques et d’un modèle de type MLP
- Export des métriques pour assurer la traçabilité des résultats

---

## 8. Répartition du travail

- **EDA.ipynb** : Analyse exploratoire et contrôle qualité des données  
- **train.py** : Pipeline d’entraînement, validation croisée, comparaison des modèles et export des résultats  

---

## 9. Reproductibilité

Le projet peut être reproduit intégralement à partir des étapes suivantes :

1. Cloner le repository
2. Installer les dépendances
3. Exécuter `python src/train.py`

Aucune donnée n’est nécessaire localement avant l’exécution.
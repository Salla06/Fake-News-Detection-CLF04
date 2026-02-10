# Fake-News-Detection-CLF04
# Détection de Fake News - Projet FCC

**Étude Comparative ML Classique vs Deep Learning pour la Détection de Fausses Nouvelles**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5.2-green)
![Statut](https://img.shields.io/badge/Statut-Projet%20Académique-yellow)

---

## Table des Matières

- [Vue d'Ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Modèles Évalués](#modèles-évalués)
- [Résultats de Performance](#résultats-de-performance)
- [Stack Technologique](#stack-technologique)
- [Structure du Projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Application Web](#application-web)
- [Limitations et Analyse Critique](#limitations-et-analyse-critique)
- [Contact](#contact)

---

## Vue d'Ensemble

Projet académique développé dans le cadre d'un cours de **Natural Language Processing** pour la Federal Communications Commission. L'objectif est de comparer les approches de Machine Learning classique et Deep Learning pour la détection automatique de fausses nouvelles, avec un focus sur les articles annonçant des catastrophes fictives.

### Points Clés

- **Approche comparative** : Évaluation de 6 modèles (4 ML classiques + 2 DL)
- **Dataset** : 32,456 articles (Kaggle Fake News)
- **Meilleure performance** : Random Forest avec 99.56% d'accuracy
- **Livrables** : Notebooks de recherche + Application web déployée

### Contexte Académique

**Institution** : Federal Communications Commission  
**Cours** : Natural Language Processing  
**Type** : Projet de fin de semestre avec présentation  
**Année** : 2024-2025

---

## Architecture

Le projet se compose de deux éléments distincts :

### 1. Pipeline de Recherche (Notebooks)

```
Exploration des données → Prétraitement NLP → Modèles ML Classiques
                                           ↓
                                    Modèles Deep Learning
                                           ↓
                              Évaluation Comparative Finale
```

### 2. Application Web (Microservices)

```
┌──────────────────┐
│   Utilisateur    │
└────────┬─────────┘
         │
         ↓
┌────────────────────────────┐
│  Frontend (Streamlit)      │
│  - Interface utilisateur   │
│  - Support multilingue     │
│  - Visualisations          │
└────────┬───────────────────┘
         │ HTTP POST/JSON
         ↓
┌────────────────────────────┐
│  Backend (Flask API)       │
│  - Prétraitement NLP       │
│  - Inférence modèle        │
│  - Endpoint /predict       │
└────────────────────────────┘
```

| Composant | Technologie | Hébergement |
|-----------|-------------|-------------|
| Frontend | Streamlit 1.29 | Streamlit Cloud |
| Backend | Flask 3.0 + Gunicorn | Render.com |
| Communication | HTTP POST/JSON | - |

---

## Modèles Évalués

### ML Classique (Représentation TF-IDF)

| Modèle | Description | Use Case |
|--------|-------------|----------|
| **Linear SVM** | Support Vector Machine linéaire | Classification haute dimension |
| **Random Forest** | Ensemble de 100 arbres de décision | Robustesse et performance |
| **Logistic Regression** | Régression logistique | Baseline interprétable |
| **Naive Bayes** | Classificateur probabiliste | Référence rapide |

**Configuration TF-IDF** :
- Vocabulaire : 10000 features
- N-grammes : (1, 2) - unigrammes et bigrammes
- Fréquence document : min=1, max=0.8

### Deep Learning (Word Embeddings)

#### CNN (Convolutional Neural Network)

```
Embedding(10000, 100) → Conv1D(128, kernel=5) → GlobalMaxPooling
→ Dense(128) + Dropout(0.5) → Dense(64) + Dropout(0.3)
→ Dense(1, sigmoid)
```

**Hyperparamètres** :
- Learning rate : 2e-5
- Batch size : 32
- Optimizer : Adam

#### BiLSTM (Bidirectional LSTM)

```
Embedding(10000, 100) → Bidirectional(LSTM(64))
→ Attention Layer → Dense(128) + Dropout(0.5)
→ Dense(1, sigmoid)
```

---

## Résultats de Performance

### Tableau Comparatif des Modèles

| Modèle | Type | Accuracy | Precision | Recall | F1-Score |
|--------|------|----------|-----------|--------|----------|
| **Random Forest** | ML Classique | **99.56%** | 99.59% | 99.62% | 99.60% |
| **Linear SVM** | ML Classique | **99.48%** | 99.44% | 99.62% | 99.53% |
| **Logistic Regression** | ML Classique | **98.88%** | 98.60% | 99.38% | 98.99% |
| **Naive Bayes** | ML Classique | **95.28%** | 95.94% | 95.43% | 95.68% |
| **CNN** | Deep Learning | **En cours** | En cours | En cours | TBD |
| **BiLSTM** | Deep Learning | **En cours** | En cours | En cours | TBD |

**Note** : L'accuracy a été choisie comme métrique principale d'évaluation pour la comparaison des modèles.

### Statistiques du Dataset

| Catégorie | Quantité | Pourcentage |
|-----------|----------|-------------|
| **Total articles** | 32,456 | 100% |
| **Ensemble d'entraînement** | 22,719 | 70% |
| **Ensemble de validation** | 4,869 | 15% |
| **Ensemble de test** | 4,868 | 15% |
| **Articles FAKE** | 23,481 | 72.3% |
| **Articles REAL** | 8,975 | 27.7% |

### Matrice de Confusion (Linear SVM)

```
                Prédiction
            REAL    FAKE
Réel REAL   2150      32
     FAKE     98    5498
```

**Interprétation** :
- Vrais Positifs : 2,150 articles réels correctement identifiés
- Vrais Négatifs : 5,498 fake news correctement détectées
- Faux Positifs : 32 articles réels classés comme fake
- Faux Négatifs : 98 fake news manquées

---

## Stack Technologique

### Recherche et Modélisation

| Catégorie | Technologies |
|-----------|--------------|
| **Data Science** | Pandas 2.1.4, NumPy 1.24.3 |
| **NLP** | NLTK 3.8.1 (tokenisation, lemmatisation, stopwords) |
| **ML Classique** | Scikit-learn 1.5.2 (SVM, RF, LR, NB, TF-IDF) |
| **Deep Learning** | TensorFlow 2.15.0, Keras 2.15.0 |
| **Visualisation** | Matplotlib 3.8.2, Seaborn 0.13.0, WordCloud 1.9.3 |

### Application Web (Dashboard)

| Composant | Technologies |
|-----------|--------------|
| **Backend** | Flask 3.0.0, Gunicorn 21.2.0, flask-cors 4.0.0 |
| **Frontend** | Streamlit 1.29.0, Plotly 5.18.0 |
| **Traitement Fichiers** | PyPDF2 3.0.1, python-docx 1.1.0, openpyxl 3.1.2 |
| **Support Multilingue** | deep-translator 1.11.4, langdetect 1.0.9 |
| **Déploiement** | Render.com (backend), Streamlit Cloud (frontend) |

---

## Structure du Projet

```
fcc-fake-news-detection/
│
├── notebooks/                        # Pipeline de recherche
│   ├── fnd_01_exploration.ipynb     # EDA et statistiques
│   ├── fnd_02_processing.ipynb      # Prétraitement NLP
│   ├── fnd_03_ml_classique.ipynb    # SVM, RF, LR, NB
│   ├── fnd_04_cnn.ipynb             # Modèle CNN
│   ├── fnd_05_bilstm.ipynb          # Modèle BiLSTM
│   └── fnd_06_evaluation.ipynb      # Comparaison finale
│
├── models/                           # Modèles entraînés
│   ├── classical/
│   │   ├── linear_svm.pkl
│   │   ├── random_forest.pkl
│   │   ├── logistic_regression.pkl
│   │   ├── naive_bayes.pkl
│   │   ├── tfidf_vectorizer.pkl
│   │   └── classical_models_results.csv
│   └── deep/
│       ├── cnn_model.h5
│       ├── bilstm_model.h5
│       ├── keras_tokenizer.pkl
│       ├── cnn_metrics.json
│       └── bilstm_metrics.json
│
├── backend/                          # API Flask
│   ├── app.py
│   └── requirements.txt
│
├── frontend/                         # Interface Streamlit
│   ├── app.py
│   ├── utils.py
│   └── requirements.txt
│
├── data/                             # Données brutes
│   ├── Fake.csv
│   └── True.csv
│
└── README.md
```

---

## Installation

### Prérequis

- Python 3.11 ou supérieur
- Git
- pip

### Configuration Locale

#### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-username/fcc-fake-news-detection.git
cd fcc-fake-news-detection
```

#### 2. Créer un environnement virtuel

**Windows** :
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux** :
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

#### 4. Télécharger les ressources NLTK

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

#### 5. Lancer Jupyter (pour les notebooks)

```bash
jupyter notebook
```

---

## Utilisation

### Pipeline de Recherche (Notebooks)

Les notebooks doivent être exécutés dans l'ordre :

1. **fnd_01_exploration.ipynb** : Analyse exploratoire des données
   - Distribution des classes
   - Statistiques textuelles
   - Visualisations (nuages de mots, histogrammes)

2. **fnd_02_processing.ipynb** : Prétraitement NLP
   - Nettoyage du texte
   - Tokenisation et lemmatisation
   - Suppression des stopwords
   - Division train/validation/test (70/15/15)

3. **fnd_03_ml_classique.ipynb** : Modèles ML classiques
   - Vectorisation TF-IDF
   - Entraînement SVM, RF, LR, NB
   - Évaluation sur ensemble de validation
   - Sauvegarde modèles (.pkl) et métriques (.csv)

4. **fnd_04_cnn.ipynb** : Modèle CNN
   - Architecture convolutionnelle 1D
   - Entraînement avec callbacks (EarlyStopping, ModelCheckpoint)
   - Sauvegarde modèle (.h5) et métriques (.json)

5. **fnd_05_bilstm.ipynb** : Modèle BiLSTM
   - Architecture LSTM bidirectionnelle
   - Mécanisme d'attention
   - Sauvegarde modèle et métriques

6. **fnd_06_evaluation.ipynb** : Évaluation comparative
   - Chargement de tous les modèles
   - Comparaison des performances
   - Visualisations comparatives
   - Sélection du meilleur modèle

### Sauvegarde des Modèles

**ML Classiques** :
```python
import pickle

# Sauvegarder modèle
with open('models/classical/linear_svm.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# Sauvegarder vectorizer
with open('models/classical/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
```

**Deep Learning** :
```python
# Sauvegarder modèle Keras
cnn_model.save('models/deep/cnn_model.h5')

# Sauvegarder métriques
import json
with open('models/deep/cnn_metrics.json', 'w') as f:
    json.dump(cnn_metrics, f, indent=4)
```

---

## Application Web

### Démarrage Local

**Backend (API Flask)** :
```bash
cd backend
python app.py
# Disponible à http://localhost:5000
```

**Frontend (Streamlit)** :
```bash
cd frontend
streamlit run app.py
# Disponible à http://localhost:8501
```

### Fonctionnalités

**Méthodes d'entrée** :
- Saisie de texte direct
- Extraction depuis URL (HTML, PDF, DOCX)
- Upload de fichiers (TXT, PDF, DOCX, XLSX)

**Support multilingue** :
- Détection automatique de la langue
- Traduction vers l'anglais (5 langues supportées)

**Visualisations** :
- Score de confiance (jauge 0-100%)
- Distribution des probabilités (graphique en barres)
- Historique des analyses (timeline)

### API REST

**Endpoint de prédiction** :

```bash
POST /predict
Content-Type: application/json

{
  "text": "Votre texte d'article ici"
}
```

**Réponse** :

```json
{
  "prediction": 1,
  "label": "FAKE",
  "confidence": 0.8534,
  "probabilities": {
    "real": 0.1466,
    "fake": 0.8534
  }
}
```

**Exemple Python** :

```python
import requests

response = requests.post(
    "https://votre-api.onrender.com/predict",
    json={"text": "Breaking news: shocking discovery!"},
    timeout=60
)
result = response.json()
print(f"{result['label']} - Confiance: {result['confidence']:.2%}")
```

---

## Limitations et Analyse Critique

### Séparabilité Artificielle du Dataset

Le dataset Kaggle présente un **biais structurel majeur** :

| Type d'article | Catégories | Vocabulaire |
|----------------|-----------|-------------|
| **Articles REAL** | Politique américaine | Politicians, government, election |
| **Articles FAKE** | Sujets variés | Shocking, truth, exposed, conspiracy |

**Conséquence** : Les modèles apprennent à distinguer les **styles et sources** plutôt que la désinformation intrinsèque.

### Impact sur les Performances

Les scores exceptionnels (>99%) reflètent cette séparabilité artificielle plutôt qu'une capacité de généralisation réelle.


### Recommandations pour Amélioration

- Utiliser un dataset équilibré (mêmes sujets pour FAKE et REAL)
- Intégrer l'analyse des sources et métadonnées
- Ajouter du fact-checking externe
- Tester sur des données en conditions réelles

---
**Liens** :
- Application web : [À compléter]
- API REST : [À compléter]

---

## Remerciements

- **Dataset** : Kaggle Fake News Competition
- **Références académiques** : Roumeliotis et al. (2025) - Comparative study of CNNs, BERT, and GPT for fake news detection
- **Technologies** : TensorFlow, Scikit-learn, Streamlit, Flask
- **Hébergement** : Render.com et Streamlit Cloud

---

**Projet académique - NLP Course 2024-2025**

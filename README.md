# 🛍️ E-commerce Product Classification

> Système de classification automatique de produits e-commerce utilisant texte et/ou images

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Démo](#démo)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Modèles](#modèles)
- [API](#api)
- [Dashboard](#dashboard)
- [Déploiement](#déploiement)
- [Résultats](#résultats)
- [Contributions](#contributions)
- [Licence](#licence)

##  Vue d'ensemble

Ce projet implémente un système complet de classification de produits e-commerce capable de prédire automatiquement la catégorie d'un produit à partir :
- 📝 **Texte** : Description du produit
- 🖼️ **Image** : Photo du produit
- 🔗 **Multimodal** : Combinaison texte + image

### Catégories supportées

| Icône | Catégorie |
|-------|-----------|
| 👶 | Baby Care |
| 💄 | Beauty and Personal Care |
| 💻 | Computers |
| 🎨 | Home Decor & Festive Needs |
| 🛋️ | Home Furnishing |
| 🍳 | Kitchen & Dining |
| ⌚ | Watches |

## Démo

### Interface Web
![App Screenshot](assets/screenshot_app.png)

### API Endpoints
```bash
# Classification par texte
curl -X POST "http://localhost:8000/predict/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Montre analogique pour homme avec bracelet en cuir"}'

# Classification par image
curl -X POST "http://localhost:8000/predict/image" \
     -F "file=@product_image.jpg"
```

##  Fonctionnalités

### Application Web (Streamlit)
- ✅ Interface utilisateur moderne et intuitive
- ✅ Classification texte, image ou multimodale
- ✅ Dashboard analytique avec visualisations interactives
- ✅ Historique des prédictions
- ✅ Export des données en CSV
- ✅ Mode responsive (desktop/mobile)

### API REST (FastAPI)
- ✅ Endpoints de prédiction (texte, image, multimodal)
- ✅ Documentation auto-générée (Swagger/ReDoc)
- ✅ Validation des données avec Pydantic
- ✅ Gestion des erreurs robuste
- ✅ CORS configuré
- ✅ Health check endpoint

### Modèles
- ✅ **Texte** : SVM avec TF-IDF vectorization
- ✅ **Images** : Transfer Learning (ResNet50, EfficientNetB0)
- ✅ **Multimodal** : Late Fusion avec pondération optimisée

## 🏗️ Architecture

```
┌─────────────────┐
│   Streamlit UI  │  ← Interface utilisateur
└────────┬────────┘
         │
    HTTP Requests
         │
┌────────▼────────┐
│   FastAPI API   │  ← Serveur REST API
└────────┬────────┘
         │
    Load Models
         │
┌────────▼────────┐
│  ML/DL Models   │  ← Modèles de classification
│  - Text (SVM)   │
│  - Image (CNN)  │
│  - Multimodal   │
└─────────────────┘
```

##  Installation

### Prérequis
- Python 3.9+
- pip
- Git

### Clonage du repository
```bash
git clone https://github.com/votre-username/ecommerce-classification.git
cd ecommerce-classification
```

### Installations des dépendances
```bash
# Créationd d'un environnement virtuel
python -m venv venv
source venv/bin/activate  

# Installation des packages
pip install -r requirements.txt
```

## 🎮 Utilisation

### 1. Lancer l'API
```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

L'API sera accessible à `http://localhost:8000`
- Documentation Swagger : `http://localhost:8000/docs`
- Documentation ReDoc : `http://localhost:8000/redoc`

### 2. Lancer l'application Streamlit
```bash
cd app
streamlit run streamlit_app.py
```

L'application sera accessible à `http://localhost:8501`

### 3. Utilisation de l'API directement

#### Python
```python
import requests

# Classification texte
response = requests.post(
    "http://localhost:8000/predict/text",
    json={"text": "Montre digitale sport avec GPS"}
)
print(response.json())

# Classification image
with open("product.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict/image",
        files={"file": f}
    )
print(response.json())
```

#### cURL
```bash
# Texte
curl -X POST "http://localhost:8000/predict/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Ordinateur portable 15 pouces"}'

# Image
curl -X POST "http://localhost:8000/predict/image" \
     -F "file=@product.jpg"
```

## Structure du projet

```
Projet_NLP/
│
├── 📂 api/                          # API FastAPI
│   ├── main.py                      # Point d'entrée API
│   ├── models.py                    # Modèles Pydantic
│   ├── routes/                      # Routes organisées
│   │   ├── predict.py
│   │   └── health.py
│   └── config.py                    # Configuration
│
├── 📂 app/                          # Application Streamlit
│   ├── streamlit_app.py
│
├── 📂 models/                       # Modèles entraînés sauvegardés
│   ├── final_best_model.pkl              # Modèle texte SVM
│   ├── tfidf_vectorizer.pkl              # Vectorizer TF-IDF
│   ├── cnn_final.keras      # Modèle image CNN
│   ├── label_encoders.pkl           # Label encoder
│
├── 📂 notebooks/                    # Notebooks Jupyter
│   ├── n1_analyse_exploratoire.ipynb
│   ├── n2_preprocessing_featuring.ipynb
│   ├── n3_modelisation.ipynb
│   ├── n4_exploration_features_clustering_images.ipynb
│   └── n5_deep_learning_classification_images.ipynb
│
├── 📂 Data/                         # Données
│   ├── Flipkart/                         
│   │   └── flipkart_com-ecommerce_sample_1050.csv    # Données brutes
│   │     └── images/      # Images produits
│   └── processed/                   # Données prétraitées                     
│
│
├── 📄 requirements.txt              # Dépendances Python
├── 📄 Dockerfile                    # Configuration Docker
├── 📄 docker-compose.yml            # Docker Compose
├── 📄 .env.example                  # Variables d'environnement
├── 📄 .gitignore                    # Fichiers à ignorer
├── 📄 LICENSE                       # Licence
└── 📄 README.md                     # Ce fichier
```

## 🤖 Modèles

### Modèle Texte (TF-IDF + SVM)

**Architecture :**
```
Input Text
    ↓
[Preprocessing]
    ↓
[TF-IDF Vectorization]
    ↓
[SVM Classifier (RBF kernel)]
    ↓
Output (7 classes)
```

**Performances :**
- Accuracy : **94.94%**
- F1-Score : **0.949**
- Precision : **0.94**
- Recall : **0.9494**

### Modèle Image (Deep Learning)

**Architecture ResNet50 :**
```
Input Image (224x224x3)
    ↓
[ResNet50 base (frozen layers)]
    ↓
[GlobalAveragePooling2D]
    ↓
[Dense(512) + BatchNorm + Dropout(0.5)]
    ↓
[Dense(256) + BatchNorm + Dropout(0.4)]
    ↓
[Dense(7, softmax)]
```

**Performances :**
- Accuracy : **0.6250**
- F1-Score : **0.6202**
- Training : 50 epochs with early stopping
- Data augmentation : rotation, flip, zoom, shear

### Fusion Multimodale

**Stratégie : Late Fusion**
```python
P_final = α × P_text + (1-α) × P_image

où α = 0.6 (optimisé)
```

## 🔌 API

### Endpoints disponibles

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Informations API |
| GET | `/health` | Health check |
| GET | `/categories` | Liste catégories |
| POST | `/predict/text` | Classification texte |
| POST | `/predict/image` | Classification image |
| POST | `/predict/multimodal` | Classification combinée |

### Exemples de réponse

```json
{
  "predicted_class": "Watches",
  "confidence": 0.8523,
  "probabilities": {
    "Baby Care": 0.0234,
    "Beauty and Personal Care": 0.0456,
    "Computers": 0.0123,
    "Home Decor & Festive Needs": 0.0289,
    "Home Furnishing": 0.0156,
    "Kitchen & Dining": 0.0219,
    "Watches": 0.8523
  }
}
```

## 📊 Dashboard

Le dashboard analytique offre :

### KPIs
- 📈 Nombre total de prédictions
- 🎯 Confiance moyenne
- 🏆 Catégorie dominante
- 📊 Modes utilisés

### Visualisations
- 🥧 Distribution des catégories (pie chart)
- 📊 Confiance par catégorie (bar chart)
- ⏱️ Évolution temporelle (line chart)
- 📋 Tableau détaillé de l'historique

### Fonctionnalités
- Export CSV de l'historique
- Filtres interactifs
- Graphiques Plotly interactifs

## ☁️ Déploiement

### Option 1: Streamlit Cloud

```bash
# 1. Push sur GitHub
git push origin main

# 2. Se connecter à https://share.streamlit.io
# 3. Créer une nouvelle app
# 4. Pointer vers app/streamlit_app.py
```

### Option 2: Docker

```bash
# Build l'image
docker-compose build

# Lancer les services
docker-compose up -d

# Services disponibles:
# - API: http://localhost:8000
# - App: http://localhost:8501
```

### Option 3: Heroku

```bash
# 1. Créer une app Heroku
heroku create mon-app-classification

# 2. Déployer
git push heroku main

# 3. Ouvrir l'app
heroku open
```

### Option 4: AWS / GCP / Azure

Voir la documentation de déploiement dans `docs/deployment.md`

## 📈 Résultats

### Comparaison des modalités

| Modalité | Modèle | Accuracy | F1-Score | Temps inférence |
|----------|--------|----------|----------|-----------------|
| **Texte** | SVM (TF-IDF) | **94.94%** | **0.949** | ~10ms |
| **Image** | CNN | 62.50% | 0.6202 | ~50ms |
| **Multimodal** | Late Fusion | XX.X% | 0.XXX | ~60ms |

### Matrice de confusion (Texte)

```
                    Predicted
                BC  BP  CO  HD  HF  KD  WA
Actual    BC   [145   2   0   1   1   0   1]
          BP   [  1 143   0   2   2   1   1]
          CO   [  0   0 148   0   1   1   0]
          HD   [  2   1   0 141   3   3   0]
          HF   [  1   2   1   2 140   4   0]
          KD   [  0   1   1   3   3 142   0]
          WA   [  0   0   0   0   0   0 150]
```

## 🤝 Contributions

Les contributions sont les bienvenues ! 

### Comment contribuer

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push sur la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

### Guidelines

- Suivre PEP 8 pour le code Python
- Ajouter des tests pour les nouvelles fonctionnalités
- Mettre à jour la documentation
- Utiliser des messages de commit clairs

## 📝 TODO

- [ ] Ajouter support de nouvelles catégories
- [ ] Implémenter BERT pour le texte
- [ ] Tester Vision Transformer pour les images
- [ ] Ajouter authentification API
- [ ] Créer dashboard admin
- [ ] Implémenter A/B testing
- [ ] Ajouter monitoring (Prometheus/Grafana)
- [ ] Créer documentation API complète

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👥 Auteurs

- **Awa GUEYE** - *Travail initial* - [GitHub](https://github.com/awa-gueye)

## 🙏 Remerciements

- Dataset : [Flipkart Products](https://www.kaggle.com/datasets/...)
- Frameworks : TensorFlow, Scikit-learn, Streamlit, FastAPI
- Inspiration : Projets e-commerce et classification multimodale

---

<div align="center">

Made with ❤️ using Python, TensorFlow & Streamlit

[⬆ Retour en haut](#Projet_NLP)

</div>

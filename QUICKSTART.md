# GUIDE RAPIDE - CORRECTION COMPLÈTE

## Vos Fichiers de Modèles

```
models/
├── final_best_model.pkl      ← Modèle SVM texte (94.9%)
├── tfidf_vectorizer.pkl      ← Vectorizer TF-IDF
├── cnn_final.keras           ← Modèle CNN images (62.5%)
└── label_enconders.pkl       ← Label encoder (note: typo dans nom)
```

---

## ÉTAPE 1 : Placer les Modèles

### Localiser vos modèles

Ils sont probablement dans :
- Le dossier de vos notebooks
- Un dossier `results/` ou `outputs/`
- Là où vous avez sauvegardé après entraînement

### Copier dans le bon dossier

**Windows (PowerShell) :**
```powershell
# Adapter les chemins selon votre emplacement
cd ecommerce_classification_project

# Copier les modèles
copy "C:\chemin\vers\final_best_model.pkl" models\
copy "C:\chemin\vers\tfidf_vectorizer.pkl" models\
copy "C:\chemin\vers\cnn_final.keras" models\
copy "C:\chemin\vers\label_enconders.pkl" models\
```

**Linux/Mac :**
```bash
cd ecommerce_classification_project

# Copier les modèles
cp ~/chemin/vers/final_best_model.pkl models/
cp ~/chemin/vers/tfidf_vectorizer.pkl models/
cp ~/chemin/vers/cnn_final.keras models/
cp ~/chemin/vers/label_enconders.pkl models/
```

### Vérifier le placement

```bash
# Lancer le script de vérification
python check_models.py
```

**Résultat attendu :**
```
✅ Modèles trouvés: 4/4
✅ Mode COMPLET : Texte ✅ + Images ✅
```

---

## ÉTAPE 2 : Lancer l'API Corrigée

```bash
cd api
uvicorn main:app --reload
```

**Vérifiez les logs au démarrage :**
```
======================================================================
🚀 DÉMARRAGE DE L'API
======================================================================
🔄 Tentative de chargement des modèles...
✅ Modèles texte chargés
✅ Modèles image chargés
======================================================================
✅ API PRÊTE - Mode: FULL
======================================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**✅ Si vous voyez "Mode: FULL" → Succès !**

---

## ÉTAPE 3 : Lancer l'Interface V2 Moderne

**Terminal séparé :**
```bash
cd app
streamlit run streamlit_app_v2.py
```

**Ouvrir :** http://localhost:8501

---

## ÉTAPE 4 : Tester

### Test 1 : Vérifier l'API

```bash
curl http://localhost:8000/health
```

**Réponse attendue :**
```json
{
  "status": "healthy",
  "mode": "full",
  "text_model_loaded": true,
  "image_model_loaded": true
}
```

### Test 2 : Classification Texte

```bash
curl -X POST "http://localhost:8000/predict/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Soft and highly absorbent baby diapers designed to keep your baby dry and comfortable all day and night"}'
```

**Réponse attendue :**
```json
{
  "predicted_class": "Baby Care",
  "confidence": 0.95,
  "probabilities": {...},
  "source": "ml_model"  // ← IMPORTANT: doit être "ml_model"
}
```

### Test 3 : Interface Web

1. Ouvrir http://localhost:8501
2. Onglet "🎯 Classification"
3. Mode "📝 Texte"
4. Entrer : "Soft and highly absorbent baby diapers..."
5. Cliquer "🚀 CLASSIFIER"

**Résultat attendu :**
- Catégorie : **Baby Care** 👶
- Confiance : **> 90%**
- Couleur : **Verte** (haute confiance)
- Source : **ml_model**

---

## Nouvelle Interface V2 - Caractéristiques

### Design Moderne
✅ Palette professionnelle (bleu/violet dégradé)
✅ Layout spacieux et aéré
✅ Cartes avec ombres douces
✅ Animations fluides

### Couleurs Adaptatives
- 🟢 **Vert** : Confiance > 80%
- 🟡 **Orange** : Confiance 60-80%
- 🔴 **Rouge** : Confiance < 60%

### Dashboard Amélioré
✅ 4 KPIs modernes
✅ Pie chart interactif (Plotly)
✅ Bar chart avec gradient
✅ Tableau filtrable
✅ Export CSV

### Feedback Visuel
✅ Affichage de la source (ml_model vs simulation)
✅ Indicateur de mode API
✅ Messages d'erreur clairs

---

## Résolution des Problèmes

### Problème : "Modèle texte non chargé"

**Diagnostic :**
```bash
# Vérifier les fichiers
ls -la models/

# Devrait afficher :
# final_best_model.pkl
# tfidf_vectorizer.pkl
# cnn_final.keras
# label_enconders.pkl
```

**Solution :**
1. Vérifier que les 4 fichiers sont présents
2. Vérifier les permissions (lecture)
3. Relancer l'API

### Problème : Classifications aléatoires

**Cause :** Mode simulation actif

**Vérification :**
```bash
curl http://localhost:8000/health | grep mode
```

**Si affiche "simulation" :**
- Modèles pas chargés
- Voir solution ci-dessus

### Problème : Interface pas moderne

**Vérification :**
```bash
# Assurez-vous d'utiliser la V2
cd app
streamlit run streamlit_app_v2.py
```

**PAS** `streamlit_app.py` (version 1)

---

## Comparaison Avant/Après

| Aspect | Avant ❌ | Après ✅ |
|--------|---------|---------|
| **Modèles** | Non chargés | Chargés automatiquement |
| **Prédictions** | Aléatoires | ML réel (94.9% texte) |
| **Interface** | Basique | Ultra-moderne |
| **Couleurs** | Criardes | Professionnelles |
| **Layout** | Serré | Spacieux |
| **Dashboard** | Basique | Analytics Pro |
| **Source** | Cachée | Affichée |
| **Feedback** | Minimal | Visuel + Couleurs |

---

##  Checklist Finale

- [ ] **Modèles copiés** dans `models/`
- [ ] **Script de vérification** : `python check_models.py` → 4/4 ✅
- [ ] **API lancée** : `uvicorn main:app --reload`
- [ ] **Mode API** : "full" (vérifié avec `/health`)
- [ ] **Interface V2** : `streamlit run streamlit_app_v2.py`
- [ ] **Test texte** : "baby diapers" → Baby Care ✅
- [ ] **Source** : "ml_model" (pas simulation)
- [ ] **Confiance** : > 90% pour textes clairs
- [ ] **Design moderne** : Couleurs douces, layout aéré ✅

---

## Aide Rapide

### Commandes Essentielles

```bash
# Vérifier modèles
python check_models.py

# Terminal 1 - API
cd api
uvicorn main:app --reload

# Terminal 2 - Interface
cd app
streamlit run streamlit_app_v2.py

# Test santé API
curl http://localhost:8000/health
```

### Fichiers Clés

| Fichier | Usage |
|---------|-------|
| `api/main.py` | ✅ API corrigée (UTILISER) |
| `app/streamlit_app_v2.py` | ✅ Interface moderne (UTILISER) |
| `check_models.py` | 🔍 Vérification modèles |
| `models/final_best_model.pkl` | 📦 SVM texte |
| `models/tfidf_vectorizer.pkl` | 📦 Vectorizer |
| `models/cnn_final.keras` | 📦 CNN images |
| `models/label_enconders.pkl` | 📦 Encoder |

---

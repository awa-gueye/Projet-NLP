"""
FastAPI - API de classification de produits e-commerce (VERSION CORRIGÉE)
Chargement automatique des modèles avec gestion d'erreurs
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn
import numpy as np
from PIL import Image
import io
import joblib
import logging
from pathlib import Path

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemins des modèles (ADAPTÉS À VOS FICHIERS)
MODELS_DIR = Path("../models")
TEXT_MODEL_PATH = MODELS_DIR / "final_best_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
IMAGE_MODEL_PATH = MODELS_DIR / "cnn_final.keras"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoders.pkl" 

# Initialisation FastAPI
app = FastAPI(
    title="E-commerce Product Classification API",
    description="API de classification de produits à partir de texte ou images",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CATÉGORIES
# ============================================================================

CATEGORIES = [
    "Baby Care",
    "Beauty and Personal Care",
    "Computers",
    "Home Decor & Festive Needs",
    "Home Furnishing",
    "Kitchen & Dining",
    "Watches"
]

# ============================================================================
# GESTIONNAIRE DE MODÈLES AMÉLIORÉ
# ============================================================================

class ModelManager:
    """Gestionnaire de chargement des modèles avec fallback intelligent"""
    
    def __init__(self):
        self.text_model = None
        self.text_vectorizer = None
        self.image_model = None
        self.label_encoder = None
        self.mode = "unknown"
        
    def load_all_models(self):
        """Charger tous les modèles disponibles"""
        logger.info("🔄 Tentative de chargement des modèles...")
        
        # Charger modèles texte
        text_loaded = self.load_text_models()
        
        # Charger modèles image
        image_loaded = self.load_image_models()
        
        # Déterminer le mode
        if text_loaded and image_loaded:
            self.mode = "full"
            logger.info("✅ Mode COMPLET : Texte + Images")
        elif text_loaded:
            self.mode = "text_only"
            logger.info("✅ Mode TEXTE uniquement")
        elif image_loaded:
            self.mode = "image_only"
            logger.info("✅ Mode IMAGE uniquement")
        else:
            self.mode = "simulation"
            logger.warning("⚠️ Mode SIMULATION : Aucun modèle chargé")
        
        return self.mode
    
    def load_text_models(self):
        """Charger les modèles texte"""
        try:
            if TEXT_MODEL_PATH.exists() and VECTORIZER_PATH.exists():
                self.text_model = joblib.load(TEXT_MODEL_PATH)
                self.text_vectorizer = joblib.load(VECTORIZER_PATH)
                logger.info("✅ Modèles texte chargés")
                return True
            else:
                logger.warning(f"⚠️ Fichiers modèles texte non trouvés:")
                logger.warning(f"   {TEXT_MODEL_PATH}: {TEXT_MODEL_PATH.exists()}")
                logger.warning(f"   {VECTORIZER_PATH}: {VECTORIZER_PATH.exists()}")
                return False
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèles texte: {e}")
            return False
    
    def load_image_models(self):
        """Charger les modèles image"""
        try:
            # Importer TensorFlow seulement si nécessaire
            import tensorflow as tf
            from tensorflow import keras
            
            if IMAGE_MODEL_PATH.exists() and LABEL_ENCODER_PATH.exists():
                self.image_model = keras.models.load_model(IMAGE_MODEL_PATH)
                self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
                logger.info("✅ Modèles image chargés")
                return True
            else:
                logger.warning(f"⚠️ Fichiers modèles image non trouvés:")
                logger.warning(f"   {IMAGE_MODEL_PATH}: {IMAGE_MODEL_PATH.exists()}")
                logger.warning(f"   {LABEL_ENCODER_PATH}: {LABEL_ENCODER_PATH.exists()}")
                return False
        except ImportError:
            logger.warning("⚠️ TensorFlow non installé - Modèles image désactivés")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèles image: {e}")
            return False
    
    def predict_text(self, text: str) -> Dict:
        """Prédiction texte avec fallback intelligent"""
        
        # Si modèle disponible, utiliser ML
        if self.text_model and self.text_vectorizer:
            try:
                # Vectorisation
                X = self.text_vectorizer.transform([text])
                
                # Prédiction
                prediction = self.text_model.predict(X)[0]
                
                # Probabilités
                if hasattr(self.text_model, 'predict_proba'):
                    probas = self.text_model.predict_proba(X)[0]
                else:
                    # Si pas de proba, créer artificielle
                    probas = np.zeros(len(CATEGORIES))
                    probas[prediction] = 0.95
                    # Distribuer le reste
                    remaining = 0.05 / (len(CATEGORIES) - 1)
                    for i in range(len(CATEGORIES)):
                        if i != prediction:
                            probas[i] = remaining
                
                # Récupérer le nom de la catégorie
                predicted_class = CATEGORIES[prediction]
                confidence = float(probas[prediction])
                
                # Formatter les probabilités
                probabilities = {
                    cat: float(probas[i]) 
                    for i, cat in enumerate(CATEGORIES)
                }
                
                return {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "source": "ml_model"
                }
                
            except Exception as e:
                logger.error(f"Erreur prédiction ML texte: {e}")
                # Fallback sur simulation
                return self._simulate_text_prediction(text)
        
        # Fallback : simulation intelligente
        return self._simulate_text_prediction(text)
    
    def predict_image(self, image: Image.Image) -> Dict:
        """Prédiction image avec fallback"""
        
        # Si modèle disponible, utiliser DL
        if self.image_model and self.label_encoder:
            try:
                # Prétraitement
                img = image.resize((224, 224))
                img_array = np.array(img)
                
                # Vérifier que c'est RGB
                if len(img_array.shape) == 2:  # Grayscale
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:, :, :3]
                
                # Normalisation
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Prédiction
                probas = self.image_model.predict(img_array, verbose=0)[0]
                
                # Résultats
                predicted_idx = np.argmax(probas)
                predicted_class = self.label_encoder.classes_[predicted_idx]
                confidence = float(probas[predicted_idx])
                
                probabilities = {
                    self.label_encoder.classes_[i]: float(probas[i])
                    for i in range(len(probas))
                }
                
                return {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "source": "dl_model"
                }
                
            except Exception as e:
                logger.error(f"Erreur prédiction DL image: {e}")
                return self._simulate_image_prediction()
        
        # Fallback : simulation
        return self._simulate_image_prediction()
    
    def _simulate_text_prediction(self, text: str) -> Dict:
        """Simulation intelligente basée sur mots-clés"""
        text_lower = text.lower()
        
        # Dictionnaire de mots-clés par catégorie
        keywords = {
            "Baby Care": ['baby', 'infant', 'diaper', 'newborn', 'toddler', 'bébé', 'nouveau-né'],
            "Beauty and Personal Care": ['cosmetic', 'makeup', 'beauty', 'lipstick', 'perfume', 'skincare', 'lotion'],
            "Computers": ['computer', 'laptop', 'gaming', 'pc', 'keyboard', 'mouse', 'monitor', 'processor'],
            "Home Decor & Festive Needs": ['decoration', 'festive', 'ornament', 'vase', 'candle', 'frame'],
            "Home Furnishing": ['furniture', 'sofa', 'bed', 'chair', 'table', 'cushion', 'curtain'],
            "Kitchen & Dining": ['kitchen', 'dining', 'cookware', 'pan', 'utensil', 'plate', 'bowl'],
            "Watches": ['watch', 'timepiece', 'wristwatch', 'clock', 'chronograph']
        }
        
        # Compter les matches par catégorie
        scores = {}
        for category, words in keywords.items():
            score = sum(1 for word in words if word in text_lower)
            scores[category] = score
        
        # Si au moins un match
        if max(scores.values()) > 0:
            predicted = max(scores, key=scores.get)
            conf = min(0.95, 0.60 + (scores[predicted] * 0.10))
        else:
            # Aucun match : prédiction aléatoire faible confiance
            predicted = np.random.choice(CATEGORIES)
            conf = 0.30
        
        # Distribuer probabilités
        probabilities = {}
        remaining = 1.0 - conf
        
        for cat in CATEGORIES:
            if cat == predicted:
                probabilities[cat] = conf
            else:
                probabilities[cat] = remaining / (len(CATEGORIES) - 1)
        
        return {
            "predicted_class": predicted,
            "confidence": float(conf),
            "probabilities": probabilities,
            "source": "simulation"
        }
    
    def _simulate_image_prediction(self) -> Dict:
        """Simulation aléatoire pour images"""
        predicted = np.random.choice(CATEGORIES)
        conf = np.random.uniform(0.30, 0.60)
        
        probabilities = {}
        remaining = 1.0 - conf
        
        for cat in CATEGORIES:
            if cat == predicted:
                probabilities[cat] = conf
            else:
                probabilities[cat] = remaining / (len(CATEGORIES) - 1)
        
        return {
            "predicted_class": predicted,
            "confidence": float(conf),
            "probabilities": probabilities,
            "source": "simulation"
        }

# Instance globale
model_manager = ModelManager()

# ============================================================================
# MODÈLES PYDANTIC
# ============================================================================

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    source: str = "unknown"

class HealthResponse(BaseModel):
    status: str
    mode: str
    text_model_loaded: bool
    image_model_loaded: bool

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "API de classification de produits e-commerce",
        "version": "1.0.0",
        "mode": model_manager.mode,
        "endpoints": {
            "health": "/health",
            "predict_text": "/predict/text",
            "predict_image": "/predict/image",
            "predict_multimodal": "/predict/multimodal",
            "categories": "/categories"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "mode": model_manager.mode,
        "text_model_loaded": model_manager.text_model is not None,
        "image_model_loaded": model_manager.image_model is not None
    }

@app.get("/categories")
async def get_categories():
    """Liste des catégories"""
    return {
        "categories": CATEGORIES,
        "count": len(CATEGORIES)
    }

@app.post("/predict/text", response_model=PredictionResponse)
async def predict_from_text(input_data: TextInput):
    """Prédiction à partir de texte"""
    try:
        result = model_manager.predict_text(input_data.text)
        return result
    except Exception as e:
        logger.error(f"Erreur prédiction texte: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/image", response_model=PredictionResponse)
async def predict_from_image(file: UploadFile = File(...)):
    """Prédiction à partir d'image"""
    try:
        # Vérifier type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Le fichier doit être une image")
        
        # Charger image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convertir en RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Prédiction
        result = model_manager.predict_image(image)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur prédiction image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/multimodal")
async def predict_multimodal(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """Prédiction multimodale"""
    if not text and not file:
        raise HTTPException(status_code=400, detail="Au moins un input requis")
    
    try:
        predictions = []
        weights = []
        sources = []
        
        # Prédiction texte
        if text:
            text_pred = model_manager.predict_text(text)
            predictions.append(text_pred["probabilities"])
            weights.append(0.7)  # Poids plus élevé pour texte (meilleur modèle)
            sources.append(text_pred["source"])
        
        # Prédiction image
        if file:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_pred = model_manager.predict_image(image)
            predictions.append(image_pred["probabilities"])
            weights.append(0.3)
            sources.append(image_pred["source"])
        
        # Fusion
        if len(predictions) == 1:
            combined_probs = predictions[0]
        else:
            weights = np.array(weights) / sum(weights)
            combined_probs = {}
            for cat in CATEGORIES:
                probs = [pred.get(cat, 0.0) for pred in predictions]
                combined_probs[cat] = float(np.average(probs, weights=weights))
        
        predicted_class = max(combined_probs.items(), key=lambda x: x[1])[0]
        confidence = combined_probs[predicted_class]
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": combined_probs,
            "source": f"multimodal({'+'.join(sources)})",
            "mode": "text+image" if len(predictions) == 2 else ("text" if text else "image")
        }
        
    except Exception as e:
        logger.error(f"Erreur prédiction multimodale: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Charger les modèles au démarrage"""
    logger.info("=" * 70)
    logger.info("🚀 DÉMARRAGE DE L'API")
    logger.info("=" * 70)
    
    # Charger les modèles
    mode = model_manager.load_all_models()
    
    logger.info("=" * 70)
    logger.info(f"✅ API PRÊTE - Mode: {mode.upper()}")
    logger.info("=" * 70)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
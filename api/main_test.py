"""
API FastAPI - Version TEST (sans modèles ML)
Pour tester l'API avant d'avoir les modèles entraînés
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import numpy as np
from datetime import datetime

# Initialisation FastAPI
app = FastAPI(
    title="E-commerce Product Classification API (TEST MODE)",
    description="API de test - Classification simulée",
    version="1.0.0-test"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Catégories
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
# MODÈLES PYDANTIC
# ============================================================================

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    mode: str = "simulation"

class HealthResponse(BaseModel):
    status: str
    mode: str
    timestamp: str

# ============================================================================
# FONCTIONS DE SIMULATION
# ============================================================================

def simulate_text_prediction(text: str) -> Dict:
    """Simule une prédiction basée sur des mots-clés"""
    
    text_lower = text.lower()
    
    # Détection basique par mots-clés
    if any(word in text_lower for word in ['baby', 'bébé', 'infant', 'enfant', 'nouveau-né']):
        predicted = "Baby Care"
        conf = 0.85
    elif any(word in text_lower for word in ['cosmetic', 'makeup', 'beauty', 'parfum', 'lipstick']):
        predicted = "Beauty and Personal Care"
        conf = 0.82
    elif any(word in text_lower for word in ['computer', 'laptop', 'ordinateur', 'pc', 'gaming']):
        predicted = "Computers"
        conf = 0.88
    elif any(word in text_lower for word in ['decor', 'decoration', 'festive', 'ornament']):
        predicted = "Home Decor & Festive Needs"
        conf = 0.75
    elif any(word in text_lower for word in ['furniture', 'sofa', 'bed', 'meuble', 'canapé']):
        predicted = "Home Furnishing"
        conf = 0.80
    elif any(word in text_lower for word in ['kitchen', 'dining', 'cuisine', 'ustensil', 'pan']):
        predicted = "Kitchen & Dining"
        conf = 0.83
    elif any(word in text_lower for word in ['watch', 'montre', 'timepiece', 'clock']):
        predicted = "Watches"
        conf = 0.90
    else:
        # Prédiction aléatoire
        predicted = np.random.choice(CATEGORIES)
        conf = np.random.uniform(0.4, 0.7)
    
    # Générer probabilités
    probs = {}
    remaining = 1.0 - conf
    
    for cat in CATEGORIES:
        if cat == predicted:
            probs[cat] = conf
        else:
            probs[cat] = remaining / (len(CATEGORIES) - 1)
    
    return {
        "predicted_class": predicted,
        "confidence": float(conf),
        "probabilities": probs
    }

def simulate_image_prediction() -> Dict:
    """Simule une prédiction d'image aléatoire"""
    
    predicted = np.random.choice(CATEGORIES)
    conf = np.random.uniform(0.5, 0.85)
    
    # Probabilités
    probs = {}
    remaining = 1.0 - conf
    
    for cat in CATEGORIES:
        if cat == predicted:
            probs[cat] = conf
        else:
            probs[cat] = remaining / (len(CATEGORIES) - 1)
    
    return {
        "predicted_class": predicted,
        "confidence": float(conf),
        "probabilities": probs
    }

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "API de classification (MODE TEST - Simulation)",
        "version": "1.0.0-test",
        "mode": "simulation",
        "note": "Cette version simule les prédictions. Installez les modèles ML pour la version complète.",
        "endpoints": {
            "health": "/health",
            "predict_text": "/predict/text",
            "predict_image": "/predict/image",
            "categories": "/categories"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "mode": "simulation",
        "timestamp": datetime.now().isoformat()
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
    """
    Prédiction de catégorie à partir d'une description textuelle
    MODE SIMULATION - Détection par mots-clés
    """
    try:
        result = simulate_text_prediction(input_data.text)
        result["mode"] = "simulation-text"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/image", response_model=PredictionResponse)
async def predict_from_image(file: UploadFile = File(...)):
    """
    Prédiction de catégorie à partir d'une image
    MODE SIMULATION - Prédiction aléatoire
    """
    try:
        # Vérifier le type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Le fichier doit être une image")
        
        result = simulate_image_prediction()
        result["mode"] = "simulation-image"
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/multimodal")
async def predict_multimodal(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Prédiction combinée texte + image
    MODE SIMULATION
    """
    if not text and not file:
        raise HTTPException(status_code=400, detail="Au moins un input requis")
    
    try:
        predictions = []
        weights = []
        
        if text:
            text_pred = simulate_text_prediction(text)
            predictions.append(text_pred["probabilities"])
            weights.append(0.6)
        
        if file:
            image_pred = simulate_image_prediction()
            predictions.append(image_pred["probabilities"])
            weights.append(0.4)
        
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
            "mode": "simulation-multimodal"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
async def info():
    """Informations sur le mode de l'API"""
    return {
        "mode": "TEST/SIMULATION",
        "description": "Cette API simule les prédictions pour tester l'infrastructure",
        "text_prediction": "Détection par mots-clés simples",
        "image_prediction": "Prédictions aléatoires",
        "to_production": [
            "1. Installer TensorFlow : pip install tensorflow>=2.15.0",
            "2. Placer les modèles dans models/",
            "3. Utiliser api/main.py (version complète)"
        ],
        "categories": CATEGORIES
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Démarrage API TEST MODE")
    print("⚠️  Mode simulation - Pas de modèles ML requis")
    print("📚 Documentation : http://localhost:8000/docs")
    
    uvicorn.run(
        "main_test:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

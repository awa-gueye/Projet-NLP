"""
Application Streamlit Niveau Entreprise
Design Corporate: Bleu foncé + Blanc + Doré
Thème clair/sombre + Images professionnelles + Dashboard Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import requests
import io
from datetime import datetime, timedelta
import json
import base64
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Place de marché",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# Enterprise Product Classification System\nPowered by AI"
    }
)

# ============================================================================
# CONSTANTES ET CONFIGURATION
# ============================================================================

API_URL = "http://localhost:8000"

CATEGORIES = [
    "Baby Care",
    "Beauty and Personal Care",
    "Computers",
    "Home Decor & Festive Needs",
    "Home Furnishing",
    "Kitchen & Dining",
    "Watches"
]

# URLs d'images professionnelles (Unsplash/Pexels - libres de droits)
CATEGORY_IMAGES = {
    "Baby Care": "https://images.unsplash.com/photo-1515488042361-ee00e0ddd4e4?w=400",
    "Beauty and Personal Care": "https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=400",
    "Computers": "https://images.unsplash.com/photo-1593642632823-8f785ba67e45?w=400",
    "Home Decor & Festive Needs": "https://images.unsplash.com/photo-1513506003901-1e6a229e2d15?w=400",
    "Home Furnishing": "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=400",
    "Kitchen & Dining": "https://images.unsplash.com/photo-1556911220-bff31c812dba?w=400",
    "Watches": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400"
}

# Thèmes Corporate
THEMES = {
    "light": {
        "primary": "#0F4C81",      # Bleu corporate foncé
        "secondary": "#1E88E5",    # Bleu clair
        "accent": "#D4AF37",       # Doré
        "background": "#FFFFFF",   # Blanc
        "surface": "#F8F9FA",      # Gris très clair
        "text": "#212529",         # Noir texte
        "text_secondary": "#6C757D", # Gris texte
        "border": "#DEE2E6",       # Bordure
        "success": "#28A745",
        "warning": "#FFC107",
        "error": "#DC3545"
    },
    "dark": {
        "primary": "#1E88E5",      # Bleu corporate
        "secondary": "#42A5F5",    # Bleu clair
        "accent": "#FFD700",       # Doré brillant
        "background": "#0A1929",   # Bleu très foncé
        "surface": "#1A2332",      # Bleu foncé surface
        "text": "#E3F2FD",         # Blanc cassé
        "text_secondary": "#90CAF9", # Bleu clair texte
        "border": "#2C3E50",       # Bordure foncée
        "success": "#4CAF50",
        "warning": "#FFB300",
        "error": "#EF5350"
    }
}

# ============================================================================
# SESSION STATE
# ============================================================================

if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

if 'history' not in st.session_state:
    st.session_state.history = []

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# ============================================================================
# FONCTIONS CSS DYNAMIQUES
# ============================================================================

def get_theme_colors():
    """Récupérer les couleurs du thème actuel"""
    return THEMES[st.session_state.theme]

def load_enterprise_css():
    """CSS Corporate professionnel avec thème dynamique"""
    colors = get_theme_colors()
    
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Montserrat:wght@600;700&display=swap');
    
    /* Reset et base */
    * {{
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    
    /* Main container */
    .main {{
        background: {colors['background']};
        color: {colors['text']};
    }}
    
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}
    
    /* Header Corporate */
    .enterprise-header {{
        background: linear-gradient(135deg, {colors['primary']} 0%, {colors['secondary']} 100%);
        padding: 2.5rem 3rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        margin: -2rem -3rem 3rem -3rem;
        position: relative;
        overflow: hidden;
    }}
    
    .enterprise-header::before {{
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, {colors['accent']}33 0%, transparent 70%);
        border-radius: 50%;
        transform: translate(30%, -30%);
    }}
    
    .header-content {{
        position: relative;
        z-index: 1;
    }}
    
    .company-logo {{
        font-size: 2rem;
        font-weight: 700;
        color: white;
        font-family: 'Montserrat', sans-serif;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }}
    
    .logo-icon {{
        width: 50px;
        height: 50px;
        background: {colors['accent']};
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }}
    
    .header-subtitle {{
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }}
    
    /* Theme Toggle */
    .theme-toggle {{
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        background: {colors['surface']};
        border: 2px solid {colors['border']};
        border-radius: 50px;
        padding: 0.5rem 1.2rem;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        color: {colors['text']};
        font-weight: 500;
    }}
    
    .theme-toggle:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        background: {colors['primary']};
        color: white;
    }}
    
    /* Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1rem;
        background: transparent;
        border-bottom: 2px solid {colors['border']};
        padding-bottom: 0;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border: none;
        border-bottom: 3px solid transparent;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        color: {colors['text_secondary']};
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        color: {colors['primary']};
        border-bottom-color: {colors['primary']}33;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: transparent;
        color: {colors['primary']};
        border-bottom-color: {colors['primary']};
    }}
    
    /* Category Cards */
    .category-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }}
    
   .category-card {{
    background: {colors['surface']};
    border: 2px solid {colors['border']};
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;
    position: relative;
    overflow: hidden;

    height: 260px;                /* 🔑 hauteur fixe */
    display: flex;                /* 🔑 flex */
    flex-direction: column;
    justify-content: space-between;
    }}

    
    .category-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.1);
        border-color: {colors['primary']};
    }}
    
    .category-card img {{
        width: 100%;
        height: 120px;
        object-fit: cover;
        border-radius: 12px;
    }}

    
    .category-name {{
        font-weight: 600;
        color: {colors['text']};
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }}
    
    /* Cards Enterprise */
    .enterprise-card {{
        background: {colors['surface']};
        border: 1px solid {colors['border']};
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }}
    
    .enterprise-card:hover {{
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        transform: translateY(-4px);
    }}
    
    .card-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {colors['primary']};
        margin-bottom: 1.5rem;
        font-family: 'Montserrat', sans-serif;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .card-title::before {{
        content: '';
        width: 4px;
        height: 24px;
        background: {colors['accent']};
        border-radius: 2px;
    }}
    
    /* Result Card */
    .result-card {{
        background: linear-gradient(135deg, {colors['primary']} 0%, {colors['secondary']} 100%);
        border-radius: 20px;
        padding: 3rem;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 12px 40px rgba(15,76,129,0.3);
        position: relative;
        overflow: hidden;
    }}
    
    .result-card::before {{
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, {colors['accent']}22 0%, transparent 70%);
    }}
    
    .result-content {{
        position: relative;
        z-index: 1;
    }}
    
    .result-image {{
        width: 150px;
        height: 150px;
        margin: 0 auto 1.5rem;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid white;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }}
    
    .result-category {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        font-family: 'Montserrat', sans-serif;
    }}
    
    .result-confidence {{
        font-size: 1.8rem;
        font-weight: 300;
        opacity: 0.95;
    }}
    
    .confidence-badge {{
        display: inline-block;
        padding: 0.5rem 1.5rem;
        background: {colors['accent']};
        color: {colors['primary']};
        border-radius: 50px;
        font-weight: 700;
        margin-top: 1rem;
        font-size: 1.1rem;
    }}
    
    /* Metrics */
    .metric-container {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }}
    
    .metric-card {{
        background: {colors['surface']};
        border: 2px solid {colors['border']};
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;

        height: 160px;             
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}

    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, {colors['primary']}, {colors['accent']});
    }}
    
    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }}
    
    .metric-value {{
    font-size: clamp(1.0rem, 1.2vw, 1.5rem);
    font-weight: 700;
    color: {colors['primary']};
    font-family: 'Montserrat', sans-serif;
    margin-bottom: 0.5rem;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
    }}

    
    .metric-label {{
    font-size: clamp(0.65rem, 0.9vw, 0.85rem);
    color: {colors['text_secondary']};
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
    }}  

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, {colors['primary']} 0%, {colors['secondary']} 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 3rem;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 6px 20px rgba(15,76,129,0.3);
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(15,76,129,0.4);
    }}
    
    /* Input Fields */
    .stTextArea textarea, .stTextInput input {{
        background: {colors['surface']};
        border: 2px solid {colors['border']};
        border-radius: 12px;
        padding: 1rem;
        color: {colors['text']};
        font-size: 1rem;
    }}
    
    .stTextArea textarea:focus, .stTextInput input:focus {{
        border-color: {colors['primary']};
        box-shadow: 0 0 0 3px {colors['primary']}22;
    }}
    
    /* File Uploader */
    .stFileUploader {{
        background: {colors['surface']};
        border: 2px dashed {colors['primary']};
        border-radius: 16px;
        padding: 2rem;
    }}
    
    /* Radio Buttons */
    .stRadio > div {{
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }}
    
    .stRadio label {{
        background: {colors['surface']};
        border: 2px solid {colors['border']};
        border-radius: 12px;
        padding: 1rem 2rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
    }}
    
    .stRadio label:hover {{
        border-color: {colors['primary']};
        background: {colors['primary']}11;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {colors['surface']};
        border-right: 2px solid {colors['border']};
    }}
    
    [data-testid="stSidebar"] .sidebar-content {{
        padding: 2rem 1rem;
    }}
    
    /* Alerts */
    .stAlert {{
        border-radius: 12px;
        border-left: 4px solid;
    }}
    
    /* Dataframe */
    .dataframe {{
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid {colors['border']};
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def toggle_theme():
    """Basculer entre thème clair et foncé"""
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

def check_api_health():
    """Vérifier l'état de l'API"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

def get_prediction(mode, text=None, image=None):
    """Obtenir une prédiction de l'API avec cache pour cohérence"""
    
    # Créer une clé de cache basée sur l'input
    cache_key = None
    if text:
        cache_key = f"text_{hash(text)}"
    elif image:
        # Hash de l'image
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        cache_key = f"image_{hash(img_bytes.getvalue())}"
    
    # Vérifier le cache
    if cache_key and st.session_state.last_prediction:
        if st.session_state.last_prediction.get('cache_key') == cache_key:
            return True, st.session_state.last_prediction['result']
    
    # Faire la prédiction
    try:
        if mode == "text":
            response = requests.post(
                f"{API_URL}/predict/text",
                json={"text": text},
                timeout=10
            )
        elif mode == "image":
            img_byte = io.BytesIO()
            image.save(img_byte, format='PNG')
            img_byte = img_byte.getvalue()
            
            response = requests.post(
                f"{API_URL}/predict/image",
                files={'file': ('image.png', img_byte, 'image/png')},
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            
            # Sauvegarder dans le cache
            st.session_state.last_prediction = {
                'cache_key': cache_key,
                'result': result
            }
            
            return True, result
        else:
            return False, {"error": response.text}
            
    except Exception as e:
        return False, {"error": str(e)}

def save_to_history(result, mode, description=""):
    """Sauvegarder dans l'historique"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    entry = {
        'timestamp': datetime.now(),
        'mode': mode,
        'description': description[:100],
        'predicted_class': result.get('predicted_class', 'Unknown'),
        'confidence': result.get('confidence', 0),
        'source': result.get('source', 'unknown')
    }
    
    st.session_state.history.insert(0, entry)

# ============================================================================
# COMPOSANTS UI
# ============================================================================

def render_header():
    """Header Corporate"""
    colors = get_theme_colors()
    
    theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
    theme_text = "Dark" if st.session_state.theme == 'light' else "Light"
    
    st.markdown(f"""
    <div class="enterprise-header">
        <div class="header-content">
            <div class="company-logo">
                <div class="logo-icon">🏢</div>
                <span>ENTERPRISE CLASSIFIER</span>
            </div>
            <div class="header-subtitle">
                Product Classification System
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Theme toggle button (utiliser un vrai bouton Streamlit)
    col1, col2, col3 = st.columns([6, 1, 1])
    with col3:
        if st.button(f"{theme_icon} {theme_text}", key="theme_toggle"):
            toggle_theme()
            st.rerun()

def render_api_status():
    """Statut API professionnel"""
    is_online, health_data = check_api_health()
    
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col2:
        if is_online:
            mode = health_data.get('mode', 'unknown')
            
            if mode == "full":
                st.success("🟢 **System Status:** Fully Operational")
            elif mode in ["text_only", "image_only"]:
                st.warning(f"🟡 **System Status:** Partial ({mode.replace('_', ' ').title()})")
            else:
                st.info("🟠 **System Status:** Demo Mode")
        else:
            st.error("🔴 **System Status:** Offline")

def render_categories():
    """Catégories avec images professionnelles"""
    st.markdown('<div class="card-title">📦 Available Product Categories</div>', unsafe_allow_html=True)
    
    # Créer une grille
    cols = st.columns(4)
    
    for idx, cat in enumerate(CATEGORIES):
        with cols[idx % 4]:
            img_url = CATEGORY_IMAGES.get(cat, "")
            
            st.markdown(f"""
            <div class="category-card">
                <img src="{img_url}" alt="{cat}" onerror="this.src='https://via.placeholder.com/400x300?text={cat.replace(' ', '+')}'">
                <div class="category-name">{cat}</div>
            </div>
            """, unsafe_allow_html=True)

def render_result(result):
    """Afficher résultat avec image"""
    cat = result['predicted_class']
    conf = result['confidence']
    source = result.get('source', 'unknown')
    
    img_url = CATEGORY_IMAGES.get(cat, "")
    
    # Déterminer le badge de confiance
    if conf > 0.8:
        badge_text = "High Confidence"
        badge_class = "success"
    elif conf > 0.6:
        badge_text = "Moderate Confidence"
        badge_class = "warning"
    else:
        badge_text = "Low Confidence"
        badge_class = "error"
    
    st.markdown(f"""
    <div class="result-card">
        <div class="result-content">
            <img src="{img_url}" class="result-image" alt="{cat}" onerror="this.src='https://via.placeholder.com/150?text={cat.replace(' ', '+')}">
            <div class="result-category">{cat}</div>
            <div class="result-confidence">Confidence: {conf*100:.1f}%</div>
            <div class="confidence-badge">{badge_text}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Graphique des probabilités
    probs = result['probabilities']
    df_probs = pd.DataFrame([
        {'Category': k, 'Probability': v} 
        for k, v in probs.items()
    ]).sort_values('Probability', ascending=True)
    
    colors_theme = get_theme_colors()
    
    fig = go.Figure(go.Bar(
        x=df_probs['Probability'],
        y=df_probs['Category'],
        orientation='h',
        marker=dict(
            color=df_probs['Probability'],
            colorscale=[[0, colors_theme['error']], [0.5, colors_theme['warning']], [1, colors_theme['success']]],
            line=dict(color=colors_theme['border'], width=1)
        ),
        text=[f'{v:.1%}' for v in df_probs['Probability']],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Probability Distribution",
        xaxis_title="Probability",
        yaxis_title="",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color=colors_theme['text']),
        showlegend=False,
        xaxis=dict(gridcolor=colors_theme['border']),
        yaxis=dict(gridcolor=colors_theme['border'])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Informations supplémentaires
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Category", cat)
    with col2:
        st.metric("Confidence Level", f"{conf*100:.1f}%")
    with col3:
        st.metric("Source", source.upper())

# ============================================================================
# TABS
# ============================================================================

def render_classification_tab():
    """Tab de classification"""
    st.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎯 Product Classification</div>', unsafe_allow_html=True)
    
    # Mode selection
    mode = st.radio(
        "Classification Mode:",
        ["📝 Text Description", "🖼️ Product Image", "🔗 Multimodal"],
        horizontal=True
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    result = None
    text_input = None
    uploaded_file = None
    
    # Input selon le mode
    st.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
    
    if "Text" in mode or "Multimodal" in mode:
        st.markdown("### 📝 Product Description")
        text_input = st.text_area(
            "Enter detailed product description:",
            placeholder="Example: Soft and highly absorbent baby diapers designed for newborns, gentle on sensitive skin...",
            height=150,
            help="Provide a detailed description for best results"
        )
    
    if "Image" in mode or "Multimodal" in mode:
        st.markdown("### 🖼️ Product Image")
        uploaded_file = st.file_uploader(
            "Upload high-quality product image:",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG (max 200MB)"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Bouton de classification
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        classify_button = st.button("🚀 CLASSIFY PRODUCT", use_container_width=True)
    
    if classify_button:
        # Validation
        if "Text" in mode and not text_input:
            st.warning("⚠️ Please enter a product description")
        elif "Image" in mode and not uploaded_file:
            st.warning("⚠️ Please upload a product image")
        elif "Multimodal" in mode and not text_input and not uploaded_file:
            st.warning("⚠️ Please provide either text or image")
        else:
            # Classification
            with st.spinner("🔄 Analyzing product..."):
                if "Text" in mode and "Multimodal" not in mode:
                    success, result = get_prediction("text", text=text_input)
                elif "Image" in mode and "Multimodal" not in mode:
                    success, result = get_prediction("image", image=image)
                else:
                    # Multimodal : privilégier le texte
                    if text_input:
                        success, result = get_prediction("text", text=text_input)
                    else:
                        success, result = get_prediction("image", image=image)
                
                if success:
                    save_to_history(result, mode.split()[0], text_input or "")
                else:
                    st.error(f"❌ Classification Error: {result.get('error', 'Unknown error')}")
    
    # Afficher le résultat
    if result and isinstance(result, dict) and 'predicted_class' in result:
        st.markdown("---")
        st.markdown('<div class="card-title">📊 Classification Results</div>', unsafe_allow_html=True)
        render_result(result)

def render_dashboard_tab():
    """Dashboard Analytics Enterprise"""
    st.markdown('<div class="card-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    
    if 'history' not in st.session_state or not st.session_state.history:
        st.info("📭 No classification data yet. Start classifying products to see analytics.")
        return
    
    history = st.session_state.history
    df = pd.DataFrame(history)
    
    colors = get_theme_colors()
    
    # KPIs
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Classifications</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_conf = df['confidence'].mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_conf:.1f}%</div>
            <div class="metric-label">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        top_cat = df['predicted_class'].mode()[0]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{top_cat[:15]}</div>
            <div class="metric-label">Top Category</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        high_conf = (df['confidence'] > 0.8).sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{high_conf}</div>
            <div class="metric-label">High Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution catégories
        cat_counts = df['predicted_class'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=cat_counts.index,
            values=cat_counts.values,
            hole=0.4,
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color=colors['background'], width=2)
            )
        )])
        
        fig.update_layout(
            title="Category Distribution",
            height=400,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=colors['text'])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confiance par catégorie
        avg_conf_cat = df.groupby('predicted_class')['confidence'].mean().sort_values(ascending=False)
        
        fig = go.Figure(go.Bar(
            x=avg_conf_cat.index,
            y=avg_conf_cat.values * 100,
            marker=dict(
                color=avg_conf_cat.values,
                colorscale=[[0, colors['error']], [0.5, colors['warning']], [1, colors['success']]],
                line=dict(color=colors['border'], width=1)
            ),
            text=[f'{v:.1f}%' for v in avg_conf_cat.values * 100],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Average Confidence by Category",
            xaxis_title="Category",
            yaxis_title="Confidence (%)",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=colors['text']),
            xaxis=dict(gridcolor=colors['border']),
            yaxis=dict(gridcolor=colors['border'])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Timeline
    st.markdown("### ⏱️ Classification Timeline")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_timeline = df.set_index('timestamp').resample('H')['confidence'].agg(['mean', 'count']).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df_timeline['timestamp'],
            y=df_timeline['mean'] * 100,
            name="Avg Confidence",
            line=dict(color=colors['primary'], width=3)
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Bar(
            x=df_timeline['timestamp'],
            y=df_timeline['count'],
            name="Classifications",
            marker=dict(color=colors['accent'], opacity=0.6)
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=colors['text']),
        xaxis=dict(gridcolor=colors['border']),
        yaxis=dict(gridcolor=colors['border']),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Confidence (%)", secondary_y=False)
    fig.update_yaxes(title_text="Count", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau détaillé
    st.markdown("### 📋 Classification History")
    
    display_df = df[['timestamp', 'mode', 'predicted_class', 'confidence', 'source']].copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
    display_df.columns = ['Date/Time', 'Mode', 'Category', 'Confidence', 'Source']
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Export
    col1, col2, col3 = st.columns([3, 1, 1])
    with col3:
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Export CSV",
            csv,
            f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )

def render_about_tab():
    """À propos Enterprise"""
    st.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
    
    st.markdown("## 🏢 About Enterprise Classifier")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 System Overview
        
        Enterprise-grade AI classification system designed for:
        - **E-commerce platforms**
        - **Retail management**
        - **Inventory categorization**
        - **Product data enrichment**
        
        ### 🤖 Technology Stack
        
        - **Machine Learning**: SVM with TF-IDF (94.9% accuracy)
        - **Deep Learning**: CNN (ongoing optimization)
        - **Framework**: Python 3.12 + TensorFlow
        - **API**: FastAPI with RESTful endpoints
        - **Frontend**: Streamlit Enterprise
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Performance Metrics
        
        | Model | Accuracy | F1-Score |
        |-------|----------|----------|
        | Text Classification | 94.9% | 0.949 |
        | Image Classification | 62.5% | 0.620 |
        
        ### 🎨 Features
        
        - ✅ Multi-modal classification
        - ✅ Real-time analytics dashboard
        - ✅ Light/Dark theme support
        - ✅ Export capabilities
        - ✅ RESTful API integration
        - ✅ Enterprise-grade security
        
        ### 📞 Support
        
        For enterprise support and customization:
        - 📧 Email: support@enterprise.com
        - 🌐 Web: www.enterprise-classifier.com
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Charger CSS
    load_enterprise_css()
    
    # Header
    render_header()
    
    # Status API
    render_api_status()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Catégories
    render_categories()
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Classification", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        render_classification_tab()
    
    with tab2:
        render_dashboard_tab()
    
    with tab3:
        render_about_tab()
    
    # Sidebar
    with st.sidebar:
        colors = get_theme_colors()
        
        st.markdown(f"<div style='color: {colors['text']};'>", unsafe_allow_html=True)
        st.markdown("## ⚙️ System Settings")
        
        st.markdown("### 🌐 API Connection")
        render_api_status()
        
        st.markdown("---")
        
        st.markdown("### 📊 Quick Stats")
        if st.session_state.history:
            st.metric("Total Classifications", len(st.session_state.history))
            avg_conf = np.mean([h['confidence'] for h in st.session_state.history])
            st.metric("Average Confidence", f"{avg_conf*100:.1f}%")
        else:
            st.info("No data yet")
        
        st.markdown("---")
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_prediction = None
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
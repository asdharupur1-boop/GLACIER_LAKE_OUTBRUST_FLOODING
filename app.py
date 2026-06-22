# ================================================================
# app.py - Complete Streamlit Dashboard (REQUIRES MODEL)
# IoT Water Level Monitoring & Prediction System
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import os
import requests
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="Water Level Monitoring Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# DEVELOPER DETAILS
# ================================================================

DEVELOPER_INFO = {
    "name": "Dr. [Your Name]",
    "role": "Lead AI/ML Engineer",
    "organization": "Water Resources Research Center",
    "email": "your.email@research.org",
    "phone": "+91-XXXXX-XXXXX",
    "github": "https://github.com/yourusername",
    "linkedin": "https://linkedin.com/in/yourprofile",
    "website": "https://yourwebsite.com",
    "research_area": "IoT & AI for Water Resource Management",
    "publications": [
        "CNN-LSTM for Water Level Prediction (2024)",
        "Reinforcement Learning for Flood Management (2024)",
        "IoT Sensor Networks for Remote Lake Monitoring (2023)"
    ]
}

# ================================================================
# SESSION STATE
# ================================================================

if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = "None"
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'model' not in st.session_state:
    st.session_state.model = None

# ================================================================
# CHECK MODEL FILES
# ================================================================

def check_model_files():
    """Check if all required model files exist."""
    required_files = [
        'models/cnn_lstm_model.h5',
        'models/scaler.pkl',
        'models/feature_columns.json',
        'models/metadata.json'
    ]
    
    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)
    
    return missing

def load_model_files():
    """Load all model files."""
    try:
        # Load scaler
        scaler = joblib.load('models/scaler.pkl')
        
        # Load feature columns
        with open('models/feature_columns.json', 'r') as f:
            feature_cols = json.load(f)
        
        # Load metadata
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load model
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        model = load_model('models/cnn_lstm_model.h5', compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model, scaler, feature_cols, metadata
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, None

# ================================================================
# TITLE
# ================================================================

st.title("🌊 IoT Water Level Monitoring & Prediction System")
st.markdown("---")

# ================================================================
# CHECK MODEL BEFORE ANYTHING ELSE
# ================================================================

missing_files = check_model_files()

if missing_files:
    st.error("🚨 **MODEL FILES NOT FOUND!**")
    st.markdown(f"""
    ### ❌ Missing Files:
    """)
    for f in missing_files:
        st.markdown(f"- `{f}`")
    
    st.markdown("""
    ---
    ### 🔧 How to Fix:
    
    **Option 1: Train the Model**
    ```bash
    # Run the training script
    python train_model.py
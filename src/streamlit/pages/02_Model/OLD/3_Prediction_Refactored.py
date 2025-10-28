import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Calcul correct du project_root
current_file = Path(__file__)  # pages/02_Model/3_Prediction.py
current_dir = current_file.parent  # pages/02_Model/
streamlit_dir = current_dir.parent.parent  # streamlit/
src_dir = streamlit_dir.parent  # src/
project_root = src_dir.parent  # DS_COVID/

sys.path.append(str(project_root))

# Imports des widgets modulaires
try:
    from src.features.Widget_Streamlit import (
        create_model_selection_widget,
        create_prediction_config_sidebar,
        create_simple_prediction_tab,
        create_batch_prediction_tab,
        create_prediction_analysis_tab
    )
    WIDGETS_OK = True
except ImportError as e:
    st.error(f"❌ Erreur d'import des widgets: {e}")
    WIDGETS_OK = False

# Imports des modules pour la prédiction
try:
    from src.features.Pipelines.Pipeline_Sklearn import PipelineManager
    from src.features.Pipelines.Pipeline_TensorFlow import TensorFlowPipelineManager
    from src.features.Pipelines.Pipeline_DataAugmentation import DataAugmentationPipeline
    from src.features.Data_Loaders.covid_data_loader import (
        load_covid_dataset, prepare_data_for_sklearn, prepare_data_for_tensorflow,
        get_data_paths, check_data_availability
    )
    IMPORTS_OK = True
except ImportError as e:
    st.error(f"❌ Erreur d'import des modules: {e}")
    IMPORTS_OK = False

# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

st.title("🔮 Prédiction COVID-19")

st.markdown("""
Cette page permet de faire des prédictions avec les modèles entraînés.
Vous pouvez utiliser différentes sources de données et modes de prédiction.
""")

# Vérification des imports
if not WIDGETS_OK or not IMPORTS_OK:
    st.error("❌ Impossible de charger tous les modules nécessaires")
    st.stop()

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

with st.sidebar:
    # Configuration de prédiction
    prediction_config = create_prediction_config_sidebar()
    
    # Sélection du modèle
    selected_model, selected_model_name, model_info = create_model_selection_widget()

# ============================================================================
# INTERFACE PRINCIPALE - ONGLETS
# ============================================================================

tab1, tab2, tab3 = st.tabs(["🎯 Prédiction Simple", "📊 Batch Prédiction", "📈 Analyse"])

# ============================================================================
# ONGLET 1 - PRÉDICTION SIMPLE
# ============================================================================

with tab1:
    prediction_results = create_simple_prediction_tab(
        selected_model=selected_model,
        selected_model_name=selected_model_name,
        prediction_mode=prediction_config["prediction_mode"]
    )

# ============================================================================
# ONGLET 2 - PRÉDICTION EN LOT
# ============================================================================

with tab2:
    batch_results = create_batch_prediction_tab(
        selected_model=selected_model,
        selected_model_name=selected_model_name
    )

# ============================================================================
# ONGLET 3 - ANALYSE DES PRÉDICTIONS
# ============================================================================

with tab3:
    create_prediction_analysis_tab()

# ============================================================================
# INFORMATIONS DE STATUT
# ============================================================================

# Afficher les informations du modèle sélectionné dans la sidebar
with st.sidebar:
    if selected_model_name:
        st.divider()
        st.subheader("🤖 Modèle Actuel")
        st.info(f"**{selected_model_name}**")
        
        if model_info:
            if 'test_accuracy' in model_info:
                st.metric("Précision", f"{model_info['test_accuracy']:.4f}")
            if 'timestamp' in model_info:
                st.write(f"📅 {model_info['timestamp'].strftime('%d/%m/%Y %H:%M')}")

# ============================================================================
# INFORMATIONS DE DEBUG (sidebar)
# ============================================================================

if st.sidebar.checkbox("🔧 Mode Debug", False):
    with st.sidebar.expander("Debug Info"):
        st.write("**Project Root:**", str(project_root))
        st.write("**Prediction Mode:**", prediction_config.get("prediction_mode", "Non défini"))
        st.write("**Selected Model:**", selected_model_name if selected_model_name else "Aucun")
        
        if selected_model:
            st.write("**Model Type:**", type(selected_model).__name__)
        
        # État des prédictions
        if 'last_predictions' in st.session_state:
            last_pred = st.session_state['last_predictions']
            st.write("**Last Prediction:**")
            st.write(f"- Model: {last_pred.get('model_name', 'N/A')}")
            st.write(f"- Samples: {last_pred.get('n_samples', 0)}")
            st.write(f"- Timestamp: {last_pred.get('timestamp', 'N/A')}")
        else:
            st.write("**Last Prediction:**", "Aucune")
        
        # Modèles disponibles
        if 'training_results' in st.session_state:
            st.write("**Available Models:**", list(st.session_state['training_results'].keys()))
        else:
            st.write("**Training Results:**", "Non disponibles")
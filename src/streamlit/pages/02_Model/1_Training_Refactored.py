import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Ajouter le répertoire racine au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))  # Ajouter le dossier streamlit

# Calcul correct du project_root
current_file = Path(__file__)  # pages/02_Model/1_Training.py
current_dir = current_file.parent  # pages/02_Model/
streamlit_dir = current_dir.parent.parent  # streamlit/
src_dir = streamlit_dir.parent  # src/
project_root = src_dir.parent  # DS_COVID/

sys.path.append(str(project_root))

# Imports des widgets modulaires
try:
    from src.features.Widget_Streamlit import (
        create_data_source_config,
        create_general_parameters,
        create_pipeline_type_selector,
        create_training_options,
        create_tensorflow_specific_params,
        show_configuration_summary,
        create_pipeline_configuration_tab,
        create_training_tab,
        create_results_comparison_tab,
        create_data_verification_widget,
        save_training_results_to_session
    )
    WIDGETS_OK = True
except ImportError as e:
    st.error(f"❌ Erreur d'import des widgets: {e}")
    WIDGETS_OK = False

# Imports des modules de données et pipelines
try:
    from src.features.Pipelines.Pipeline_Sklearn import PipelineManager
    from src.features.Pipelines.Pipeline_DataAugmentation import DataAugmentationPipeline
    from src.features.Pipelines.Pipeline_TensorFlow import TensorFlowPipelineManager
    from src.features.Data_Loaders.covid_data_loader import (
        load_covid_dataset, check_data_availability, prepare_data_for_sklearn,
        prepare_data_for_tensorflow, CLASS_NAMES, COVID_CLASSES, get_data_paths
    )
    IMPORTS_OK = True
except ImportError as e:
    st.error(f"❌ Erreur d'import des modules: {e}")
    IMPORTS_OK = False

# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

st.title("🏋️ Entraînement du Modèle")

st.markdown("""
Cette page permet d'entraîner différents modèles de classification pour la détection COVID-19 
en utilisant des pipelines configurables et modulaires.
""")

# Vérification des imports
if not WIDGETS_OK or not IMPORTS_OK:
    st.error("❌ Impossible de charger tous les modules nécessaires")
    st.stop()

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Configuration de la source des données
    data_config = create_data_source_config()
    
    # Type de pipeline
    pipeline_type = create_pipeline_type_selector()
    
    # Paramètres généraux
    general_params = create_general_parameters()
    
    # Paramètres spécifiques TensorFlow si nécessaire
    tf_params = None
    if pipeline_type == "TensorFlow (Deep Learning)":
        tf_params = create_tensorflow_specific_params()

# ============================================================================
# INTERFACE PRINCIPALE - ONGLETS
# ============================================================================

tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "🚀 Entraînement", "📊 Résultats"])

# ============================================================================
# ONGLET 1 - CONFIGURATION
# ============================================================================

with tab1:
    # Vérification des données si nécessaire
    data_available = create_data_verification_widget(project_root, data_config)
    
    if data_available:
        # Configuration des pipelines
        selected_configs, manager = create_pipeline_configuration_tab(
            project_root, 
            pipeline_type
        )
        
        # Afficher le résumé de configuration
        if selected_configs:
            complete_config = {
                **data_config,
                **general_params,
                "pipeline_type": pipeline_type,
                "selected_configs": selected_configs
            }
            if tf_params:
                complete_config.update(tf_params)
            
            show_configuration_summary(complete_config)

# ============================================================================
# ONGLET 2 - ENTRAÎNEMENT
# ============================================================================

with tab2:
    # Vérifier que les configurations sont disponibles
    if 'selected_configs' not in locals() or not selected_configs:
        st.warning("⚠️ Veuillez d'abord configurer les pipelines dans l'onglet Configuration")
    else:
        # Lancer l'interface d'entraînement
        training_results = create_training_tab(
            selected_configs=selected_configs,
            pipeline_type=pipeline_type,
            data_config=data_config,
            general_params=general_params,
            tf_params=tf_params
        )
        
        # Si l'entraînement a produit des résultats, les sauvegarder
        if training_results:
            comparison_data = {
                'configs': selected_configs,
                'pipeline_type': pipeline_type,
                'data_source': data_config["data_source"],
                'data_info': {
                    'n_samples': data_config.get('n_samples', 1000) if data_config["data_source"] == "Données simulées (test)" else 0,
                    'n_features': data_config.get('n_features', 16384) if data_config["data_source"] == "Données simulées (test)" else 0,
                    'n_classes': data_config.get('n_classes', 4) if data_config["data_source"] == "Données simulées (test)" else len(data_config.get('selected_classes', [])),
                    'test_size': general_params.get('test_size', 0.2)
                }
            }
            
            save_training_results_to_session(
                results=training_results,
                pipeline_type=pipeline_type,
                comparison_data=comparison_data
            )

# ============================================================================
# ONGLET 3 - RÉSULTATS
# ============================================================================

with tab3:
    # Afficher la comparaison des résultats
    create_results_comparison_tab()

# ============================================================================
# INFORMATIONS DE DEBUG (sidebar)
# ============================================================================

if st.sidebar.checkbox("🔧 Mode Debug", False):
    with st.sidebar.expander("Debug Info"):
        st.write("**Project Root:**", str(project_root))
        st.write("**Pipeline Type:**", pipeline_type if 'pipeline_type' in locals() else "Non défini")
        st.write("**Data Source:**", data_config.get("data_source", "Non défini") if 'data_config' in locals() else "Non défini")
        
        if 'selected_configs' in locals():
            st.write("**Selected Configs:**", selected_configs)
        
        if 'training_results' in st.session_state:
            st.write("**Training Results Available:**", len(st.session_state['training_results']))
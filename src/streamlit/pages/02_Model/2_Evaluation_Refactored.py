import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Calcul correct du project_root
current_file = Path(__file__)  # pages/02_Model/2_Evaluation.py
current_dir = current_file.parent  # pages/02_Model/
streamlit_dir = current_dir.parent.parent  # streamlit/
src_dir = streamlit_dir.parent  # src/
project_root = src_dir.parent  # DS_COVID/

sys.path.append(str(project_root))

# Imports des widgets modulaires
try:
    from src.features.Widget_Streamlit import (
        create_model_selection_tab,
        create_detailed_metrics_tab,
        create_visualizations_tab,
        create_evaluation_report_tab
    )
    WIDGETS_OK = True
except ImportError as e:
    st.error(f"❌ Erreur d'import des widgets: {e}")
    WIDGETS_OK = False

# Imports des modules pour l'évaluation
try:
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_curve, auc,
        precision_recall_curve, average_precision_score
    )
    from sklearn.preprocessing import label_binarize
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    IMPORTS_OK = True
except ImportError as e:
    st.error(f"❌ Erreur d'import des modules d'évaluation: {e}")
    IMPORTS_OK = False

# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

st.title("📊 Évaluation des Modèles")

st.markdown("""
Cette page permet d'évaluer en détail les modèles entraînés et de comparer leurs performances
avec différentes métriques et visualisations avancées.
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
    
    # Mode d'évaluation
    eval_mode = st.selectbox(
        "Mode d'évaluation",
        ["Modèles en session", "Modèles sauvegardés", "Upload modèle"]
    )
    
    # Options d'affichage
    st.subheader("📈 Visualisations")
    show_confusion_matrix = st.checkbox("Matrice de confusion", True)
    show_roc_curves = st.checkbox("Courbes ROC", True)
    show_precision_recall = st.checkbox("Précision-Rappel", True)
    show_feature_importance = st.checkbox("Importance des features", False)

# ============================================================================
# INTERFACE PRINCIPALE - ONGLETS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Sélection Modèles", "📊 Métriques", "📈 Visualisations", "📋 Rapport"])

# ============================================================================
# ONGLET 1 - SÉLECTION DES MODÈLES
# ============================================================================

with tab1:
    selected_models = create_model_selection_tab(eval_mode)

# ============================================================================
# ONGLET 2 - MÉTRIQUES DÉTAILLÉES
# ============================================================================

with tab2:
    if selected_models:
        create_detailed_metrics_tab(selected_models)
    else:
        st.info("ℹ️ Sélectionnez des modèles dans l'onglet 'Sélection Modèles' pour voir les métriques détaillées.")

# ============================================================================
# ONGLET 3 - VISUALISATIONS
# ============================================================================

with tab3:
    if selected_models:
        create_visualizations_tab(selected_models)
    else:
        st.info("ℹ️ Sélectionnez des modèles dans l'onglet 'Sélection Modèles' pour voir les visualisations.")

# ============================================================================
# ONGLET 4 - RAPPORT
# ============================================================================

with tab4:
    if selected_models:
        create_evaluation_report_tab(selected_models)
    else:
        st.info("ℹ️ Sélectionnez des modèles dans l'onglet 'Sélection Modèles' pour générer un rapport.")

# ============================================================================
# INFORMATIONS DE DEBUG (sidebar)
# ============================================================================

if st.sidebar.checkbox("🔧 Mode Debug", False):
    with st.sidebar.expander("Debug Info"):
        st.write("**Project Root:**", str(project_root))
        st.write("**Eval Mode:**", eval_mode)
        
        if 'selected_models' in locals() and selected_models:
            st.write("**Selected Models:**", selected_models)
        
        if 'training_results' in st.session_state:
            st.write("**Available Models:**", list(st.session_state['training_results'].keys()))
        else:
            st.write("**Training Results:**", "Non disponibles")
        
        # Options d'affichage sélectionnées
        st.write("**Display Options:**")
        st.write(f"- Confusion Matrix: {show_confusion_matrix}")
        st.write(f"- ROC Curves: {show_roc_curves}")
        st.write(f"- Precision-Recall: {show_precision_recall}")
        st.write(f"- Feature Importance: {show_feature_importance}")
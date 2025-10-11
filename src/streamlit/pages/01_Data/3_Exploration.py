import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import random
import cv2
from PIL import Image
import sys
import os

# Ajouter le chemin du répertoire courant pour les imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import des onglets avec gestion d'erreur
try:
    from tabs.overview_tab import show_overview_tab
    from tabs.metadata_tab import show_metadata_tab
    from tabs.images_tab import show_images_tab
    #from tabs.comparison_tab import show_comparison_tab
    from tabs.filtering_tab import show_filtering_tab
    st.success("✅ Tous les modules d'onglets importés avec succès")
except ImportError as e:
    st.error(f"❌ Erreur d'import des modules d'onglets: {e}")
    st.info("💡 Vérifiez que tous les fichiers d'onglets sont présents dans le dossier 'tabs/'")
    st.stop()
except Exception as e:
    st.error(f"❌ Erreur inattendue lors de l'import: {e}")
    st.exception(e)
    st.stop()

# ====================================================================
# 🔧 BACKEND - LOGIQUE D'EXPLORATION
# ====================================================================

def check_data_loaded():
    """Vérifie si les données ont été chargées."""
    if "loaded_data" not in st.session_state:
        return False, "Aucune donnée chargée"
    
    data = st.session_state["loaded_data"]
    required_keys = ["metadata_dfs", "stats", "data_dir"]
    
    for key in required_keys:
        if key not in data:
            return False, f"Données incomplètes: {key} manquant"
    
    return True, "Données disponibles"

# ====================================================================
# 🎨 FRONTEND - INTERFACE D'EXPLORATION
# ====================================================================

def show_data_status():
    """Vérifie si les données ont été chargées."""
    status_ok, message = check_data_loaded()
    
    if status_ok:
        st.success(f"✅ {message} - Exploration disponible", icon="✅")
        return True
    else:
        st.error(f"❌ {message}", icon="❌")
        st.info("💡 Veuillez d'abord charger les données dans la page précédente.")
        return False


def main():
    """Fonction principale."""
    try:
        st.markdown("# 🔍 Exploration des Données COVID-19")
        st.markdown("---")
        
        # Vérification des prérequis
        if not show_data_status():
            return
        
        # Récupération des données
        loaded_data = st.session_state["loaded_data"]
        metadata_dfs = loaded_data["metadata_dfs"]
        stats = loaded_data["stats"]
        data_dir = loaded_data["data_dir"]
        categories = list(metadata_dfs.keys())
        
        st.success(f"✅ Données chargées: {len(categories)} catégories disponibles")
        
        # Navigation par onglets
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Vue d'ensemble",
            "🔍 Métadonnées", 
            "🖼️ Images",
            "🔎 Filtrage"
        ])
        
        with tab1:
            try:
                show_overview_tab(stats, metadata_dfs)
            except Exception as e:
                st.error(f"❌ Erreur dans l'onglet Vue d'ensemble: {e}")
                st.exception(e)
        
        with tab2:
            try:
                show_metadata_tab(metadata_dfs)
            except Exception as e:
                st.error(f"❌ Erreur dans l'onglet Métadonnées: {e}")
                st.exception(e)
        
        with tab3:
            try:
                show_images_tab(data_dir, categories)
            except Exception as e:
                st.error(f"❌ Erreur dans l'onglet Images: {e}")
                st.exception(e)
        
        with tab4:
            try:
                show_filtering_tab(metadata_dfs)
            except Exception as e:
                st.error(f"❌ Erreur dans l'onglet Filtrage: {e}")
                st.exception(e)
        
        # Message de fin
        st.markdown("---")
        #st.info("💡 **Prochaine étape:** Vous pouvez maintenant procéder à l'entraînement du modèle.")
        
    except Exception as e:
        st.error(f"❌ Erreur critique dans l'exploration: {e}")
        st.exception(e)
        st.info("🔧 Détails techniques pour le débogage affichés ci-dessus")

if __name__ == "__main__":
    main()
else:
    # Exécution automatique quand importé
    main()

import streamlit as st
from pathlib import Path
import sys

# ====================================================================
# 🎨 FRONTEND - INTERFACE UTILISATEUR
# ====================================================================

def show_global_status(results):
    """Affiche l'indicateur de statut global en haut de page."""
    if results["all_checks_passed"]:
        st.sidebar.success("🎉 **Vérifications OK !**", icon="✅")
    else:
        st.sidebar.error("⚠️ **Vérifications échouées.**", icon="❌")


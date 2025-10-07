import streamlit as st

st.set_page_config(page_title="Menu Personnalisé", page_icon="📚")

st.title("📚 Menu Personnalisé avec Sidebar")
st.sidebar.success("Navigation latérale activée ✔️")

st.markdown(
    """
La `sidebar` permet de créer une **navigation structurée** et d'y placer des filtres, des contrôles ou du contenu contextuel.
"""
)

# Exemple de contrôle dans la sidebar
choix = st.sidebar.radio(
    "Choisissez une catégorie",
    ["Exploration", "Visualisation", "Modélisation"]
)

st.write(f"Vous avez choisi : **{choix}**")

import streamlit as st

st.set_page_config(
    page_title="Démo Streamlit",
    page_icon="🎯",
    layout="centered",
)

st.title("🎯 Démonstration Streamlit")

st.markdown(
    """
Bienvenue sur cette application **interne** pour explorer ce que Streamlit permet !

Ce prototype démontre de manière ludique et interactive :
- 📁 L’upload de fichiers
- 📊 L’analyse de données
- 🧮 Une mini calculatrice
- 🔮 Une simulation de modèle

---

👉 Utilisez le **menu de gauche** pour naviguer entre les pages.

"""
)

st.balloons()

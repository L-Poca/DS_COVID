import streamlit as st
import json

st.set_page_config(page_title="Affichage JSON", page_icon="🧾")

st.title("🧾 JSON Viewer")
st.markdown("Affichage de données structurées avec `st.json()`")

# Exemple fixe
example_json = {
    "utilisateur": "steven.moire",
    "projet": "Démo Streamlit",
    "features": ["upload", "visualisation", "modèle", "audio", "vidéo"],
    "avancement": {"frontend": "OK", "modèle": "à venir", "CI/CD": "en place"}
}

st.subheader("🔍 JSON statique")
st.json(example_json)

# Exemple dynamique
st.subheader("📂 Uploader un fichier JSON")
uploaded_json = st.file_uploader("Choisissez un fichier JSON", type="json")

if uploaded_json:
    try:
        content = json.load(uploaded_json)
        st.success("JSON chargé avec succès 🎉")
        st.json(content)
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")

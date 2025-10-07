import streamlit as st

st.title("🏋️ Entraînement du Modèle")

st.write("Cette page permettra d'entraîner le modèle de détection COVID-19.")
st.info("🚧 Page en cours de développement")

# Exemple de structure pour l'entraînement
col1, col2 = st.columns(2)

with col1:
    st.subheader("Paramètres du modèle")
    epochs = st.slider("Nombre d'époques", 1, 100, 10)
    batch_size = st.selectbox("Taille du batch", [16, 32, 64, 128])

with col2:
    st.subheader("Architecture")
    model_type = st.selectbox("Type de modèle", ["CNN", "ResNet", "VGG"])

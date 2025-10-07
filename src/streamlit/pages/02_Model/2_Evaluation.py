import streamlit as st

st.title("📈 Évaluation du Modèle")

st.write("Cette page permettra d'évaluer les performances du modèle.")
st.info("🚧 Page en cours de développement")

# Exemple de métriques
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Précision", "85%", "2%")

with col2:
    st.metric("Rappel", "82%", "1%")

with col3:
    st.metric("F1-Score", "83%", "1.5%")

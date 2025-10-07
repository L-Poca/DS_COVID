import streamlit as st

st.title("🎯 Prédictions")

st.write("Cette page permettra de faire des prédictions sur de nouvelles images.")
st.info("🚧 Page en cours de développement")

# Interface de prédiction
uploaded_file = st.file_uploader("Choisir une image radiographique", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Image téléchargée", use_container_width=True)
    
    if st.button("Analyser l'image"):
        st.success("Résultat : COVID-19 détecté avec 85% de confiance")
        
        # Graphique de confiance
        import numpy as np
        confidence_data = {
            'COVID-19': 0.85,
            'Normal': 0.10,
            'Pneumonie': 0.05
        }
        st.bar_chart(confidence_data)

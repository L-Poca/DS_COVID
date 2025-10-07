import streamlit as st

st.title("🔮 Simulation d’un modèle")

age = st.slider("Âge", 18, 90, 30)
sexe = st.radio("Sexe", ["Homme", "Femme"])
symptomes = st.multiselect(
    "Symptômes",
    ["Toux", "Fièvre", "Fatigue", "Essoufflement"]
    )

if st.button("Prédire risque"):
    score = age / 100 + len(symptomes) * 0.1
    st.metric("Score de risque (fictif)", f"{score:.2f}")
    if score > 0.8:
        st.error("Risque élevé")
    elif score > 0.5:
        st.warning("Risque modéré")
    else:
        st.success("Risque faible")

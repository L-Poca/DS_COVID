import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Upload & Analyse", page_icon="📁")

st.title("📁 Upload & 📊 Analyse de données")

# Étape 1 – Upload du CSV
st.header("Étape 1 : Upload de données")

uploaded_file = st.file_uploader("📂 Choisissez un fichier CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Fichier chargé avec succès !")

    st.subheader("🔍 Aperçu des données")
    st.dataframe(df.head())

    # Étape 2 – Analyse
    st.header("Étape 2 : Analyse simple")

    numeric_cols = df.select_dtypes(include="number").columns

    if numeric_cols.empty:
        st.warning("Aucune colonne numérique trouvée pour l’analyse.")
    else:
        col = st.selectbox(
            "Choisissez une colonne numérique à analyser",
            numeric_cols
            )
        st.write("📈 Statistiques de base :")
        st.write(df[col].describe())

        # Histogramme
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=30, color="skyblue", edgecolor="black")
        ax.set_title(f"Histogramme de la colonne : {col}")
        st.pyplot(fig)

else:
    st.info("⏳ En attente d’un fichier à analyser...")

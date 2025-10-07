import streamlit as st
import pandas as pd
import plotly.express as px

# Config de la page
st.set_page_config(page_title="Analyse Graphique", page_icon="📊")

st.title("📊 Analyse graphique automatisée")

# Chargement du fichier CSV en dur
csv_path = "reports/explorationdata/Template_rapport_exploration_donnees_19_06_2025_17_37.csv" # noqa

try:
    df = pd.read_csv(csv_path)
    st.success(f"✅ Fichier chargé depuis : `{csv_path}`")
except FileNotFoundError:
    st.error(f"❌ Fichier introuvable à l’emplacement : `{csv_path}`")
    st.stop()

# Création d'une étiquette combinée
df["Etiquette"] = df[
    "Nom du dossier"
    ].astype(str) + " - " + df["Nom du sous dossier"].astype(str)

# Colonnes numériques disponibles (hors colonnes à exclure)
colonnes_exclues = ["Nom du dossier", "Nom du sous dossier"]
colonnes_numeriques = df.select_dtypes(
    include="number"
    ).drop(
        columns=[col for col in colonnes_exclues if col in df.columns],
        errors="ignore"
        ).columns

if colonnes_numeriques.empty:
    st.warning("Aucune colonne numérique disponible pour l’analyse graphique.")
else:
    # Sélection d’une colonne à analyser
    col = st.selectbox(
        "Choisissez une colonne numérique à visualiser :",
        colonnes_numeriques
        )

    # Affichage de statistiques
    st.write("📌 Statistiques descriptives :")
    st.write(df[col].describe())

    # Création d’un graphique interactif Plotly
    fig = px.bar(
        df,
        x="Etiquette",
        y=col,
        title=f"📊 Graphique de la colonne : {col}",
        labels={"Etiquette": "Dossier / Sous-dossier", col: col},
        hover_data=["Etiquette"]
    )
    fig.update_layout(xaxis_tickangle=-45)

    st.plotly_chart(fig, use_container_width=True)

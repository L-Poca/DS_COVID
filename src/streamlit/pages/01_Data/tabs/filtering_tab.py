import streamlit as st
import pandas as pd


def show_data_filtering(metadata_dfs=None):
    """Affiche les options de filtrage et de recherche."""
    st.markdown("## 🔎 **Filtrage et Recherche**")
    
    # Utiliser les données passées en paramètre ou celles du session_state
    if metadata_dfs is None and "loaded_data" in st.session_state:
        metadata_dfs = st.session_state["loaded_data"]["metadata_dfs"]
    
    if metadata_dfs:
        combined_df = pd.concat(metadata_dfs.values(), ignore_index=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Filtre par catégorie
            selected_categories = st.multiselect(
                "Filtrer par catégorie:",
                options=combined_df["category"].unique(),
                default=combined_df["category"].unique(),
                key="filtering_categories"
            )
        
        with col2:
            # Recherche textuelle (si colonnes texte disponibles)
            text_columns = combined_df.select_dtypes(include=['object']).columns
            if len(text_columns) > 1:  # Exclure 'category'
                search_column = st.selectbox(
                    "Rechercher dans la colonne:",
                    options=[col for col in text_columns if col != 'category'],
                    key="filtering_search_column"
                )
                search_term = st.text_input("Terme de recherche:", key="filtering_search_term")
        
        # Appliquer les filtres
        filtered_df = combined_df[combined_df["category"].isin(selected_categories)]
        
        if 'search_column' in locals() and 'search_term' in locals() and search_term:
            filtered_df = filtered_df[
                filtered_df[search_column].str.contains(search_term, case=False, na=False)
            ]
        
        st.markdown(f"### 📊 Résultats ({len(filtered_df)} échantillons)")
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.warning("❌ Aucune donnée de métadonnées disponible pour le filtrage.")


def show_filtering_tab(metadata_dfs=None):
    """Affiche l'onglet filtrage."""
    show_data_filtering(metadata_dfs)

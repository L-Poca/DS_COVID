import streamlit as st
import pandas as pd


def show_metadata_exploration(metadata_dfs):
    """Affiche l'exploration des métadonnées."""
    st.markdown("## 🔍 **Exploration des Métadonnées**")
    
    # Sélecteur de catégorie
    selected_category = st.selectbox(
        "Choisissez une catégorie à explorer:",
        options=list(metadata_dfs.keys()),
        key="metadata_category_selector"
    )
    
    if selected_category:
        df = metadata_dfs[selected_category]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### 📋 Métadonnées - {selected_category}")
            st.dataframe(df, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Informations")
            st.write(f"**Nombre de lignes:** {len(df)}")
            st.write(f"**Nombre de colonnes:** {len(df.columns)}")
            
            st.markdown("**Colonnes disponibles:**")
            for col in df.columns:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                st.write(f"• `{col}` ({missing_pct:.1f}% manquant)")
            
            # Analyse des valeurs uniques pour les colonnes catégorielles
            st.markdown("**Valeurs uniques (colonnes texte):**")
            for col in df.select_dtypes(include=['object']).columns:
                unique_count = df[col].nunique()
                if unique_count < 20:  # Afficher seulement si peu de valeurs uniques
                    st.write(f"• `{col}`: {unique_count} valeurs")
                    unique_vals = df[col].dropna().unique()
                    st.write(f"  → {', '.join(str(v) for v in unique_vals[:5])}{'...' if len(unique_vals) > 5 else ''}")


def show_metadata_tab(metadata_dfs):
    """Affiche l'onglet métadonnées."""
    show_metadata_exploration(metadata_dfs)

import streamlit as st
import pandas as pd
import plotly.express as px


def create_comparative_analysis(metadata_dfs):
    """Crée une analyse comparative entre catégories."""
    combined_df = pd.concat(metadata_dfs.values(), ignore_index=True)
    
    analysis = {
        "category_distribution": combined_df['category'].value_counts(),
        "common_columns": set.intersection(*[set(df.columns) for df in metadata_dfs.values()]),
        "total_samples": len(combined_df),
        "categories_count": combined_df['category'].nunique()
    }
    
    return analysis, combined_df


def show_comparative_analysis(metadata_dfs):
    """Affiche une analyse comparative."""
    st.markdown("## ⚖️ **Analyse Comparative**")
    
    analysis, combined_df = create_comparative_analysis(metadata_dfs)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Distribution par catégorie")
        dist_fig = px.bar(
            x=analysis["category_distribution"].index,
            y=analysis["category_distribution"].values,
            title="Nombre d'échantillons par catégorie"
        )
        st.plotly_chart(dist_fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Informations générales")
        st.metric("🔢 Total échantillons", analysis["total_samples"])
        st.metric("📁 Catégories", analysis["categories_count"])
        st.metric("🔗 Colonnes communes", len(analysis["common_columns"]))
        
        if analysis["common_columns"]:
            st.markdown("**Colonnes communes:**")
            for col in sorted(analysis["common_columns"]):
                st.write(f"• `{col}`")


def show_comparison_tab(metadata_dfs):
    """Affiche l'onglet comparaison."""
    show_comparative_analysis(metadata_dfs)

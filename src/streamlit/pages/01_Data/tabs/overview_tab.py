import streamlit as st
import pandas as pd
import plotly.express as px


def show_dataset_summary(stats, metadata_dfs):
    """Affiche un résumé du dataset."""
    st.markdown("## 📊 **Résumé du Dataset**")
    
    # Métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📁 Catégories", len(stats["categories"]))
    with col2:
        st.metric("🖼️ Images", stats["total_images"])
    with col3:
        st.metric("🎭 Masques", stats["total_masks"])
    with col4:
        total_metadata = sum(len(df) for df in metadata_dfs.values())
        st.metric("📋 Métadonnées", total_metadata)
    with col5:
        avg_per_cat = stats["total_images"] // len(stats["categories"])
        st.metric("📈 Moy/Catégorie", avg_per_cat)


def show_interactive_charts(stats, metadata_dfs):
    """Affiche des graphiques interactifs."""
    st.markdown("## 📈 **Analyses Visuelles**")
    
    # Répartition par catégorie
    col1, col2 = st.columns(2)
    
    with col1:
        categories = list(stats["categories"].keys())
        values = [stats["categories"][cat]["images"] for cat in categories]
        
        fig_pie = px.pie(
            values=values,
            names=categories,
            title="Répartition des Images par Catégorie",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Comparaison Images vs Masques
        comparison_data = []
        for cat, data in stats["categories"].items():
            comparison_data.extend([
                {"Catégorie": cat, "Type": "Images", "Nombre": data["images"]},
                {"Catégorie": cat, "Type": "Masques", "Nombre": data["masks"]}
            ])
        
        comp_df = pd.DataFrame(comparison_data)
        fig_bar = px.bar(
            comp_df,
            x="Catégorie",
            y="Nombre",
            color="Type",
            title="Images vs Masques par Catégorie",
            barmode="group"
        )
        st.plotly_chart(fig_bar, use_container_width=True)


def show_overview_tab(stats, metadata_dfs):
    """Affiche l'onglet vue d'ensemble."""
    show_dataset_summary(stats, metadata_dfs)
    show_interactive_charts(stats, metadata_dfs)

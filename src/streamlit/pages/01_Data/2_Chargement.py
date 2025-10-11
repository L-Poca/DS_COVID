import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from src.features.Verifs_Env.Vérifications_Back import *
from src.features.Widget_Streamlit.W_Vérifications_Front import *

show_global_status(run_all_checks())

# ====================================================================
# 🔧 BACKEND - LOGIQUE DE CHARGEMENT
# ====================================================================

def check_verification_status():
    """Vérifie si les vérifications préalables ont été effectuées."""
    if "verification_results" not in st.session_state:
        return False, "Aucune vérification trouvée"
    
    results = st.session_state["verification_results"]
    if not results.get("all_checks_passed", False):
        return False, "Les vérifications préalables ont échoué"
    
    return True, "Vérifications OK"

def load_metadata_files(data_dir):
    """Charge tous les fichiers de métadonnées."""
    metadata_files = {
        "COVID": "COVID.metadata.xlsx",
        "Normal": "Normal.metadata.xlsx",
        "Lung_Opacity": "Lung_Opacity.metadata.xlsx",
        "Viral Pneumonia": "Viral Pneumonia.metadata.xlsx"
    }
    
    metadata_dfs = {}
    for category, filename in metadata_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            try:
                df = pd.read_excel(filepath)
                df['category'] = category
                metadata_dfs[category] = df
            except Exception as e:
                st.error(f"Erreur lors du chargement de {filename}: {e}")
                return None
        else:
            st.error(f"Fichier {filename} non trouvé")
            return None
    
    return metadata_dfs

def get_dataset_statistics(data_dir, metadata_dfs):
    """Calcule les statistiques du dataset."""
    stats = {
        "categories": {},
        "total_images": 0,
        "total_masks": 0,
        "file_sizes": {}
    }
    
    for category in metadata_dfs.keys():
        category_path = data_dir / category
        images_path = category_path / "images"
        masks_path = category_path / "masks"
        
        # Compter les fichiers
        images_count = len(list(images_path.glob("*"))) if images_path.exists() else 0
        masks_count = len(list(masks_path.glob("*"))) if masks_path.exists() else 0
        
        stats["categories"][category] = {
            "images": images_count,
            "masks": masks_count,
            "metadata_rows": len(metadata_dfs[category])
        }
        
        stats["total_images"] += images_count
        stats["total_masks"] += masks_count
    
    return stats

def load_sample_images(data_dir, categories, n_samples=3):
    """Charge des échantillons d'images pour chaque catégorie."""
    samples = {}
    
    for category in categories:
        images_path = data_dir / category / "images"
        if images_path.exists():
            image_files = list(images_path.glob("*.png"))[:n_samples]
            samples[category] = []
            
            for img_file in image_files:
                try:
                    img = Image.open(img_file)
                    samples[category].append({
                        "name": img_file.name,
                        "image": img,
                        "size": img.size,
                        "path": str(img_file)
                    })
                except Exception as e:
                    st.warning(f"Impossible de charger {img_file.name}: {e}")
    
    return samples

# ====================================================================
# 🎨 FRONTEND - INTERFACE UTILISATEUR
# ====================================================================

def show_verification_status():
    """Affiche le statut des vérifications."""
    status_ok, message = check_verification_status()
    
    if status_ok:
        st.success(f"✅ {message} - Chargement autorisé", icon="✅")
        return True
    else:
        st.error(f"❌ {message}", icon="❌")
        st.info("💡 Veuillez d'abord effectuer les vérifications dans la page précédente.")
        return False

def show_loading_progress(data_dir):
    """Affiche la progression du chargement."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Étape 1: Chargement des métadonnées
    status_text.text("📊 Chargement des métadonnées...")
    progress_bar.progress(0.2)
    metadata_dfs = load_metadata_files(data_dir)
    
    if metadata_dfs is None:
        st.error("❌ Échec du chargement des métadonnées")
        return None, None, None
    
    # Étape 2: Calcul des statistiques
    status_text.text("📈 Calcul des statistiques...")
    progress_bar.progress(0.5)
    stats = get_dataset_statistics(data_dir, metadata_dfs)
    
    # Étape 3: Chargement des échantillons
    status_text.text("🖼️ Chargement des échantillons d'images...")
    progress_bar.progress(0.8)
    samples = load_sample_images(data_dir, metadata_dfs.keys())
    
    # Finalisation
    status_text.text("✅ Chargement terminé !")
    progress_bar.progress(1.0)
    
    return metadata_dfs, stats, samples

def show_dataset_overview(stats):
    """Affiche un aperçu du dataset."""
    st.markdown("## 📊 **Aperçu du Dataset**")
    
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📁 Catégories", len(stats["categories"]))
    with col2:
        st.metric("🖼️ Images totales", stats["total_images"])
    with col3:
        st.metric("🎭 Masques totaux", stats["total_masks"])
    with col4:
        st.metric("⚖️ Ratio", f"{stats['total_masks']/stats['total_images']:.1%}" if stats['total_images'] > 0 else "N/A")
    
    # Graphique de répartition
    categories = list(stats["categories"].keys())
    images_counts = [stats["categories"][cat]["images"] for cat in categories]
    
    fig = px.bar(
        x=categories,
        y=images_counts,
        title="Répartition des images par catégorie",
        labels={"x": "Catégories", "y": "Nombre d'images"},
        color=images_counts,
        color_continuous_scale="viridis"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def show_category_details(stats, metadata_dfs):
    """Affiche les détails par catégorie."""
    st.markdown("## 🏷️ **Détails par Catégorie**")
    
    for category, data in stats["categories"].items():
        with st.expander(f"📁 {category}", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🖼️ Images", data["images"])
            with col2:
                st.metric("🎭 Masques", data["masks"])
            with col3:
                st.metric("📋 Métadonnées", data["metadata_rows"])
            
            # Aperçu des métadonnées
            if category in metadata_dfs:
                st.markdown("**Aperçu des métadonnées:**")
                st.dataframe(metadata_dfs[category].head(), use_container_width=True)

def show_sample_images(samples):
    """Affiche des échantillons d'images."""
    st.markdown("## 🖼️ **Échantillons d'Images**")
    
    for category, images in samples.items():
        if images:
            st.markdown(f"### 📁 {category}")
            cols = st.columns(min(len(images), 3))
            
            for i, img_data in enumerate(images[:3]):
                with cols[i]:
                    st.image(
                        img_data["image"],
                        caption=f"{img_data['name']}\n{img_data['size'][0]}x{img_data['size'][1]}",
                        use_container_width=True
                    )

def show_data_export_options(metadata_dfs, stats):
    """Affiche les options d'export des données."""
    st.markdown("## 💾 **Options d'Export**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Exporter les métadonnées (CSV)", use_container_width=True):
            # Combine toutes les métadonnées
            combined_df = pd.concat(metadata_dfs.values(), ignore_index=True)
            csv = combined_df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger le CSV",
                data=csv,
                file_name="covid_dataset_metadata.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.success("✅ CSV prêt pour téléchargement")
    
    with col2:
        if st.button("📈 Exporter les statistiques (JSON)", use_container_width=True):
            import json
            stats_json = json.dumps(stats, indent=2, default=str)
            st.download_button(
                label="📥 Télécharger le JSON",
                data=stats_json,
                file_name="covid_dataset_stats.json",
                mime="application/json",
                use_container_width=True
            )
            st.success("✅ JSON prêt pour téléchargement")

def show_data_quality_check(metadata_dfs, stats):
    """Affiche un contrôle qualité des données."""
    st.markdown("## 🔍 **Contrôle Qualité**")
    
    with st.expander("📊 Analyse de cohérence", expanded=True):
        issues = []
        
        for category, data in stats["categories"].items():
            # Vérifier la cohérence images/métadonnées
            if data["images"] != data["metadata_rows"]:
                issues.append(f"⚠️ {category}: {data['images']} images vs {data['metadata_rows']} métadonnées")
            
            # Vérifier la présence de masques
            if data["masks"] == 0:
                issues.append(f"❌ {category}: Aucun masque trouvé")
        
        if issues:
            st.warning("Problèmes détectés:")
            for issue in issues:
                st.write(issue)
        else:
            st.success("✅ Toutes les vérifications de qualité sont passées")
    
    # Statistiques par colonne des métadonnées
    with st.expander("📋 Analyse des métadonnées", expanded=True):
        for category, df in metadata_dfs.items():
            st.markdown(f"**{category}:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Colonnes disponibles:")
                for col in df.columns:
                    st.write(f"• `{col}`")
            
            with col2:
                st.write("Valeurs manquantes:")
                missing = df.isnull().sum()
                for col, count in missing.items():
                    if count > 0:
                        st.write(f"• `{col}`: {count}")
                if missing.sum() == 0:
                    st.success("Aucune valeur manquante")

def main():
    """Fonction principale."""
    st.markdown("# 📂 Chargement des Données COVID-19")
    st.markdown("---")
    
    # Vérification des prérequis
    if not show_verification_status():
        return
    
    # Récupération des informations de vérification
    verification_results = st.session_state["verification_results"]
    data_dir = verification_results["data_dir"]
    
    # Chargement des données
    st.markdown("## 🔄 **Chargement en cours...**")
    metadata_dfs, stats, samples = show_loading_progress(data_dir)
    
    if metadata_dfs is None:
        st.error("❌ Échec du chargement. Veuillez vérifier vos données.")
        return
    
    # Stockage des données chargées pour les autres pages
    st.session_state["loaded_data"] = {
        "metadata_dfs": metadata_dfs,
        "stats": stats,
        "samples": samples,
        "data_dir": data_dir
    }
    
    st.success("✅ **Données chargées avec succès !**")
    st.markdown("---")
    
    # Affichage des sections
    show_dataset_overview(stats)
    show_category_details(stats, metadata_dfs)
    show_sample_images(samples)
    show_data_quality_check(metadata_dfs, stats)
    show_data_export_options(metadata_dfs, stats)
    
    # Message de fin
    st.markdown("---")
    st.info("💡 **Prochaine étape:** Vous pouvez maintenant explorer les données en détail.")

if __name__ == "__main__":
    main()
else:
    # Exécution automatique quand importé
    main()

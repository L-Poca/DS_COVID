import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from PIL import Image
import random
from scipy import ndimage, stats
from scipy.ndimage import median_filter
import time
import json
import zlib
from typing import List, Tuple, Dict
import multiprocessing as mp
from datetime import datetime


# ====================================================================
# 🚀 SYSTÈME DE MODES D'ANALYSE
# ====================================================================

def get_analysis_mode_config(context="default"):
    """Configure le mode d'analyse à utiliser."""
    st.markdown("### ⚙️ **Configuration de l'Analyse**")
    
    # Sélection du mode
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analysis_mode = st.selectbox(
            "Mode d'analyse:",
            [
                "🚀 Rapide (échantillon)",
                "📊 Complet (toutes les images)", 
                "⚖️ Équilibré (pourcentage)",
                "🎯 Personnalisé"
            ],
            key=f"analysis_mode_{context}"
        )
    
    with col2:
        # Option pour l'analyse avancée
        enhanced_analysis = st.checkbox("🚀 Analyse améliorée", value=False, key=f"enhanced_{context}")
        if enhanced_analysis:
            st.info("🔬 Métriques avancées activées")
        
    # Configuration selon le mode
    config = {
        "mode": analysis_mode,
        "name": analysis_mode,  # Ajout du nom pour l'affichage
        "enhanced": enhanced_analysis  # Ajout du flag enhanced
    }
    
    if analysis_mode == "🚀 Rapide (échantillon)":
        config["sample_size"] = st.slider("Nombre d'échantillons:", 5, 100, 20, key=f"rapid_sample_{context}")
        st.info("⚡ Analyse rapide sur un échantillon réduit")
        
    elif analysis_mode == "📊 Complet (toutes les images)":
        st.warning("⚠️ Cette analyse peut prendre du temps selon le nombre d'images")
        config["confirm"] = st.checkbox("Je confirme vouloir analyser toutes les images", key=f"full_confirm_{context}")
        if not config["confirm"]:
            st.stop()
        config["sample_size"] = None  # Toutes les images
        
    elif analysis_mode == "⚖️ Équilibré (pourcentage)":
        percentage = st.slider("Pourcentage des données:", 10, 100, 50, step=10, key=f"balanced_percent_{context}")
        config["percentage"] = percentage
        config["sample_size"] = None  # Sera calculé dynamiquement
        st.info(f"📊 Analyse sur {percentage}% des données disponibles")
        
    elif analysis_mode == "🎯 Personnalisé":
        config["sample_size"] = st.number_input(
            "Nombre d'images par catégorie:", 
            min_value=1, 
            max_value=10000, 
            value=100,
            key=f"custom_sample_{context}"
        )
        
    return config


def get_image_files_for_analysis(data_dir: Path, category: str, config: Dict) -> List[Path]:
    """Récupère les fichiers d'images selon la configuration d'analyse."""
    images_dir = data_dir / category / "images"
    
    if not images_dir.exists():
        return []
    
    all_files = list(images_dir.glob("*.png"))
    
    # Selon le mode, retourner le bon nombre de fichiers
    mode = config["mode"]
    
    if mode == "🚀 Rapide (échantillon)" or mode == "🎯 Personnalisé":
        sample_size = config.get("sample_size", 20)
        return all_files[:sample_size]
        
    elif mode == "📊 Complet (toutes les images)":
        return all_files
        
    elif mode == "⚖️ Équilibré (pourcentage)":
        percentage = config.get("percentage", 50)
        sample_size = int(len(all_files) * percentage / 100)
        return all_files[:sample_size]
    
    return all_files[:20]  # Fallback


def analyze_images_batch(image_paths: List[Path], batch_size: int = 10) -> Dict:
    """Analyse un lot d'images avec gestion de la mémoire (version simple)."""
    results = {
        'sizes': [],
        'means': [],
        'stds': [],
        'file_sizes': [],
        'formats': []
    }
    
    total_files = len(image_paths)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed = 0
    start_time = time.time()
    
    # Traitement par batch
    for i in range(0, total_files, batch_size):
        batch = image_paths[i:i + batch_size]
        
        for img_path in batch:
            try:
                with Image.open(img_path) as img:
                    # Convertir en array numpy
                    img_array = np.array(img)
                    
                    # Collecter les statistiques
                    results['sizes'].append(img.size)
                    results['means'].append(np.mean(img_array))
                    results['stds'].append(np.std(img_array))
                    results['file_sizes'].append(img_path.stat().st_size)
                    results['formats'].append(img.format or 'PNG')
                    
            except Exception as e:
                st.warning(f"⚠️ Erreur avec {img_path.name}: {e}")
                continue
            
            processed += 1
            
            # Mise à jour de la progression
            progress = processed / total_files
            progress_bar.progress(progress)
            
            # Estimation du temps restant
            elapsed = time.time() - start_time
            if processed > 0:
                eta = (elapsed / processed) * (total_files - processed)
                status_text.text(f"Traitement: {processed}/{total_files} images - ETA: {eta:.1f}s")
    
    progress_bar.empty()
    status_text.empty()
    
    return results


def analyze_images_batch_enhanced(image_paths: List[Path], batch_size: int = 10) -> Dict:
    """Analyse un lot d'images avec gestion de la mémoire et métriques avancées."""
    results = {
        'sizes': [],
        'means': [],
        'stds': [],
        'file_sizes': [],
        'formats': [],
        'entropy': [],
        'sharpness': [],
        'contrast_ratio': [],
        'edge_density': [],
        'brightness': [],
        'skewness': [],
        'kurtosis': []
    }
    
    total_files = len(image_paths)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed = 0
    start_time = time.time()
    
    # Traitement par batch
    for i in range(0, total_files, batch_size):
        batch = image_paths[i:i + batch_size]
        
        for img_path in batch:
            try:
                with Image.open(img_path) as img:
                    # Convertir en array numpy
                    img_array = np.array(img.convert('L'))
                    
                    # Statistiques de base
                    results['sizes'].append(img.size)
                    results['means'].append(np.mean(img_array))
                    results['stds'].append(np.std(img_array))
                    results['file_sizes'].append(img_path.stat().st_size)
                    results['formats'].append(img.format or 'PNG')
                    
                    # Métriques avancées
                    results['entropy'].append(calculate_entropy(img_array))
                    results['sharpness'].append(calculate_sharpness(img_array))
                    results['contrast_ratio'].append(calculate_contrast_ratio(img_array))
                    results['edge_density'].append(calculate_edge_density(img_array))
                    results['brightness'].append(np.mean(img_array) / 255.0)  # Normaliser 0-1
                    
                    # Statistiques de distribution
                    from scipy import stats
                    results['skewness'].append(stats.skew(img_array.flatten()))
                    results['kurtosis'].append(stats.kurtosis(img_array.flatten()))
                    
            except Exception as e:
                st.warning(f"⚠️ Erreur avec {img_path.name}: {e}")
                continue
            
            processed += 1
            
            # Mise à jour de la progression
            progress = processed / total_files
            progress_bar.progress(progress)
            
            # Estimation du temps restant
            elapsed = time.time() - start_time
            if processed > 0:
                eta = (elapsed / processed) * (total_files - processed)
                status_text.text(f"Traitement: {processed}/{total_files} images - ETA: {eta:.1f}s")
    
    progress_bar.empty()
    status_text.empty()
    
    return results
    """Analyse un lot d'images avec gestion de la mémoire."""
    results = {
        'sizes': [],
        'means': [],
        'stds': [],
        'file_sizes': [],
        'formats': []
    }
    
    total_files = len(image_paths)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed = 0
    start_time = time.time()
    
    # Traitement par batch
    for i in range(0, total_files, batch_size):
        batch = image_paths[i:i + batch_size]
        
        for img_path in batch:
            try:
                with Image.open(img_path) as img:
                    # Convertir en array numpy
                    img_array = np.array(img)
                    
                    # Collecter les statistiques
                    results['sizes'].append(img.size)
                    results['means'].append(np.mean(img_array))
                    results['stds'].append(np.std(img_array))
                    results['file_sizes'].append(img_path.stat().st_size)
                    results['formats'].append(img.format or 'PNG')
                    
            except Exception as e:
                st.warning(f"⚠️ Erreur avec {img_path.name}: {e}")
                continue
            
            processed += 1
            
            # Mise à jour de la progression
            progress = processed / total_files
            progress_bar.progress(progress)
            
            # Estimation du temps restant
            elapsed = time.time() - start_time
            if processed > 0:
                eta = (elapsed / processed) * (total_files - processed)
                status_text.text(f"Traitement: {processed}/{total_files} images - ETA: {eta:.1f}s")
    
    progress_bar.empty()
    status_text.empty()
    
    return results


def display_analysis_results(results: Dict, category: str, total_processed: int):
    """Affiche les résultats d'analyse de manière attractive."""
    st.markdown(f"### 📊 **Résultats pour {category}** ({total_processed} images)")
    
    if not results['sizes']:
        st.warning("Aucune donnée à afficher")
        return
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_width = np.mean([size[0] for size in results['sizes']])
        st.metric("Largeur moyenne", f"{avg_width:.0f}px")
    
    with col2:
        avg_height = np.mean([size[1] for size in results['sizes']])
        st.metric("Hauteur moyenne", f"{avg_height:.0f}px")
    
    with col3:
        avg_intensity = np.mean(results['means'])
        st.metric("Intensité moyenne", f"{avg_intensity:.1f}")
    
    with col4:
        avg_file_size = np.mean(results['file_sizes']) / 1024  # KB
        st.metric("Taille fichier moy.", f"{avg_file_size:.1f} KB")
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogramme des intensités
        fig_hist = px.histogram(
            x=results['means'],
            title="Distribution des Intensités Moyennes",
            labels={'x': 'Intensité moyenne', 'count': 'Nombre d\'images'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Scatter plot taille vs intensité
        widths = [size[0] for size in results['sizes']]
        fig_scatter = px.scatter(
            x=widths,
            y=results['means'],
            title="Taille vs Intensité",
            labels={'x': 'Largeur (px)', 'y': 'Intensité moyenne'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


def show_radiological_analysis(data_dir, categories):
    """Affiche une analyse radiologique spécialisée pour les images médicales."""
    st.markdown("### 🏥 Analyse Radiologique Spécialisée")
    
    st.info("🔬 **Analyse spécialisée** pour l'imagerie médicale pulmonaire")
    
    # Paramètres d'analyse radiologique
    col1, col2 = st.columns(2)
    
    with col1:
        selected_categories = st.multiselect(
            "Catégories à comparer:",
            categories,
            default=categories[:2] if len(categories) >= 2 else categories,
            key="radio_categories"
        )
    
    with col2:
        analysis_focus = st.selectbox(
            "Focus d'analyse:",
            [
                "Densité pulmonaire",
                "Motifs de texture",
                "Symétrie thoracique",
                "Analyse comparative"
            ],
            key="radio_focus"
        )
    
    if len(selected_categories) < 2:
        st.warning("⚠️ Sélectionnez au moins 2 catégories pour l'analyse comparative.")
        return
    
    sample_per_category = st.slider("Échantillons par catégorie:", 5, 30, 10, key="radio_samples")
    
    if st.button("🏥 Lancer l'analyse radiologique", key="radio_analysis_btn"):
        st.info("🚧 Analyse radiologique avancée en développement...")
        st.markdown("### Fonctionnalités prévues:")
        st.markdown("- Analyse de densité pulmonaire")
        st.markdown("- Détection de motifs pathologiques")
        st.markdown("- Comparaison de symétrie thoracique")
        st.markdown("- Tests statistiques entre catégories")


def analyze_basic_properties(data_dir, category, sample_size):
    """Analyse les propriétés de base des images."""
    images_dir = Path(data_dir) / category / "images"
    
    if not images_dir.exists():
        st.error(f"Dossier non trouvé: {images_dir}")
        return
    
    image_files = list(images_dir.glob("*.png"))[:sample_size]
    
    if not image_files:
        st.warning("Aucune image trouvée.")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    properties = []
    
    for i, image_path in enumerate(image_files):
        try:
            status_text.text(f"Analyse de {image_path.name}...")
            progress_bar.progress((i + 1) / len(image_files))
            
            img = Image.open(image_path)
            img_array = np.array(img.convert('L'))
            
            properties.append({
                "Image": image_path.name,
                "Largeur": img.size[0],
                "Hauteur": img.size[1],
                "Pixels_total": img.size[0] * img.size[1],
                "Intensité_min": np.min(img_array),
                "Intensité_max": np.max(img_array),
                "Intensité_moyenne": np.mean(img_array),
                "Écart_type": np.std(img_array),
                "Médiane": np.median(img_array),
                "Contraste": np.max(img_array) - np.min(img_array)
            })
            
        except Exception as e:
            st.error(f"Erreur avec {image_path.name}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    if properties:
        df_props = pd.DataFrame(properties)
        
        st.markdown("### 📊 Propriétés des Images")
        st.dataframe(df_props, use_container_width=True)
        
        # Visualisations
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des dimensions
            fig_dims = px.scatter(
                df_props,
                x="Largeur",
                y="Hauteur",
                title="Distribution des Dimensions",
                hover_data=["Image"]
            )
            st.plotly_chart(fig_dims, use_container_width=True)
        
        with col2:
            # Distribution de l'intensité moyenne
            fig_intensity = px.histogram(
                df_props,
                x="Intensité_moyenne",
                title="Distribution de l'Intensité Moyenne",
                marginal="box"
            )
            st.plotly_chart(fig_intensity, use_container_width=True)
        
        # Statistiques résumées
        with st.expander("📈 Statistiques résumées", expanded=False):
            st.write(df_props.describe())


def show_basic_image_analysis(data_dir, categories):
    """Analyse avancée des propriétés des images avec différents modes."""
    st.markdown("### 📊 **Analyse Complète des Images**")
    
    # Configuration du mode d'analyse
    config = get_analysis_mode_config("basic_analysis")
    
    # Sélection de la catégorie
    col1, col2 = st.columns(2)
    
    with col1:
        selected_category = st.selectbox(
            "Catégorie à analyser:",
            categories,
            key="enhanced_analysis_category"
        )
    
    with col2:
        # Estimation du nombre d'images à traiter
        images_dir = Path(data_dir) / selected_category / "images"
        if images_dir.exists():
            total_images = len(list(images_dir.glob("*.png")))
            files_to_process = get_image_files_for_analysis(Path(data_dir), selected_category, config)
            st.info(f"📂 **{len(files_to_process)}** images seront analysées sur **{total_images}** disponibles")
        else:
            st.warning("⚠️ Dossier d'images non trouvé")
            return
    
    # Options d'analyse avancées
    with st.expander("⚙️ Options avancées"):
        save_results = st.checkbox("💾 Sauvegarder les résultats", value=False)
        show_detailed_stats = st.checkbox("📈 Statistiques détaillées", value=True)
        batch_size = st.slider("Taille des lots (pour la mémoire):", 5, 50, 10)
    
    # Bouton d'analyse
    if st.button("🚀 Lancer l'analyse", key="enhanced_analysis_btn", type="primary"):
        with st.spinner("🔍 Analyse en cours..."):
            
            # Récupération des fichiers à analyser
            image_files = get_image_files_for_analysis(Path(data_dir), selected_category, config)
            
            if not image_files:
                st.error("❌ Aucune image trouvée pour l'analyse")
                return
            
            st.success(f"✅ Analyse de **{len(image_files)}** images de la catégorie **{selected_category}**")
            
            # Analyse par lots
            results = analyze_images_batch(image_files, batch_size)
            
            # Affichage des résultats
            display_analysis_results(results, selected_category, len(image_files))
            
            # Statistiques détaillées si demandées
            if show_detailed_stats and results['sizes']:
                st.markdown("### � **Statistiques Détaillées**")
                
                # Créer un DataFrame pour les statistiques
                stats_df = pd.DataFrame({
                    'Largeur': [size[0] for size in results['sizes']],
                    'Hauteur': [size[1] for size in results['sizes']],
                    'Intensité_Moyenne': results['means'],
                    'Écart_Type': results['stds'],
                    'Taille_Fichier_KB': [size/1024 for size in results['file_sizes']]
                })
                
                # Statistiques descriptives
                st.dataframe(stats_df.describe(), use_container_width=True)
                
                # Graphiques de distribution supplémentaires
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_size_dist = px.histogram(
                        stats_df, 
                        x='Taille_Fichier_KB',
                        title="Distribution des Tailles de Fichier",
                        labels={'Taille_Fichier_KB': 'Taille (KB)'}
                    )
                    st.plotly_chart(fig_size_dist, use_container_width=True)
                
                with col2:
                    fig_intensity_std = px.scatter(
                        x=stats_df['Intensité_Moyenne'],
                        y=stats_df['Écart_Type'],
                        title="Intensité vs Variabilité",
                        labels={'x': 'Intensité moyenne', 'y': 'Écart-type'}
                    )
                    st.plotly_chart(fig_intensity_std, use_container_width=True)
            
            # Sauvegarde si demandée
            if save_results:
                # Créer un résumé des résultats
                summary = {
                    'category': selected_category,
                    'total_images': len(image_files),
                    'analysis_mode': config['mode'],
                    'avg_width': np.mean([size[0] for size in results['sizes']]),
                    'avg_height': np.mean([size[1] for size in results['sizes']]),
                    'avg_intensity': np.mean(results['means']),
                    'avg_file_size_kb': np.mean(results['file_sizes']) / 1024
                }
                
                # Stocker dans le session state pour utilisation ultérieure
                if 'analysis_results' not in st.session_state:
                    st.session_state['analysis_results'] = []
                
                st.session_state['analysis_results'].append(summary)
                st.success("💾 Résultats sauvegardés dans la session")
                
                # Afficher un résumé des analyses sauvegardées
                if len(st.session_state['analysis_results']) > 1:
                    st.markdown("### 📋 **Comparaison avec les Analyses Précédentes**")
                    comparison_df = pd.DataFrame(st.session_state['analysis_results'])
                    st.dataframe(comparison_df, use_container_width=True)


def show_random_image_grid(data_dir, categories):
    """Affiche une grille d'images aléatoires avec analyse instantanée."""
    st.markdown("## 🎲 **Échantillons Aléatoires**")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🔄 Générer de nouveaux échantillons", use_container_width=True, key="generate_random_samples"):
            st.rerun()
    
    with col2:
        grid_size = st.selectbox("Taille de la grille:", [2, 3, 4], index=1, key="random_grid_size")
    
    # Génération d'échantillons aléatoires
    cols = st.columns(grid_size)
    
    for i in range(grid_size * 2):  # 2 lignes
        col_idx = i % grid_size
        
        # Sélection aléatoire d'une catégorie et d'une image
        category = random.choice(categories)
        images_dir = Path(data_dir) / category / "images"
        
        if images_dir.exists():
            image_files = list(images_dir.glob("*.png"))
            if image_files:
                image_path = random.choice(image_files)
                
                with cols[col_idx]:
                    try:
                        # Affichage de l'image
                        img = Image.open(image_path)
                        st.image(img, caption=f"{category}\n{image_path.name}", use_column_width=True)
                        
                        # Analyse rapide
                        with st.expander("📊 Analyse rapide", expanded=False):
                            # Statistiques de base
                            img_array = np.array(img.convert('L'))
                            
                            st.write(f"**Dimensions:** {img.size[0]} × {img.size[1]}")
                            st.write(f"**Intensité moyenne:** {np.mean(img_array):.1f}")
                            st.write(f"**Contraste (std):** {np.std(img_array):.1f}")
                            
                            # Histogramme mini
                            hist_fig = px.histogram(
                                x=img_array.flatten(),
                                nbins=50,
                                title="Distribution d'intensité"
                            )
                            hist_fig.update_layout(
                                height=200,
                                showlegend=False,
                                margin=dict(l=0, r=0, t=30, b=0)
                            )
                            st.plotly_chart(hist_fig, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Erreur lors du chargement de {image_path.name}")


# Fonctions de calcul de métriques
def calculate_sharpness(img_array):
    """Calcule la netteté d'une image using Laplacian variance."""
    laplacian = ndimage.laplace(img_array)
    return np.var(laplacian)


def calculate_noise_level(img_array):
    """Estime le niveau de bruit dans une image."""
    # Utilisation d'un filtre médian pour estimer le bruit
    filtered = median_filter(img_array, size=3)
    noise = img_array.astype(float) - filtered.astype(float)
    return np.std(noise)


def calculate_contrast_ratio(img_array):
    """Calcule le ratio de contraste."""
    return np.std(img_array) / np.mean(img_array) if np.mean(img_array) > 0 else 0


def calculate_entropy(img_array):
    """Calcule l'entropie de l'image."""
    hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
    hist = hist[hist > 0]  # Éviter log(0)
    prob = hist / hist.sum()
    return -np.sum(prob * np.log2(prob))


def calculate_homogeneity(img_array):
    """Calcule l'homogénéité de texture."""
    try:
        # Matrice de co-occurrence simplifiée
        if img_array.shape[0] > 1 and img_array.shape[1] > 1:
            diff = np.abs(img_array[:-1, :-1] - img_array[1:, 1:])
            mean_diff = np.mean(diff)
            return 1.0 / (1.0 + mean_diff)
        else:
            return 1.0  # Image trop petite, considérée comme homogène
    except Exception as e:
        return 0.0


def calculate_energy(img_array):
    """Calcule l'énergie de texture."""
    hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
    prob = hist / hist.sum()
    return np.sum(prob ** 2)


def calculate_edge_density(img_array):
    """Calcule la densité des contours."""
    # Détection de contours avec Sobel
    sobel_x = ndimage.sobel(img_array, axis=0)
    sobel_y = ndimage.sobel(img_array, axis=1)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Seuillage pour binariser les contours
    threshold = np.mean(edges) + np.std(edges)
    edge_pixels = np.sum(edges > threshold)
    
    return edge_pixels / img_array.size


def calculate_edge_strength(img_array):
    """Calcule la force moyenne des contours."""
    sobel_x = ndimage.sobel(img_array, axis=0)
    sobel_y = ndimage.sobel(img_array, axis=1)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    
    return np.mean(edges)


def calculate_gradient_magnitude(img_array):
    """Calcule la magnitude du gradient."""
    grad_x = ndimage.sobel(img_array, axis=0)
    grad_y = ndimage.sobel(img_array, axis=1)
    
    return np.mean(np.sqrt(grad_x**2 + grad_y**2))


# ====================================================================
# 🔬 NOUVELLES FONCTIONS POUR MÉTRIQUES AVANCÉES
# ====================================================================

def calculate_sobel_sharpness(img_array):
    """Calcule la netteté basée sur l'opérateur Sobel."""
    sobel_x = ndimage.sobel(img_array, axis=0)
    sobel_y = ndimage.sobel(img_array, axis=1)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.var(sobel_magnitude)


def calculate_variance_sharpness(img_array):
    """Calcule la netteté basée sur la variance des gradients."""
    grad_x = np.gradient(img_array, axis=0)
    grad_y = np.gradient(img_array, axis=1)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.var(gradient_magnitude)


def calculate_rms_contrast(img_array):
    """Calcule le contraste RMS (Root Mean Square)."""
    mean_intensity = np.mean(img_array)
    return np.sqrt(np.mean((img_array - mean_intensity)**2))


def calculate_michelson_contrast(img_array):
    """Calcule le contraste de Michelson."""
    max_intensity = np.max(img_array)
    min_intensity = np.min(img_array)
    if max_intensity + min_intensity == 0:
        return 0
    return (max_intensity - min_intensity) / (max_intensity + min_intensity)


def detect_blur_level(img_array):
    """Détecte le niveau de flou dans l'image."""
    # Utilise la variance du Laplacien pour détecter le flou
    laplacian = ndimage.laplace(img_array)
    return np.var(laplacian)


def detect_artifacts(img_array):
    """Détecte les artefacts dans l'image."""
    # Détection simplifiée basée sur les variations locales extrêmes
    grad_x = np.gradient(img_array, axis=0)
    grad_y = np.gradient(img_array, axis=1)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Score basé sur les gradients extrêmes
    threshold = np.mean(gradient_magnitude) + 3 * np.std(gradient_magnitude)
    artifacts = np.sum(gradient_magnitude > threshold)
    return artifacts / img_array.size


def calculate_ssim(img1, img2):
    """Calcule l'indice de similarité structurelle (SSIM)."""
    try:
        if img1.shape != img2.shape:
            # Redimensionner si nécessaire vers la plus petite dimension commune
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1_resized = img1[:min_height, :min_width].astype(float)
            img2_resized = img2[:min_height, :min_width].astype(float)
        else:
            img1_resized = img1.astype(float)
            img2_resized = img2.astype(float)
        
        # Vérifier que les images ont une taille suffisante
        if img1_resized.size == 0 or img2_resized.size == 0:
            return 0.0
        
        # Paramètres SSIM
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        
        mu1 = np.mean(img1_resized)
        mu2 = np.mean(img2_resized)
        sigma1_sq = np.var(img1_resized)
        sigma2_sq = np.var(img2_resized)
        
        # Calculer la covariance de manière sécurisée
        if img1_resized.size > 1:
            sigma12 = np.cov(img1_resized.flatten(), img2_resized.flatten())[0, 1]
        else:
            sigma12 = 0
        
        # Calcul SSIM avec vérification de division par zéro
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        
        if denominator == 0:
            return 1.0 if mu1 == mu2 else 0.0
        
        ssim = numerator / denominator
        return float(ssim)
        
    except Exception as e:
        return 0.0


def calculate_mse(img1, img2):
    """Calcule l'erreur quadratique moyenne (MSE)."""
    try:
        if img1.shape != img2.shape:
            # Redimensionner vers la plus petite dimension commune
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1_resized = img1[:min_height, :min_width].astype(float)
            img2_resized = img2[:min_height, :min_width].astype(float)
        else:
            img1_resized = img1.astype(float)
            img2_resized = img2.astype(float)
        
        # Vérifier que les images ont une taille suffisante
        if img1_resized.size == 0 or img2_resized.size == 0:
            return 0.0
        
        mse = np.mean((img1_resized - img2_resized)**2)
        return float(mse)
        
    except Exception as e:
        return 0.0


def calculate_psnr(img1, img2):
    """Calcule le rapport signal/bruit de crête (PSNR)."""
    try:
        mse = calculate_mse(img1, img2)
        if mse == 0:
            return float('inf')
        if mse < 0:
            return 0.0
        
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        return float(psnr)
        
    except Exception as e:
        return 0.0


def calculate_glcm_contrast(img_array):
    """Calcule le contraste GLCM (Gray Level Co-occurrence Matrix)."""
    try:
        # Simplification : utilise la différence entre pixels adjacents
        if img_array.shape[1] > 1:  # Au moins 2 colonnes
            diff_h = np.abs(img_array[:, :-1] - img_array[:, 1:])
            contrast_h = np.mean(diff_h)
        else:
            contrast_h = 0
            
        if img_array.shape[0] > 1:  # Au moins 2 lignes
            diff_v = np.abs(img_array[:-1, :] - img_array[1:, :])
            contrast_v = np.mean(diff_v)
        else:
            contrast_v = 0
            
        return contrast_h + contrast_v
        
    except Exception as e:
        return 0.0


def calculate_glcm_dissimilarity(img_array):
    """Calcule la dissimilarité GLCM."""
    try:
        if img_array.shape[1] > 1:  # Au moins 2 colonnes
            diff_h = np.abs(img_array[:, :-1] - img_array[:, 1:])
            dissim_h = np.mean(np.sqrt(diff_h**2))
        else:
            dissim_h = 0
            
        if img_array.shape[0] > 1:  # Au moins 2 lignes
            diff_v = np.abs(img_array[:-1, :] - img_array[1:, :])
            dissim_v = np.mean(np.sqrt(diff_v**2))
        else:
            dissim_v = 0
            
        return np.mean([dissim_h, dissim_v])
        
    except Exception as e:
        return 0.0


def calculate_glcm_correlation(img_array):
    """Calcule la corrélation GLCM."""
    try:
        # Corrélation entre pixels adjacents
        # Assurer que les dimensions sont cohérentes
        if img_array.shape[1] > 1:  # Vérifier qu'il y a au moins 2 colonnes
            pixels_h1 = img_array[:, :-1].flatten()
            pixels_h2 = img_array[:, 1:].flatten()
            
            # Vérifier que les arrays ont la même taille
            min_size = min(len(pixels_h1), len(pixels_h2))
            pixels_h1 = pixels_h1[:min_size]
            pixels_h2 = pixels_h2[:min_size]
            
            if len(pixels_h1) > 1 and np.std(pixels_h1) > 0 and np.std(pixels_h2) > 0:
                corr_h = np.corrcoef(pixels_h1, pixels_h2)[0, 1]
            else:
                corr_h = 0
        else:
            corr_h = 0
        
        if img_array.shape[0] > 1:  # Vérifier qu'il y a au moins 2 lignes
            pixels_v1 = img_array[:-1, :].flatten()
            pixels_v2 = img_array[1:, :].flatten()
            
            # Vérifier que les arrays ont la même taille
            min_size = min(len(pixels_v1), len(pixels_v2))
            pixels_v1 = pixels_v1[:min_size]
            pixels_v2 = pixels_v2[:min_size]
            
            if len(pixels_v1) > 1 and np.std(pixels_v1) > 0 and np.std(pixels_v2) > 0:
                corr_v = np.corrcoef(pixels_v1, pixels_v2)[0, 1]
            else:
                corr_v = 0
        else:
            corr_v = 0
        
        # Retourner la moyenne des corrélations si elles sont valides
        correlations = [corr_h, corr_v]
        valid_correlations = [c for c in correlations if not np.isnan(c) and not np.isinf(c)]
        
        return np.mean(valid_correlations) if valid_correlations else 0.0
        
    except Exception as e:
        return 0.0


def calculate_local_variance(img_array):
    """Calcule la variance locale de l'image."""
    from scipy.ndimage import generic_filter
    return np.mean(generic_filter(img_array.astype(float), np.var, size=5))


def calculate_local_std(img_array):
    """Calcule l'écart-type local de l'image."""
    from scipy.ndimage import generic_filter
    return np.mean(generic_filter(img_array.astype(float), np.std, size=5))


def calculate_texture_repetitiveness(img_array):
    """Calcule la répétitivité de la texture."""
    try:
        # Utilise l'autocorrélation pour détecter la répétitivité
        if img_array.size == 0:
            return 0.0
            
        fft = np.fft.fft2(img_array)
        power_spectrum = np.abs(fft)**2
        autocorr = np.fft.ifft2(power_spectrum).real
        
        mean_autocorr = np.mean(autocorr)
        if mean_autocorr > 0:
            return np.std(autocorr) / mean_autocorr
        else:
            return 0.0
    except Exception as e:
        return 0.0


def calculate_texture_regularity(img_array):
    """Calcule la régularité de la texture."""
    try:
        # Basé sur la variance des gradients locaux
        if img_array.size == 0:
            return 0.0
            
        grad_x = np.gradient(img_array, axis=0)
        grad_y = np.gradient(img_array, axis=1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        std_gradient = np.std(gradient_magnitude)
        return 1.0 / (1.0 + std_gradient)
    except Exception as e:
        return 0.0


def calculate_canny_edges(img_array):
    """Calcule la densité des contours avec l'algorithme Canny."""
    # Implémentation simplifiée du détecteur Canny
    sigma = 1.0
    # Lissage gaussien
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(img_array, sigma)
    
    # Gradients
    grad_x = ndimage.sobel(smoothed, axis=0)
    grad_y = ndimage.sobel(smoothed, axis=1)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Seuillage simple
    threshold = np.mean(gradient_magnitude) + np.std(gradient_magnitude)
    edges = gradient_magnitude > threshold
    
    return np.sum(edges) / img_array.size


def calculate_prewitt_edges(img_array):
    """Calcule la densité des contours avec l'opérateur Prewitt."""
    # Opérateur Prewitt
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    from scipy.ndimage import convolve
    edge_x = convolve(img_array.astype(float), prewitt_x)
    edge_y = convolve(img_array.astype(float), prewitt_y)
    
    edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)
    threshold = np.mean(edge_magnitude) + np.std(edge_magnitude)
    
    return np.sum(edge_magnitude > threshold) / img_array.size


def calculate_dominant_orientation(img_array):
    """Calcule l'orientation dominante des contours."""
    grad_x = ndimage.sobel(img_array, axis=0)
    grad_y = ndimage.sobel(img_array, axis=1)
    
    # Calcul de l'angle d'orientation
    angles = np.arctan2(grad_y, grad_x)
    angles = angles[np.abs(grad_x) + np.abs(grad_y) > np.mean(np.abs(grad_x) + np.abs(grad_y))]
    
    if len(angles) > 0:
        # Histogramme des angles
        hist, bins = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
        dominant_bin = np.argmax(hist)
        return bins[dominant_bin]
    return 0


def calculate_orientation_coherence(img_array):
    """Calcule la cohérence d'orientation des contours."""
    grad_x = ndimage.sobel(img_array, axis=0)
    grad_y = ndimage.sobel(img_array, axis=1)
    
    angles = np.arctan2(grad_y, grad_x)
    angles = angles[np.abs(grad_x) + np.abs(grad_y) > np.mean(np.abs(grad_x) + np.abs(grad_y))]
    
    if len(angles) > 0:
        # Cohérence basée sur la variance des angles
        return 1.0 / (1.0 + np.var(angles))
    return 0


def calculate_circularity(img_array):
    """Calcule la circularité des formes dans l'image."""
    # Détection de contours
    edges = ndimage.sobel(img_array) > np.mean(ndimage.sobel(img_array))
    
    # Approximation de la circularité basée sur la forme des contours
    if np.sum(edges) > 0:
        # Utilise le moment d'inertie pour approximer la circularité
        y, x = np.where(edges)
        if len(x) > 0 and len(y) > 0:
            center_x, center_y = np.mean(x), np.mean(y)
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            return 1.0 / (1.0 + np.std(distances) / np.mean(distances)) if np.mean(distances) > 0 else 0
    return 0


def calculate_rectangularity(img_array):
    """Calcule la rectangularité des formes dans l'image."""
    # Basé sur l'analyse des gradients horizontaux et verticaux
    grad_x = ndimage.sobel(img_array, axis=0)
    grad_y = ndimage.sobel(img_array, axis=1)
    
    horizontal_edges = np.sum(np.abs(grad_x) > np.abs(grad_y))
    vertical_edges = np.sum(np.abs(grad_y) > np.abs(grad_x))
    total_edges = horizontal_edges + vertical_edges
    
    if total_edges > 0:
        balance = min(horizontal_edges, vertical_edges) / total_edges
        return balance * 2  # Normalisation
    return 0


def calculate_shape_complexity(img_array):
    """Calcule la complexité des formes dans l'image."""
    # Basé sur la densité et la variabilité des contours
    edges = calculate_edge_density(img_array)
    edge_strength_var = np.var(ndimage.sobel(img_array))
    
    return edges * edge_strength_var / 1000  # Normalisation


# Fonctions pour l'analyse statistique approfondie

def perform_shapiro_test(data_sample):
    """Effectue le test de normalité de Shapiro-Wilk."""
    if len(data_sample) > 5000:
        data_sample = np.random.choice(data_sample, 5000, replace=False)
    
    try:
        from scipy.stats import shapiro
        stat, p_value = shapiro(data_sample)
        return p_value
    except:
        return 0.5  # Valeur par défaut


def perform_ks_test(data_sample):
    """Effectue le test de Kolmogorov-Smirnov."""
    try:
        from scipy.stats import kstest
        stat, p_value = kstest(data_sample, 'norm')
        return p_value
    except:
        return 0.5


def calculate_local_entropy(img_array):
    """Calcule l'entropie locale de l'image."""
    from scipy.ndimage import generic_filter
    
    def local_entropy(window):
        hist, _ = np.histogram(window, bins=16, range=(0, 255))
        hist = hist[hist > 0]
        prob = hist / hist.sum()
        return -np.sum(prob * np.log2(prob))
    
    entropy_map = generic_filter(img_array.astype(float), local_entropy, size=5)
    return np.mean(entropy_map)


def calculate_distribution_uniformity(img_array):
    """Calcule l'uniformité de la distribution des intensités."""
    hist, _ = np.histogram(img_array, bins=256, range=(0, 255))
    expected_freq = img_array.size / 256
    chi_square = np.sum((hist - expected_freq)**2 / expected_freq)
    return 1.0 / (1.0 + chi_square / img_array.size)


# Fonctions pour l'analyse des couleurs

def calculate_average_saturation(img_array_color):
    """Calcule la saturation moyenne."""
    if len(img_array_color.shape) == 3:
        r, g, b = img_array_color[:,:,0], img_array_color[:,:,1], img_array_color[:,:,2]
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
        return np.mean(saturation)
    return 0


def calculate_average_brightness(img_array_color):
    """Calcule la luminosité moyenne."""
    if len(img_array_color.shape) == 3:
        r, g, b = img_array_color[:,:,0], img_array_color[:,:,1], img_array_color[:,:,2]
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        return np.mean(brightness) / 255.0
    return np.mean(img_array_color) / 255.0


def calculate_color_entropy(img_array_color):
    """Calcule l'entropie des couleurs."""
    if len(img_array_color.shape) == 3:
        # Réduire la résolution des couleurs pour le calcul
        quantized = (img_array_color // 32) * 32
        unique_colors, counts = np.unique(quantized.reshape(-1, 3), axis=0, return_counts=True)
        prob = counts / counts.sum()
        return -np.sum(prob * np.log2(prob))
    return 0


def calculate_color_contrast(img_array_color):
    """Calcule le contraste colorimétrique."""
    if len(img_array_color.shape) == 3:
        r, g, b = img_array_color[:,:,0], img_array_color[:,:,1], img_array_color[:,:,2]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return np.std(luminance)
    return np.std(img_array_color)


# Fonctions pour l'analyse fréquentielle

def find_dominant_frequency(fft):
    """Trouve la fréquence dominante."""
    magnitude = np.abs(fft)
    max_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    return max_idx


def calculate_frequency_distribution(magnitude_spectrum):
    """Calcule la répartition des fréquences."""
    total_energy = np.sum(magnitude_spectrum**2)
    if total_energy > 0:
        center = np.array(magnitude_spectrum.shape) // 2
        y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
        distances = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Énergie en fonction de la distance du centre
        max_dist = np.max(distances)
        bins = 10
        freq_energy = np.zeros(bins)
        
        for i in range(bins):
            mask = (distances >= i * max_dist / bins) & (distances < (i + 1) * max_dist / bins)
            freq_energy[i] = np.sum(magnitude_spectrum[mask]**2)
        
        return freq_energy / total_energy
    return np.zeros(10)


def detect_horizontal_periodicity(img_array):
    """Détecte la périodicité horizontale."""
    # Autocorrélation horizontale
    row_means = np.mean(img_array, axis=0)
    autocorr = np.correlate(row_means, row_means, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    
    # Trouve les pics pour détecter la périodicité
    if len(autocorr) > 1:
        peaks = autocorr[1:] - autocorr[:-1]
        return np.std(peaks) / np.mean(autocorr) if np.mean(autocorr) > 0 else 0
    return 0


def detect_vertical_periodicity(img_array):
    """Détecte la périodicité verticale."""
    # Autocorrélation verticale
    col_means = np.mean(img_array, axis=1)
    autocorr = np.correlate(col_means, col_means, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    
    if len(autocorr) > 1:
        peaks = autocorr[1:] - autocorr[:-1]
        return np.std(peaks) / np.mean(autocorr) if np.mean(autocorr) > 0 else 0
    return 0


# Fonctions pour l'analyse médicale

def calculate_horizontal_symmetry(img_array):
    """Calcule la symétrie horizontale de l'image."""
    height, width = img_array.shape
    left_half = img_array[:, :width//2]
    right_half = np.fliplr(img_array[:, width//2:])
    
    # Assurer que les deux moitiés ont la même taille
    min_width = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, :min_width]
    right_half = right_half[:, :min_width]
    
    # Calcul de la similarité
    diff = np.abs(left_half.astype(float) - right_half.astype(float))
    return 1.0 / (1.0 + np.mean(diff))


def calculate_vertical_symmetry(img_array):
    """Calcule la symétrie verticale de l'image."""
    height, width = img_array.shape
    top_half = img_array[:height//2, :]
    bottom_half = np.flipud(img_array[height//2:, :])
    
    # Assurer que les deux moitiés ont la même taille
    min_height = min(top_half.shape[0], bottom_half.shape[0])
    top_half = top_half[:min_height, :]
    bottom_half = bottom_half[:min_height, :]
    
    diff = np.abs(top_half.astype(float) - bottom_half.astype(float))
    return 1.0 / (1.0 + np.mean(diff))


def detect_abnormal_texture_patterns(img_array):
    """Détecte les patterns de texture anormaux."""
    # Basé sur l'entropie locale et la variance
    local_entropy = calculate_local_entropy(img_array)
    local_variance = calculate_local_variance(img_array)
    
    # Score composite
    entropy_threshold = np.mean(local_entropy) + 2 * np.std(local_entropy)
    variance_threshold = np.mean(local_variance) + 2 * np.std(local_variance)
    
    abnormal_score = (local_entropy > entropy_threshold) + (local_variance > variance_threshold)
    return np.mean(abnormal_score)


def calculate_regional_homogeneity(img_array):
    """Calcule l'homogénéité régionale de l'image."""
    # Divise l'image en régions et calcule l'homogénéité
    height, width = img_array.shape
    region_size = min(height, width) // 8
    
    if region_size < 5:
        return calculate_homogeneity(img_array)
    
    homogeneities = []
    for i in range(0, height - region_size, region_size):
        for j in range(0, width - region_size, region_size):
            region = img_array[i:i+region_size, j:j+region_size]
            homogeneities.append(calculate_homogeneity(region))
    
    return np.mean(homogeneities) if homogeneities else 0


# Fonctions pour l'analyse de complexité IA

def approximate_kolmogorov_complexity(img_array):
    """Approxime la complexité de Kolmogorov."""
    # Utilise la compression pour approximer la complexité
    import zlib
    compressed = zlib.compress(img_array.tobytes())
    return len(compressed) / img_array.size


def calculate_fractal_dimension(img_array):
    """Calcule la dimension fractale (box-counting)."""
    # Binarisation
    threshold = np.mean(img_array)
    binary = img_array > threshold
    
    # Box-counting method simplifié
    sizes = [2, 4, 8, 16, 32]
    counts = []
    
    for size in sizes:
        boxes = 0
        height, width = binary.shape
        for i in range(0, height, size):
            for j in range(0, width, size):
                box = binary[i:min(i+size, height), j:min(j+size, width)]
                if np.any(box):
                    boxes += 1
        counts.append(boxes)
    
    # Régression linéaire pour estimer la dimension
    if len(counts) > 1 and all(c > 0 for c in counts):
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)
        slope, _ = np.polyfit(log_sizes, log_counts, 1)
        return -slope
    return 1.0


def calculate_renyi_entropy(img_array, alpha=2):
    """Calcule l'entropie de Rényi."""
    hist, _ = np.histogram(img_array, bins=256, range=(0, 255))
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    
    if alpha == 1:
        return -np.sum(prob * np.log2(prob))
    else:
        return (1 / (1 - alpha)) * np.log2(np.sum(prob ** alpha))


def calculate_predictability_score(img_array):
    """Calcule un score de prédictibilité de l'image."""
    # Basé sur l'autocorrélation
    fft = np.fft.fft2(img_array)
    power_spectrum = np.abs(fft)**2
    autocorr = np.fft.ifft2(power_spectrum).real
    
    # Normalisation
    autocorr = autocorr / autocorr[0, 0] if autocorr[0, 0] != 0 else autocorr
    
    # Score basé sur la décroissance de l'autocorrélation
    center = np.array(autocorr.shape) // 2
    distances = []
    values = []
    
    for i in range(1, min(center)):
        if center[0] + i < autocorr.shape[0] and center[1] + i < autocorr.shape[1]:
            distances.append(i)
            values.append(autocorr[center[0] + i, center[1] + i])
    
    if len(values) > 1:
        return np.mean(values)
    return 0


def calculate_visual_complexity(img_array):
    """Calcule la complexité visuelle de l'image."""
    # Combinaison de plusieurs métriques
    entropy = calculate_entropy(img_array)
    edge_density = calculate_edge_density(img_array)
    local_variance = calculate_local_variance(img_array)
    
    # Score composite normalisé
    complexity = (entropy / 8) + (edge_density * 10) + (local_variance / 10000)
    return min(complexity, 1.0)  # Normalisation à [0, 1]


# Fonctions d'affichage des résultats

def display_comparative_summary(combined_df, selected_categories):
    """Affiche un résumé comparatif entre les catégories."""
    st.markdown("### 📊 **Résumé Comparatif Multi-Catégories**")
    
    # Métriques générales par catégorie
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📈 **Statistiques Générales**")
        
        # Tableau de résumé
        summary_data = []
        for category in selected_categories:
            cat_data = combined_df[combined_df['Categorie'] == category]
            if not cat_data.empty:
                summary_row = {
                    'Catégorie': category,
                    'Nombre d\'images': len(cat_data),
                    'Taille moy. (KB)': f"{cat_data['Taille_Fichier_KB'].mean():.1f}",
                    'Pixels moyens': f"{cat_data['Pixels_Total'].mean():.0f}"
                }
                
                # Ajouter des métriques si disponibles
                if 'Score_Qualite_Global' in cat_data.columns:
                    summary_row['Qualité moyenne'] = f"{cat_data['Score_Qualite_Global'].mean():.2f}"
                if 'Nettet_Gradient' in cat_data.columns:
                    summary_row['Netteté moyenne'] = f"{cat_data['Nettet_Gradient'].mean():.2f}"
                if 'Contraste_Michelson' in cat_data.columns:
                    summary_row['Contraste moyen'] = f"{cat_data['Contraste_Michelson'].mean():.3f}"
                
                summary_data.append(summary_row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 **Comparaison Visuelle**")
        
        # Graphique comparatif de la qualité
        if 'Score_Qualite_Global' in combined_df.columns:
            fig = px.box(
                combined_df, 
                x='Categorie', 
                y='Score_Qualite_Global',
                title="Distribution des Scores de Qualité par Catégorie",
                color='Categorie'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Graphique de la taille des fichiers
        if 'Taille_Fichier_KB' in combined_df.columns:
            fig = px.violin(
                combined_df, 
                x='Categorie', 
                y='Taille_Fichier_KB',
                title="Distribution des Tailles de Fichiers par Catégorie"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Analyses statistiques comparatives
    st.markdown("#### 🔍 **Analyses Statistiques Comparatives**")
    
    # Test ANOVA si nous avons des données quantitatives
    quantitative_columns = combined_df.select_dtypes(include=[np.number]).columns
    quantitative_columns = [col for col in quantitative_columns if col not in ['Pixels_Total', 'Taille_Fichier_KB']]
    
    if len(quantitative_columns) > 0 and len(selected_categories) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            selected_metric = st.selectbox(
                "Sélectionner une métrique pour l'analyse ANOVA:",
                quantitative_columns,
                key="anova_metric"
            )
        
        with col2:
            if st.button("🧮 Effectuer ANOVA", key="perform_anova"):
                try:
                    from scipy.stats import f_oneway
                    
                    groups = []
                    group_names = []
                    for category in selected_categories:
                        cat_data = combined_df[combined_df['Categorie'] == category][selected_metric].dropna()
                        if len(cat_data) > 0:
                            groups.append(cat_data)
                            group_names.append(category)
                    
                    if len(groups) >= 2:
                        f_stat, p_value = f_oneway(*groups)
                        
                        st.markdown("**Résultats ANOVA:**")
                        st.write(f"- F-statistique: {f_stat:.4f}")
                        st.write(f"- p-value: {p_value:.4f}")
                        
                        if p_value < 0.05:
                            st.success("✅ Différence significative entre les catégories (p < 0.05)")
                        else:
                            st.info("ℹ️ Pas de différence significative entre les catégories (p ≥ 0.05)")
                    else:
                        st.warning("⚠️ Pas assez de données pour effectuer l'ANOVA.")
                
                except Exception as e:
                    st.error(f"Erreur lors du calcul ANOVA: {e}")


def display_global_comparative_analysis(combined_df, analysis_options):
    """Affiche une analyse comparative globale détaillée."""
    st.markdown("### 🌍 **Analyse Comparative Globale Détaillée**")
    
    # Onglets pour différents types d'analyses
    comp_tabs = st.tabs([
        "📊 Comparaison Générale", 
        "🏆 Analyse de Qualité", 
        "🎨 Comparaison de Texture",
        "📈 Métriques Avancées",
        "🔍 Corrélations"
    ])
    
    with comp_tabs[0]:  # Comparaison Générale
        st.markdown("#### 📊 **Vue d'Ensemble Comparative**")
        
        # Graphiques de distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des catégories
            fig = px.pie(
                combined_df.groupby('Categorie').size().reset_index(name='count'),
                values='count',
                names='Categorie',
                title="Répartition des Images par Catégorie"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tailles des fichiers
            if 'Taille_Fichier_KB' in combined_df.columns:
                fig = px.histogram(
                    combined_df,
                    x='Taille_Fichier_KB',
                    color='Categorie',
                    title="Distribution des Tailles de Fichiers",
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Tableau de statistiques descriptives
        st.markdown("#### 📋 **Statistiques Descriptives par Catégorie**")
        
        numeric_columns = combined_df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['Pixels_Total']]
        
        if len(numeric_columns) > 0:
            stats_by_category = combined_df.groupby('Categorie')[numeric_columns].agg(['mean', 'std', 'min', 'max'])
            st.dataframe(stats_by_category.round(3), use_container_width=True)
    
    with comp_tabs[1]:  # Analyse de Qualité
        if "🏆 Qualité d'image complète" in str(analysis_options):
            st.markdown("#### 🏆 **Comparaison Détaillée de la Qualité**")
            
            quality_metrics = [col for col in combined_df.columns if any(q in col for q in 
                ['Score_Qualite', 'Nettet', 'Contraste', 'PSNR', 'MSE', 'SSIM'])]
            
            if quality_metrics:
                # Graphique radar comparatif
                categories = combined_df['Categorie'].unique()
                
                # Normaliser les métriques pour le radar
                normalized_data = combined_df[quality_metrics + ['Categorie']].copy()
                for metric in quality_metrics:
                    if normalized_data[metric].std() > 0:
                        normalized_data[metric] = (normalized_data[metric] - normalized_data[metric].min()) / (normalized_data[metric].max() - normalized_data[metric].min())
                
                # Créer le graphique radar
                fig = go.Figure()
                
                for category in categories:
                    cat_data = normalized_data[normalized_data['Categorie'] == category]
                    if not cat_data.empty:
                        avg_values = cat_data[quality_metrics].mean()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=avg_values.values,
                            theta=quality_metrics,
                            fill='toself',
                            name=category,
                            line_color=px.colors.qualitative.Set1[list(categories).index(category) % len(px.colors.qualitative.Set1)]
                        ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    title="Profil de Qualité Comparatif (Normalisé)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Comparaison métrique par métrique
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_quality_metric = st.selectbox(
                        "Sélectionner une métrique de qualité:",
                        quality_metrics,
                        key="quality_comparison_metric"
                    )
                
                with col2:
                    if selected_quality_metric in combined_df.columns:
                        fig = px.box(
                            combined_df,
                            x='Categorie',
                            y=selected_quality_metric,
                            title=f"Distribution de {selected_quality_metric}",
                            points="all"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
    
    with comp_tabs[2]:  # Comparaison de Texture
        if "🎨 Analyse de texture avancée" in str(analysis_options):
            st.markdown("#### 🎨 **Comparaison des Textures**")
            
            texture_metrics = [col for col in combined_df.columns if any(t in col for t in 
                ['GLCM', 'LBP', 'Texture', 'Homogeneit', 'Energie', 'Contrast'])]
            
            if texture_metrics:
                # Analyse en composantes principales des textures
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                
                texture_data = combined_df[texture_metrics].dropna()
                categories_clean = combined_df.loc[texture_data.index, 'Categorie']
                
                if len(texture_data) > 0 and len(texture_metrics) > 1:
                    # Standardisation
                    scaler = StandardScaler()
                    texture_scaled = scaler.fit_transform(texture_data)
                    
                    # PCA
                    pca = PCA(n_components=min(3, len(texture_metrics)))
                    texture_pca = pca.fit_transform(texture_scaled)
                    
                    # Création du DataFrame PCA
                    pca_df = pd.DataFrame(texture_pca, columns=[f'PC{i+1}' for i in range(texture_pca.shape[1])])
                    pca_df['Categorie'] = categories_clean.values
                    
                    # Graphique PCA 3D ou 2D
                    if texture_pca.shape[1] >= 3:
                        fig = px.scatter_3d(
                            pca_df,
                            x='PC1',
                            y='PC2',
                            z='PC3',
                            color='Categorie',
                            title="Analyse en Composantes Principales des Textures"
                        )
                    else:
                        fig = px.scatter(
                            pca_df,
                            x='PC1',
                            y='PC2',
                            color='Categorie',
                            title="Analyse en Composantes Principales des Textures"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Variance expliquée
                    st.write(f"**Variance expliquée par les composantes:** {pca.explained_variance_ratio_.round(3)}")
    
    with comp_tabs[3]:  # Métriques Avancées
        st.markdown("#### 📈 **Métriques Avancées Comparatives**")
        
        # Matrice de corrélation par catégorie
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['Pixels_Total', 'Taille_Fichier_KB']]
        
        if len(numeric_cols) > 1:
            selected_category_corr = st.selectbox(
                "Sélectionner une catégorie pour la matrice de corrélation:",
                ['Toutes'] + list(combined_df['Categorie'].unique()),
                key="correlation_category"
            )
            
            if selected_category_corr == 'Toutes':
                corr_data = combined_df[numeric_cols]
            else:
                corr_data = combined_df[combined_df['Categorie'] == selected_category_corr][numeric_cols]
            
            if not corr_data.empty:
                correlation_matrix = corr_data.corr()
                
                fig = px.imshow(
                    correlation_matrix,
                    title=f"Matrice de Corrélation - {selected_category_corr}",
                    color_continuous_scale='RdBu',
                    zmin=-1,
                    zmax=1
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
    
    with comp_tabs[4]:  # Corrélations
        st.markdown("#### 🔍 **Analyse des Corrélations Inter-Catégories**")
        
        # Sélection de deux métriques pour analyse
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['Pixels_Total', 'Taille_Fichier_KB']]
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                metric_x = st.selectbox("Métrique X:", numeric_cols, key="corr_metric_x")
            
            with col2:
                metric_y = st.selectbox("Métrique Y:", numeric_cols, index=1, key="corr_metric_y")
            
            if metric_x != metric_y:
                # Scatter plot avec droites de régression
                fig = px.scatter(
                    combined_df,
                    x=metric_x,
                    y=metric_y,
                    color='Categorie',
                    title=f"Corrélation: {metric_x} vs {metric_y}",
                    trendline="ols"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calcul des corrélations par catégorie
                correlations = {}
                for category in combined_df['Categorie'].unique():
                    cat_data = combined_df[combined_df['Categorie'] == category]
                    if len(cat_data) > 1:
                        corr = cat_data[metric_x].corr(cat_data[metric_y])
                        correlations[category] = corr
                
                if correlations:
                    st.markdown("**Corrélations par catégorie:**")
                    for cat, corr in correlations.items():
                        st.write(f"- {cat}: {corr:.3f}")


def display_export_options(results_df, analysis_options, category):
    """Affiche les options d'export des résultats."""
    st.markdown("---")
    st.markdown("### 💾 **Options d'Export**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📄 **Export CSV**")
        if st.button("💾 Télécharger CSV", key="export_csv"):
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Fichier CSV",
                data=csv_data,
                file_name=f"analyse_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        st.markdown("#### 📊 **Export Excel**")
        if st.button("💾 Télécharger Excel", key="export_excel"):
            try:
                from io import BytesIO
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    results_df.to_excel(writer, sheet_name='Resultats', index=False)
                    
                    # Feuille de métadonnées
                    metadata = pd.DataFrame({
                        'Paramètre': ['Catégorie', 'Nombre d\'images', 'Date d\'analyse', 'Types d\'analyses'],
                        'Valeur': [
                            category,
                            len(results_df),
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            ', '.join(analysis_options)
                        ]
                    })
                    metadata.to_excel(writer, sheet_name='Metadata', index=False)
                
                excel_data = output.getvalue()
                st.download_button(
                    label="📥 Fichier Excel",
                    data=excel_data,
                    file_name=f"analyse_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except ImportError:
                st.warning("⚠️ openpyxl non installé. Utilisez pip install openpyxl")
    
    with col3:
        st.markdown("#### 📈 **Export JSON**")
        if st.button("💾 Télécharger JSON", key="export_json"):
            json_data = results_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Fichier JSON",
                data=json_data,
                file_name=f"analyse_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


def display_comprehensive_results(results_df, analysis_options, category, processing_times):
    """Affiche les résultats complets avec visualisations avancées."""
    st.markdown("---")
    st.markdown("## 📊 **Résultats de l'Analyse Complète**")
    
    # Résumé exécutif
    st.markdown("### 🎯 **Résumé Exécutif**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_quality = results_df.get('Score_Qualite_Global', pd.Series([0])).mean()
        st.metric("🏆 Qualité Moyenne", f"{avg_quality:.3f}")
    
    with col2:
        total_processing_time = sum(processing_times.values())
        st.metric("⏱️ Temps Total", f"{total_processing_time:.2f}s")
    
    with col3:
        num_images = len(results_df)
        st.metric("📷 Images Analysées", f"{num_images}")
    
    with col4:
        avg_complexity = results_df.get('Complexite_Visuelle', pd.Series([0])).mean()
        st.metric("🧠 Complexité Moy.", f"{avg_complexity:.3f}")
    
    # Affichage par onglets selon les analyses sélectionnées
    tabs = []
    tab_names = []
    
    if "🏆 Qualité d'image complète" in analysis_options:
        tab_names.append("🏆 Qualité")
    if "🎨 Analyse de texture avancée" in analysis_options:
        tab_names.append("🎨 Texture")
    if "🔲 Détection de contours et formes" in analysis_options:
        tab_names.append("🔲 Contours")
    if "📊 Analyse statistique approfondie" in analysis_options:
        tab_names.append("📊 Statistiques")
    if "🌈 Analyse des couleurs" in analysis_options:
        tab_names.append("🌈 Couleurs")
    if "🔍 Analyse de fréquences" in analysis_options:
        tab_names.append("🔍 Fréquences")
    if "🏥 Métriques médicales" in analysis_options:
        tab_names.append("🏥 Médical")
    if "🤖 Score de complexité IA" in analysis_options:
        tab_names.append("🤖 IA")
    
    if tab_names:
        tabs = st.tabs(tab_names)
        
        tab_idx = 0
        
        if "🏆 Qualité d'image complète" in analysis_options:
            with tabs[tab_idx]:
                display_quality_results(results_df)
            tab_idx += 1
        
        if "🎨 Analyse de texture avancée" in analysis_options:
            with tabs[tab_idx]:
                display_texture_results(results_df)
            tab_idx += 1
        
        if "🔲 Détection de contours et formes" in analysis_options:
            with tabs[tab_idx]:
                display_edge_results(results_df)
            tab_idx += 1
        
        if "📊 Analyse statistique approfondie" in analysis_options:
            with tabs[tab_idx]:
                display_statistical_results(results_df)
            tab_idx += 1
        
        if "🌈 Analyse des couleurs" in analysis_options:
            with tabs[tab_idx]:
                display_color_results(results_df)
            tab_idx += 1
        
        if "🔍 Analyse de fréquences" in analysis_options:
            with tabs[tab_idx]:
                display_frequency_results(results_df)
            tab_idx += 1
        
        if "🏥 Métriques médicales" in analysis_options:
            with tabs[tab_idx]:
                display_medical_results(results_df)
            tab_idx += 1
        
        if "🤖 Score de complexité IA" in analysis_options:
            with tabs[tab_idx]:
                display_ai_results(results_df)
            tab_idx += 1


def display_quality_results(results_df):
    """Affiche les résultats d'analyse de qualité avec visualisations riches."""
    st.markdown("### 🏆 **Analyse de Qualité d'Image**")
    
    quality_cols = [col for col in results_df.columns if any(keyword in col for keyword in 
                   ['Nettete', 'Contraste', 'Bruit', 'Blur', 'Score_Qualite'])]
    
    if quality_cols:
        # Métriques clés en haut
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'Score_Qualite_Global' in results_df.columns:
                avg_quality = results_df['Score_Qualite_Global'].mean()
                st.metric("🏆 Qualité Moyenne", f"{avg_quality:.3f}")
        with col2:
            if 'Nettete_Laplacian' in results_df.columns:
                avg_sharpness = results_df['Nettete_Laplacian'].mean()
                st.metric("🔍 Netteté Moyenne", f"{avg_sharpness:.1f}")
        with col3:
            if 'Niveau_Bruit' in results_df.columns:
                avg_noise = results_df['Niveau_Bruit'].mean()
                st.metric("📢 Bruit Moyen", f"{avg_noise:.2f}")
        with col4:
            if 'Contraste_RMS' in results_df.columns:
                avg_contrast = results_df['Contraste_RMS'].mean()
                st.metric("🌈 Contraste Moyen", f"{avg_contrast:.1f}")
        
        # Tableau des résultats
        with st.expander("📊 Tableau détaillé des résultats"):
            st.dataframe(results_df[['Image'] + quality_cols], use_container_width=True)
        
        # Visualisations avancées
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Score_Qualite_Global' in results_df.columns:
                fig = px.histogram(
                    results_df, 
                    x='Score_Qualite_Global', 
                    title="📊 Distribution du Score de Qualité Global",
                    color_discrete_sequence=['#1f77b4'],
                    marginal="box"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Niveau_Bruit' in results_df.columns:
                fig = px.box(
                    results_df, 
                    y='Niveau_Bruit', 
                    title="📦 Distribution du Niveau de Bruit",
                    color_discrete_sequence=['#ff7f0e']
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # Graphiques de corrélation
        if len(quality_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Nettete_Laplacian' in results_df.columns and 'Contraste_RMS' in results_df.columns:
                    fig = px.scatter(
                        results_df,
                        x='Nettete_Laplacian',
                        y='Contraste_RMS',
                        title="🔍 Netteté vs Contraste",
                        hover_data=['Image'],
                        color_discrete_sequence=['#2ca02c']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Score_Qualite_Global' in results_df.columns and 'Niveau_Bruit' in results_df.columns:
                    fig = px.scatter(
                        results_df,
                        x='Niveau_Bruit',
                        y='Score_Qualite_Global',
                        title="📢 Bruit vs Qualité Globale",
                        hover_data=['Image'],
                        color_discrete_sequence=['#d62728']
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Graphique radar pour les métriques multiples
        if len(quality_cols) >= 3:
            st.markdown("#### 🕷️ **Profil Radar des Métriques de Qualité**")
            radar_metrics = ['Nettete_Laplacian', 'Contraste_RMS', 'Score_Qualite_Global']
            available_radar = [col for col in radar_metrics if col in results_df.columns]
            
            if len(available_radar) >= 3:
                # Créer un graphique radar moyen
                fig = go.Figure()
                
                # Normaliser les données pour le radar
                radar_data = {}
                for metric in available_radar:
                    if results_df[metric].max() != results_df[metric].min():
                        radar_data[metric] = (results_df[metric].mean() - results_df[metric].min()) / (results_df[metric].max() - results_df[metric].min())
                    else:
                        radar_data[metric] = 0.5
                
                fig.add_trace(go.Scatterpolar(
                    r=list(radar_data.values()),
                    theta=[m.replace('_', ' ') for m in available_radar],
                    fill='toself',
                    name='Qualité Moyenne'
                ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Profil de Qualité Moyen"
                )
                st.plotly_chart(fig, use_container_width=True)


def display_texture_results(results_df):
    """Affiche les résultats d'analyse de texture avec visualisations riches."""
    st.markdown("### 🎨 **Analyse de Texture Avancée**")
    
    texture_cols = [col for col in results_df.columns if any(keyword in col for keyword in 
                   ['Entropie', 'Homogeneite', 'Energie', 'GLCM', 'Texture', 'Variance_Locale'])]
    
    if texture_cols:
        # Métriques clés en haut
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'Entropie' in results_df.columns:
                avg_entropy = results_df['Entropie'].mean()
                st.metric("🌀 Entropie Moyenne", f"{avg_entropy:.2f}")
        with col2:
            if 'Homogeneite' in results_df.columns:
                avg_homogeneity = results_df['Homogeneite'].mean()
                st.metric("🔄 Homogénéité Moyenne", f"{avg_homogeneity:.3f}")
        with col3:
            if 'Variance_Locale' in results_df.columns:
                avg_variance = results_df['Variance_Locale'].mean()
                st.metric("📊 Variance Locale", f"{avg_variance:.1f}")
        with col4:
            if 'Contraste_GLCM' in results_df.columns:
                avg_glcm = results_df['Contraste_GLCM'].mean()
                st.metric("🎭 Contraste GLCM", f"{avg_glcm:.1f}")
        
        # Tableau détaillé dans un expander
        with st.expander("📊 Tableau détaillé des métriques de texture"):
            st.dataframe(results_df[['Image'] + texture_cols], use_container_width=True)
        
        # Visualisations principales
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Entropie' in results_df.columns:
                fig = px.histogram(
                    results_df, 
                    x='Entropie',
                    title="🌀 Distribution de l'Entropie",
                    color_discrete_sequence=['#9467bd'],
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Homogeneite' in results_df.columns:
                fig = px.histogram(
                    results_df, 
                    x='Homogeneite',
                    title="🔄 Distribution de l'Homogénéité",
                    color_discrete_sequence=['#8c564b'],
                    marginal="violin"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Graphiques de corrélation avancés
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Entropie' in results_df.columns and 'Homogeneite' in results_df.columns:
                fig = px.scatter(
                    results_df,
                    x='Entropie',
                    y='Homogeneite',
                    title="🌀 Entropie vs Homogénéité",
                    hover_data=['Image'],
                    color_discrete_sequence=['#e377c2']
                )
                # Ajouter une ligne de tendance
                from scipy import stats as scipy_stats
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                    results_df['Entropie'], results_df['Homogeneite']
                )
                line_x = np.linspace(results_df['Entropie'].min(), results_df['Entropie'].max(), 100)
                line_y = slope * line_x + intercept
                fig.add_scatter(x=line_x, y=line_y, mode='lines', name=f'Tendance (R²={r_value**2:.3f})')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Variance_Locale' in results_df.columns and 'Contraste_GLCM' in results_df.columns:
                fig = px.scatter(
                    results_df,
                    x='Variance_Locale',
                    y='Contraste_GLCM',
                    title="📊 Variance Locale vs Contraste GLCM",
                    hover_data=['Image'],
                    color_discrete_sequence=['#7f7f7f']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap des corrélations entre métriques de texture
        if len(texture_cols) > 2:
            st.markdown("#### 🔥 **Matrice de Corrélation des Métriques de Texture**")
            numeric_texture_cols = [col for col in texture_cols if results_df[col].dtype in ['float64', 'int64']]
            
            if len(numeric_texture_cols) > 1:
                corr_matrix = results_df[numeric_texture_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    title="Corrélations entre Métriques de Texture",
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    text_auto=True
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        # Graphique 3D si suffisamment de métriques
        if len(texture_cols) >= 3:
            available_3d = texture_cols[:3]  # Prendre les 3 premières métriques
            if all(col in results_df.columns for col in available_3d):
                st.markdown("#### 🌐 **Visualisation 3D des Métriques de Texture**")
                fig = px.scatter_3d(
                    results_df,
                    x=available_3d[0],
                    y=available_3d[1],
                    z=available_3d[2],
                    title="Analyse 3D des Métriques de Texture",
                    hover_data=['Image'],
                    opacity=0.7
                )
                st.plotly_chart(fig, use_container_width=True)


def display_edge_results(results_df):
    """Affiche les résultats d'analyse de contours avec visualisations riches."""
    st.markdown("### 🔲 **Analyse de Contours et Formes**")
    
    edge_cols = [col for col in results_df.columns if any(keyword in col for keyword in 
                ['Contours', 'Force', 'Orientation', 'Circularite', 'Rectangularite', 'Complexite_Forme'])]
    
    if edge_cols:
        # Métriques clés
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'Contours_Sobel' in results_df.columns:
                avg_sobel = results_df['Contours_Sobel'].mean()
                st.metric("🔍 Contours Sobel", f"{avg_sobel:.4f}")
        with col2:
            if 'Force_Contours' in results_df.columns:
                avg_strength = results_df['Force_Contours'].mean()
                st.metric("💪 Force Contours", f"{avg_strength:.1f}")
        with col3:
            if 'Circularite' in results_df.columns:
                avg_circularity = results_df['Circularite'].mean()
                st.metric("⭕ Circularité", f"{avg_circularity:.3f}")
        with col4:
            if 'Complexite_Forme' in results_df.columns:
                avg_complexity = results_df['Complexite_Forme'].mean()
                st.metric("🧩 Complexité", f"{avg_complexity:.4f}")
        
        # Tableau détaillé
        with st.expander("📊 Tableau détaillé des contours et formes"):
            st.dataframe(results_df[['Image'] + edge_cols], use_container_width=True)
        
        # Visualisations
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Contours_Sobel' in results_df.columns:
                fig = px.histogram(
                    results_df,
                    x='Contours_Sobel',
                    title="🔍 Distribution de la Densité de Contours",
                    color_discrete_sequence=['#ff7f0e'],
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Circularite' in results_df.columns and 'Rectangularite' in results_df.columns:
                fig = px.scatter(
                    results_df,
                    x='Circularite',
                    y='Rectangularite',
                    title="⭕ Circularité vs Rectangularité",
                    hover_data=['Image'],
                    color_discrete_sequence=['#2ca02c']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Graphique polaire pour les orientations
        if 'Orientation_Dominante' in results_df.columns:
            st.markdown("#### 🧭 **Distribution des Orientations Dominantes**")
            fig = go.Figure()
            
            # Convertir les orientations en degrés et créer un histogramme polaire
            orientations = results_df['Orientation_Dominante'] * 180 / np.pi  # Convertir en degrés
            theta_bins = np.linspace(0, 360, 37)  # 36 bins de 10 degrés
            hist, _ = np.histogram(orientations, bins=theta_bins)
            
            fig.add_trace(go.Scatterpolar(
                r=hist,
                theta=theta_bins[:-1],
                mode='lines+markers',
                fill='toself',
                name='Orientations'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, max(hist)]),
                    angularaxis=dict(tickmode='linear', tick0=0, dtick=45)
                ),
                title="Distribution Polaire des Orientations"
            )
            st.plotly_chart(fig, use_container_width=True)


def display_statistical_results(results_df):
    """Affiche les résultats d'analyse statistique avec visualisations riches."""
    st.markdown("### 📊 **Analyse Statistique Approfondie**")
    
    stats_cols = [col for col in results_df.columns if any(keyword in col for keyword in 
                 ['Moyenne', 'Mediane', 'Ecart_Type', 'Asymetrie', 'Aplatissement', 'Percentile', 'Test'])]
    
    if stats_cols:
        # Métriques clés
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'Moyenne' in results_df.columns:
                overall_mean = results_df['Moyenne'].mean()
                st.metric("📊 Intensité Moyenne", f"{overall_mean:.1f}")
        with col2:
            if 'Asymetrie' in results_df.columns:
                avg_skew = results_df['Asymetrie'].mean()
                st.metric("📈 Asymétrie Moyenne", f"{avg_skew:.2f}")
        with col3:
            if 'Aplatissement' in results_df.columns:
                avg_kurtosis = results_df['Aplatissement'].mean()
                st.metric("⛰️ Aplatissement", f"{avg_kurtosis:.2f}")
        with col4:
            if 'Coefficient_Variation' in results_df.columns:
                avg_cv = results_df['Coefficient_Variation'].mean()
                st.metric("📏 Coeff. Variation", f"{avg_cv:.3f}")
        
        # Tableau détaillé
        with st.expander("📊 Tableau détaillé des statistiques"):
            st.dataframe(results_df[['Image'] + stats_cols], use_container_width=True)
        
        # Visualisations statistiques
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Asymetrie' in results_df.columns and 'Aplatissement' in results_df.columns:
                fig = px.scatter(
                    results_df,
                    x='Asymetrie',
                    y='Aplatissement',
                    title="📈 Asymétrie vs Aplatissement",
                    hover_data=['Image'],
                    color_discrete_sequence=['#d62728']
                )
                # Ajouter des lignes de référence
                fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Normale")
                fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Symétrique")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Test_Normalite_Shapiro' in results_df.columns:
                fig = px.histogram(
                    results_df,
                    x='Test_Normalite_Shapiro',
                    title="🧪 Distribution des p-values (Test Shapiro)",
                    color_discrete_sequence=['#1f77b4'],
                    marginal="box"
                )
                fig.add_vline(x=0.05, line_dash="dash", line_color="red", annotation_text="α=0.05")
                st.plotly_chart(fig, use_container_width=True)
        
        # Distribution des percentiles
        if 'Percentile_5' in results_df.columns and 'Percentile_95' in results_df.columns:
            st.markdown("#### 📏 **Analyse des Percentiles**")
            fig = go.Figure()
            
            for i, row in results_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[5, 95],
                    y=[row['Percentile_5'], row['Percentile_95']],
                    mode='lines+markers',
                    name=row['Image'][:15] + "..." if len(row['Image']) > 15 else row['Image'],
                    opacity=0.7
                ))
            
            fig.update_layout(
                title="Étendue des Intensités (P5-P95)",
                xaxis_title="Percentile",
                yaxis_title="Intensité",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)


def display_medical_results(results_df):
    """Affiche les résultats d'analyse médicale avec visualisations spécialisées."""
    st.markdown("### 🏥 **Métriques Médicales Spécialisées**")
    
    medical_cols = [col for col in results_df.columns if any(keyword in col for keyword in 
                   ['Densite', 'Regions', 'Symetrie', 'Texture_Anormale', 'Homogeneite_Regionale'])]
    
    if medical_cols:
        # Métriques cliniques clés
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'Densite_Moyenne' in results_df.columns:
                avg_density = results_df['Densite_Moyenne'].mean()
                st.metric("🫁 Densité Moyenne", f"{avg_density:.3f}")
        with col2:
            if 'Symetrie_Horizontale' in results_df.columns:
                avg_h_symmetry = results_df['Symetrie_Horizontale'].mean()
                st.metric("↔️ Symétrie H", f"{avg_h_symmetry:.3f}")
        with col3:
            if 'Score_Texture_Anormale' in results_df.columns:
                avg_abnormal = results_df['Score_Texture_Anormale'].mean()
                st.metric("⚠️ Texture Anormale", f"{avg_abnormal:.3f}")
        with col4:
            if 'Homogeneite_Regionale' in results_df.columns:
                avg_regional = results_df['Homogeneite_Regionale'].mean()
                st.metric("🎯 Homogénéité Rég.", f"{avg_regional:.3f}")
        
        # Tableau médical détaillé
        with st.expander("🏥 Tableau détaillé des métriques médicales"):
            st.dataframe(results_df[['Image'] + medical_cols], use_container_width=True)
        
        # Visualisations médicales spécialisées
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique en secteurs pour les régions
            if all(col in results_df.columns for col in ['Regions_Sombres_Pct', 'Regions_Claires_Pct', 'Regions_Moyennes_Pct']):
                avg_dark = results_df['Regions_Sombres_Pct'].mean()
                avg_bright = results_df['Regions_Claires_Pct'].mean()
                avg_medium = results_df['Regions_Moyennes_Pct'].mean()
                
                fig = px.pie(
                    values=[avg_dark, avg_medium, avg_bright],
                    names=['Régions Sombres', 'Régions Moyennes', 'Régions Claires'],
                    title="🫁 Répartition Moyenne des Régions Pulmonaires",
                    color_discrete_sequence=['#2E4057', '#8B8B8D', '#F5F5F5']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Analyse de symétrie
            if 'Symetrie_Horizontale' in results_df.columns and 'Symetrie_Verticale' in results_df.columns:
                fig = px.scatter(
                    results_df,
                    x='Symetrie_Horizontale',
                    y='Symetrie_Verticale',
                    title="↔️ Analyse de Symétrie Thoracique",
                    hover_data=['Image'],
                    color_discrete_sequence=['#e74c3c']
                )
                # Ligne de symétrie parfaite
                fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="gray", dash="dash"))
                st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart médical
        if len(medical_cols) >= 4:
            st.markdown("#### 🕷️ **Profil Médical Radar**")
            radar_medical = ['Densite_Moyenne', 'Symetrie_Horizontale', 'Score_Texture_Anormale', 'Homogeneite_Regionale']
            available_medical = [col for col in radar_medical if col in results_df.columns]
            
            if len(available_medical) >= 3:
                fig = go.Figure()
                
                # Normaliser pour le radar
                radar_data = {}
                for metric in available_medical:
                    if results_df[metric].max() != results_df[metric].min():
                        if 'Anormale' in metric:  # Inverser pour les métriques "mauvaises"
                            radar_data[metric] = 1 - (results_df[metric].mean() - results_df[metric].min()) / (results_df[metric].max() - results_df[metric].min())
                        else:
                            radar_data[metric] = (results_df[metric].mean() - results_df[metric].min()) / (results_df[metric].max() - results_df[metric].min())
                    else:
                        radar_data[metric] = 0.5
                
                fig.add_trace(go.Scatterpolar(
                    r=list(radar_data.values()),
                    theta=[m.replace('_', ' ') for m in available_medical],
                    fill='toself',
                    name='Profil Médical Moyen'
                ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Profil de Santé Radiologique"
                )
                st.plotly_chart(fig, use_container_width=True)


def display_color_results(results_df):
    """Affiche les résultats d'analyse des couleurs avec visualisations riches."""
    st.markdown("### 🌈 **Analyse des Couleurs**")
    
    color_cols = [col for col in results_df.columns if any(keyword in col for keyword in 
                 ['Dominance', 'Saturation', 'Luminosite', 'Couleur', 'Rouge', 'Vert', 'Bleu'])]
    
    if color_cols:
        # Métriques de couleur
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'Dominance_Rouge' in results_df.columns:
                avg_red = results_df['Dominance_Rouge'].mean()
                st.metric("🔴 Rouge Moyen", f"{avg_red:.1f}")
        with col2:
            if 'Dominance_Vert' in results_df.columns:
                avg_green = results_df['Dominance_Vert'].mean()
                st.metric("🟢 Vert Moyen", f"{avg_green:.1f}")
        with col3:
            if 'Dominance_Bleu' in results_df.columns:
                avg_blue = results_df['Dominance_Bleu'].mean()
                st.metric("🔵 Bleu Moyen", f"{avg_blue:.1f}")
        with col4:
            if 'Saturation_Moyenne' in results_df.columns:
                avg_saturation = results_df['Saturation_Moyenne'].mean()
                st.metric("🎨 Saturation Moy.", f"{avg_saturation:.3f}")
        
        # Tableau détaillé
        with st.expander("🌈 Tableau détaillé des couleurs"):
            st.dataframe(results_df[['Image'] + color_cols], use_container_width=True)
        
        # Visualisations couleur
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique de dominance des couleurs
            if all(col in results_df.columns for col in ['Dominance_Rouge', 'Dominance_Vert', 'Dominance_Bleu']):
                avg_red = results_df['Dominance_Rouge'].mean()
                avg_green = results_df['Dominance_Vert'].mean()
                avg_blue = results_df['Dominance_Bleu'].mean()
                
                fig = go.Figure(data=[
                    go.Bar(name='Rouge', x=['Dominance'], y=[avg_red], marker_color='red'),
                    go.Bar(name='Vert', x=['Dominance'], y=[avg_green], marker_color='green'),
                    go.Bar(name='Bleu', x=['Dominance'], y=[avg_blue], marker_color='blue')
                ])
                fig.update_layout(
                    title="🎨 Dominance Moyenne des Couleurs",
                    yaxis_title="Intensité Moyenne",
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter 3D RGB si disponible
            if all(col in results_df.columns for col in ['Dominance_Rouge', 'Dominance_Vert', 'Dominance_Bleu']):
                fig = px.scatter_3d(
                    results_df,
                    x='Dominance_Rouge',
                    y='Dominance_Vert',
                    z='Dominance_Bleu',
                    title="🌈 Espace Colorimétrique RGB",
                    hover_data=['Image'],
                    opacity=0.7
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Distribution de saturation et luminosité
        if 'Saturation_Moyenne' in results_df.columns and 'Luminosite_Moyenne' in results_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    results_df,
                    x='Saturation_Moyenne',
                    title="🎨 Distribution de la Saturation",
                    color_discrete_sequence=['#ff6b6b'],
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    results_df,
                    x='Saturation_Moyenne',
                    y='Luminosite_Moyenne',
                    title="💡 Saturation vs Luminosité",
                    hover_data=['Image'],
                    color_discrete_sequence=['#4ecdc4']
                )
                st.plotly_chart(fig, use_container_width=True)


def display_frequency_results(results_df):
    """Affiche les résultats d'analyse fréquentielle avec visualisations avancées."""
    st.markdown("### 🔍 **Analyse de Fréquences**")
    
    freq_cols = [col for col in results_df.columns if any(keyword in col for keyword in 
                ['Energie_Spectrale', 'Frequence', 'Periodicite'])]
    
    if freq_cols:
        # Métriques de fréquence
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'Energie_Spectrale' in results_df.columns:
                avg_energy = results_df['Energie_Spectrale'].mean()
                st.metric("⚡ Énergie Spectrale", f"{avg_energy:.2e}")
        with col2:
            if 'Periodicite_Horizontale' in results_df.columns:
                avg_h_period = results_df['Periodicite_Horizontale'].mean()
                st.metric("↔️ Périodicité H", f"{avg_h_period:.3f}")
        with col3:
            if 'Periodicite_Verticale' in results_df.columns:
                avg_v_period = results_df['Periodicite_Verticale'].mean()
                st.metric("↕️ Périodicité V", f"{avg_v_period:.3f}")
        with col4:
            if 'Frequence_Dominante' in results_df.columns:
                # Convertir les tuples en string pour affichage
                freq_str = str(results_df['Frequence_Dominante'].iloc[0]) if len(results_df) > 0 else "N/A"
                st.metric("🎯 Fréq. Dominante", freq_str[:10] + "...")
        
        # Tableau détaillé
        with st.expander("🔍 Tableau détaillé des fréquences"):
            st.dataframe(results_df[['Image'] + freq_cols], use_container_width=True)
        
        # Visualisations fréquentielles
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Energie_Spectrale' in results_df.columns:
                fig = px.histogram(
                    results_df,
                    x='Energie_Spectrale',
                    title="⚡ Distribution de l'Énergie Spectrale",
                    color_discrete_sequence=['#ffa500'],
                    marginal="box"
                )
                fig.update_xaxis(type="log")  # Échelle logarithmique
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Periodicite_Horizontale' in results_df.columns and 'Periodicite_Verticale' in results_df.columns:
                fig = px.scatter(
                    results_df,
                    x='Periodicite_Horizontale',
                    y='Periodicite_Verticale',
                    title="📊 Périodicité Horizontale vs Verticale",
                    hover_data=['Image'],
                    color_discrete_sequence=['#9b59b6']
                )
                # Ligne de référence pour périodicité égale
                max_val = max(results_df['Periodicite_Horizontale'].max(), results_df['Periodicite_Verticale'].max())
                fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="gray", dash="dash"))
                st.plotly_chart(fig, use_container_width=True)
        
        # Analyse spectrale si données de répartition disponibles
        if 'Repartition_Frequences' in results_df.columns:
            st.markdown("#### 📈 **Analyse Spectrale Détaillée**")
            # Cette partie nécessiterait un décodage des données de répartition
            st.info("📊 Les données de répartition spectrale sont disponibles mais nécessitent un décodage spécialisé")


def display_ai_results(results_df):
    """Affiche les résultats d'analyse IA avec visualisations avancées."""
    st.markdown("### 🤖 **Analyse de Complexité IA**")
    
    ai_cols = [col for col in results_df.columns if any(keyword in col for keyword in 
              ['Complexite', 'Dimension_Fractale', 'Entropie_Renyi', 'Predictibilite', 'Visuelle'])]
    
    if ai_cols:
        # Métriques IA
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'Complexite_Kolmogorov' in results_df.columns:
                avg_kolmogorov = results_df['Complexite_Kolmogorov'].mean()
                st.metric("🧠 Kolmogorov", f"{avg_kolmogorov:.4f}")
        with col2:
            if 'Dimension_Fractale' in results_df.columns:
                avg_fractal = results_df['Dimension_Fractale'].mean()
                st.metric("🌿 Fractale", f"{avg_fractal:.3f}")
        with col3:
            if 'Entropie_Renyi' in results_df.columns:
                avg_renyi = results_df['Entropie_Renyi'].mean()
                st.metric("🌀 Entropie Rényi", f"{avg_renyi:.2f}")
        with col4:
            if 'Complexite_Visuelle' in results_df.columns:
                avg_visual = results_df['Complexite_Visuelle'].mean()
                st.metric("👁️ Complexité Visuelle", f"{avg_visual:.3f}")
        
        # Tableau détaillé
        with st.expander("🤖 Tableau détaillé de l'analyse IA"):
            st.dataframe(results_df[['Image'] + ai_cols], use_container_width=True)
        
        # Visualisations IA
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Complexite_Kolmogorov' in results_df.columns and 'Dimension_Fractale' in results_df.columns:
                fig = px.scatter(
                    results_df,
                    x='Complexite_Kolmogorov',
                    y='Dimension_Fractale',
                    title="🧠 Complexité Kolmogorov vs Dimension Fractale",
                    hover_data=['Image'],
                    color_discrete_sequence=['#e67e22']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Predictibilite' in results_df.columns and 'Complexite_Visuelle' in results_df.columns:
                fig = px.scatter(
                    results_df,
                    x='Predictibilite',
                    y='Complexite_Visuelle',
                    title="🔮 Prédictibilité vs Complexité Visuelle",
                    hover_data=['Image'],
                    color_discrete_sequence=['#3498db']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart IA
        if len(ai_cols) >= 4:
            st.markdown("#### 🕷️ **Profil IA Multi-Dimensionnel**")
            ai_radar_metrics = ['Complexite_Kolmogorov', 'Dimension_Fractale', 'Entropie_Renyi', 'Complexite_Visuelle']
            available_ai = [col for col in ai_radar_metrics if col in results_df.columns]
            
            if len(available_ai) >= 3:
                fig = go.Figure()
                
                # Normaliser pour le radar
                radar_data = {}
                for metric in available_ai:
                    if results_df[metric].max() != results_df[metric].min():
                        radar_data[metric] = (results_df[metric].mean() - results_df[metric].min()) / (results_df[metric].max() - results_df[metric].min())
                    else:
                        radar_data[metric] = 0.5
                
                fig.add_trace(go.Scatterpolar(
                    r=list(radar_data.values()),
                    theta=[m.replace('_', ' ').replace('Complexite', 'Cpx') for m in available_ai],
                    fill='toself',
                    name='Profil IA Moyen'
                ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Profil de Complexité IA"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Histogramme de complexité globale
        if 'Complexite_Visuelle' in results_df.columns:
            st.markdown("#### 📊 **Distribution de la Complexité Visuelle**")
            fig = px.histogram(
                results_df,
                x='Complexite_Visuelle',
                title="Distribution de la Complexité Visuelle",
                color_discrete_sequence=['#9b59b6'],
                marginal="violin"
            )
            # Ajouter des seuils de complexité
            fig.add_vline(x=0.33, line_dash="dash", line_color="green", annotation_text="Faible")
            fig.add_vline(x=0.66, line_dash="dash", line_color="orange", annotation_text="Élevée")
            st.plotly_chart(fig, use_container_width=True)


def display_export_options(results_df, analysis_options, category):
    """Affiche les options d'export."""
    st.markdown("---")
    st.markdown("### 💾 **Options d'Export**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Exporter CSV Complet"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name=f"analyse_avancee_{category}_{int(time.time())}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📄 Générer Rapport JSON"):
            report = {
                'metadata': {
                    'category': category,
                    'analysis_types': analysis_options,
                    'timestamp': time.time(),
                    'num_images': len(results_df)
                },
                'results': results_df.to_dict('records'),
                'summary': {
                    'avg_quality': results_df.get('Score_Qualite_Global', pd.Series([0])).mean(),
                    'avg_complexity': results_df.get('Complexite_Visuelle', pd.Series([0])).mean()
                }
            }
            json_str = json.dumps(report, indent=2, default=str)
            st.download_button(
                label="Télécharger JSON",
                data=json_str,
                file_name=f"rapport_avance_{category}_{int(time.time())}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("📈 Exporter Statistiques"):
            stats = results_df.describe()
            stats_csv = stats.to_csv()
            st.download_button(
                label="Télécharger Stats",
                data=stats_csv,
                file_name=f"statistiques_{category}_{int(time.time())}.csv",
                mime="text/csv"
            )


def show_advanced_image_metrics(data_dir, categories):
    """Affiche des métriques avancées d'analyse d'images avec de nombreuses améliorations."""
    st.markdown("## 🔬 **Métriques Avancées d'Image - Version Complète**")
    
    # Interface améliorée
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 **Configuration de l'Analyse**")
        
        # Option pour analyser toutes les catégories ou une seule
        analysis_scope = st.radio(
            "📂 Portée de l'analyse:",
            ["🎯 Une catégorie spécifique", "🌍 Toutes les catégories (comparatif)"],
            key="analysis_scope"
        )
        
        if analysis_scope == "🎯 Une catégorie spécifique":
            # Sélection d'une catégorie
            selected_categories = [st.selectbox(
                "📂 Sélectionner une catégorie pour l'analyse détaillée:",
                categories,
                key="advanced_metrics_category"
            )]
        else:
            # Sélection multiple de catégories
            selected_categories = st.multiselect(
                "📂 Sélectionner les catégories à comparer:",
                categories,
                default=categories,
                key="advanced_metrics_multi_category"
            )
            
            if not selected_categories:
                st.warning("⚠️ Veuillez sélectionner au moins une catégorie.")
                return
    
    with col2:
        st.markdown("### ⚙️ **Paramètres**")
        max_images_per_category = st.slider("📷 Images par catégorie:", 5, 100, 20, key="advanced_metrics_sample_size")
        show_progress_details = st.checkbox("📊 Afficher les détails de progression", value=True)
    
    # Types d'analyses disponibles
    st.markdown("### 🔍 **Types d'Analyses Disponibles**")
    
    analysis_options = st.multiselect(
        "Sélectionnez les analyses à effectuer:",
        [
            "🏆 Qualité d'image complète",
            "🎨 Analyse de texture avancée", 
            "🔲 Détection de contours et formes",
            "📊 Analyse statistique approfondie",
            "🌈 Analyse des couleurs",
            "🔍 Analyse de fréquences",
            "🏥 Métriques médicales",
            "🤖 Score de complexité IA"
        ],
        default=["🏆 Qualité d'image complète", "🎨 Analyse de texture avancée"],
        key="advanced_analysis_types"
    )
    
    if not analysis_options:
        st.warning("⚠️ Veuillez sélectionner au moins un type d'analyse.")
        return
    
    # Options avancées
    with st.expander("🔧 **Options Avancées**"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            save_intermediate = st.checkbox("💾 Sauvegarder résultats intermédiaires", value=False)
            compare_with_reference = st.checkbox("📏 Comparer avec image de référence", value=False)
            
        with col2:
            export_visualizations = st.checkbox("📈 Exporter visualisations", value=False)
            generate_report = st.checkbox("📄 Générer rapport PDF", value=False)
            
        with col3:
            batch_processing = st.checkbox("⚡ Traitement par lots optimisé", value=True)
            parallel_computing = st.checkbox("🔄 Calcul parallèle", value=False)
    
    # Image de référence si sélectionnée
    reference_img_array = None
    if compare_with_reference and len(selected_categories) == 1:
        st.markdown("### 📌 **Image de Référence**")
        ref_images_dir = Path(data_dir) / selected_categories[0] / "images"
        if ref_images_dir.exists():
            ref_images = list(ref_images_dir.glob("*.png"))[:10]  # Limiter le choix
            if ref_images:
                ref_image_name = st.selectbox(
                    "Choisir l'image de référence:",
                    [img.name for img in ref_images],
                    key="reference_image"
                )
                ref_image_path = ref_images_dir / ref_image_name
                try:
                    ref_img = Image.open(ref_image_path).convert('L')
                    reference_img_array = np.array(ref_img)
                    st.image(ref_img, caption=f"Image de référence: {ref_image_name}", width=200)
                except Exception as e:
                    st.error(f"Erreur lors du chargement de l'image de référence: {e}")
    
    # Estimation du temps de traitement
    total_images = sum([len(list((Path(data_dir) / cat / "images").glob("*.png"))) for cat in selected_categories if (Path(data_dir) / cat / "images").exists()])
    estimated_time = (min(total_images, len(selected_categories) * max_images_per_category) * len(analysis_options) * 0.1)
    
    if estimated_time > 60:
        st.warning(f"⏱️ Temps estimé: {estimated_time/60:.1f} minutes. Considérez réduire le nombre d'images ou d'analyses.")
    
    # Bouton de lancement amélioré
    if st.button("🚀 **Lancer l'Analyse Avancée Complète**", key="launch_comprehensive_analysis", type="primary"):
        # Dictionnaire pour stocker tous les résultats par catégorie
        all_category_results = {}
        all_processing_times = {}
        
        # Barre de progression globale
        total_categories = len(selected_categories)
        global_progress = st.progress(0)
        global_status = st.empty()
        
        start_time = time.time()
        
        # Analyser chaque catégorie
        for cat_idx, category in enumerate(selected_categories):
            global_status.text(f"🔍 Analyse de la catégorie: {category} ({cat_idx+1}/{total_categories})")
            
            images_dir = Path(data_dir) / category / "images"
            
            if not images_dir.exists():
                st.error(f"📁 Dossier d'images non trouvé: {images_dir}")
                continue
            
            image_files = list(images_dir.glob("*.png"))[:max_images_per_category]
            
            if not image_files:
                st.warning(f"⚠️ Aucune image trouvée dans la catégorie {category}.")
                continue
            
            # Interface de progression pour cette catégorie
            if len(selected_categories) == 1:  # Affichage détaillé pour une seule catégorie
                st.markdown("---")
                st.markdown(f"### 🔄 **Progression de l'Analyse - {category}**")
                
                # Métriques en temps réel
                if show_progress_details:
                    metrics_cols = st.columns(4)
                    with metrics_cols[0]:
                        progress_metric = st.empty()
                    with metrics_cols[1]:
                        speed_metric = st.empty()
                    with metrics_cols[2]:
                        eta_metric = st.empty()
                    with metrics_cols[3]:
                        quality_metric = st.empty()
                
                # Barres de progression
                main_progress = st.progress(0)
                status_text = st.empty()
            
            # Conteneurs pour résultats de cette catégorie
            category_results = []
            category_processing_times = {}
            
            category_start_time = time.time()
            
            # Analyser chaque image de cette catégorie
            for i, image_path in enumerate(image_files):
                img_start_time = time.time()
                
                try:
                    if len(selected_categories) == 1 and show_progress_details:
                        status_text.text(f"🔍 Analyse en cours: {image_path.name} ({i+1}/{len(image_files)})")
                    
                    img = Image.open(image_path)
                    img_array_color = np.array(img)
                    img_array = np.array(img.convert('L'))
                    
                    # Dictionnaire pour stocker tous les résultats de cette image
                    image_results = {
                        "Image": image_path.name,
                        "Categorie": category,
                        "Taille_Fichier_KB": image_path.stat().st_size / 1024,
                        "Dimensions": f"{img.size[0]}x{img.size[1]}",
                        "Pixels_Total": img.size[0] * img.size[1]
                    }
                    
                    # Exécuter les analyses sélectionnées
                    for analysis_type in analysis_options:
                        analysis_start = time.time()
                        
                        try:
                            if "🏆 Qualité d'image complète" in analysis_type:
                                # Gérer le cas où reference_img_array n'est pas défini
                                ref_img = reference_img_array if 'reference_img_array' in locals() else None
                                quality_results = perform_comprehensive_quality_analysis(img_array, ref_img)
                                image_results.update(quality_results)
                            
                            if "🎨 Analyse de texture avancée" in analysis_type:
                                texture_results = perform_advanced_texture_analysis(img_array)
                                image_results.update(texture_results)
                            
                            if "🔲 Détection de contours et formes" in analysis_type:
                                edge_results = perform_comprehensive_edge_analysis(img_array)
                                image_results.update(edge_results)
                            
                            if "📊 Analyse statistique approfondie" in analysis_type:
                                stats_results = perform_deep_statistical_analysis(img_array)
                                image_results.update(stats_results)
                            
                            if "🌈 Analyse des couleurs" in analysis_type and len(img_array_color.shape) == 3:
                                color_results = perform_color_analysis(img_array_color)
                                image_results.update(color_results)
                            
                            if "🔍 Analyse de fréquences" in analysis_type:
                                freq_results = perform_frequency_analysis(img_array)
                                image_results.update(freq_results)
                            
                            if "🏥 Métriques médicales" in analysis_type:
                                medical_results = perform_medical_metrics_analysis(img_array)
                                image_results.update(medical_results)
                            
                            if "🤖 Score de complexité IA" in analysis_type:
                                ai_results = perform_ai_complexity_analysis(img_array)
                                image_results.update(ai_results)
                            
                            category_processing_times[f"{analysis_type}_{i}"] = time.time() - analysis_start
                            
                        except Exception as analysis_error:
                            st.warning(f"⚠️ Erreur lors de l'analyse '{analysis_type}' pour {image_path.name}: {str(analysis_error)}")
                            # Continuer avec les autres types d'analyse
                            continue
                    
                    # Stocker les résultats de cette image
                    category_results.append(image_results)
                    
                    # Mise à jour des métriques en temps réel (uniquement pour une catégorie)
                    if len(selected_categories) == 1 and show_progress_details:
                        elapsed = time.time() - category_start_time
                        avg_speed = (i + 1) / elapsed if elapsed > 0 else 0
                        eta = (len(image_files) - i - 1) / avg_speed if avg_speed > 0 else 0
                        
                        progress_metric.metric("📷 Images", f"{i+1}/{len(image_files)}")
                        speed_metric.metric("⚡ Vitesse", f"{avg_speed:.1f} img/s")
                        eta_metric.metric("⏳ ETA", f"{eta:.1f}s")
                        
                        # Score de qualité moyen
                        if 'Score_Qualite_Global' in image_results:
                            quality_metric.metric("🏆 Qualité Moy.", f"{image_results['Score_Qualite_Global']:.2f}")
                    
                    img_processing_time = time.time() - img_start_time
                    
                except Exception as e:
                    st.error(f"❌ Erreur critique avec {image_path.name}: {str(e)}")
                    # Continuer avec l'image suivante
                    continue
                
                # Mise à jour de la progression principale
                if len(selected_categories) == 1:
                    main_progress.progress((i + 1) / len(image_files))
            
            # Stocker les résultats de cette catégorie
            all_category_results[category] = category_results
            all_processing_times[category] = category_processing_times
            
            # Nettoyage des barres de progression pour une seule catégorie
            if len(selected_categories) == 1:
                main_progress.empty()
                status_text.empty()
            
            # Mise à jour de la progression globale
            global_progress.progress((cat_idx + 1) / total_categories)
        
        # Nettoyage des barres de progression globales
        global_progress.empty()
        global_status.empty()
        
        total_processing_time = time.time() - start_time
        
        # Affichage des résultats
        if all_category_results:
            st.success(f"✅ **Analyse terminée en {total_processing_time:.2f}s** pour {len(selected_categories)} catégorie(s)")
            
            if analysis_scope == "🎯 Une catégorie spécifique":
                # Affichage pour une seule catégorie
                category = selected_categories[0]
                if category in all_category_results and all_category_results[category]:
                    results_df = pd.DataFrame(all_category_results[category])
                    
                    # Affichage des résultats avec tabs
                    display_comprehensive_results(results_df, analysis_options, category, all_processing_times[category])
                    
                    # Options d'export
                    if save_intermediate or export_visualizations or generate_report:
                        display_export_options(results_df, analysis_options, category)
            else:
                # Affichage comparatif pour plusieurs catégories
                st.markdown("---")
                st.markdown("## 🌍 **Analyse Comparative Multi-Catégories**")
                
                # Créer un DataFrame combiné
                all_results_list = []
                for category, results in all_category_results.items():
                    if results:
                        all_results_list.extend(results)
                
                if all_results_list:
                    combined_df = pd.DataFrame(all_results_list)
                    
                    # Affichage du résumé comparatif
                    display_comparative_summary(combined_df, selected_categories)
                    
                    # Affichage par catégorie dans des tabs
                    category_tabs = st.tabs([f"📂 {cat}" for cat in selected_categories])
                    
                    for i, category in enumerate(selected_categories):
                        with category_tabs[i]:
                            if category in all_category_results and all_category_results[category]:
                                category_df = pd.DataFrame(all_category_results[category])
                                display_comprehensive_results(category_df, analysis_options, category, all_processing_times[category])
                    
                    # Analyse comparative globale
                    st.markdown("---")
                    display_global_comparative_analysis(combined_df, analysis_options)
                
        else:
            st.error("❌ Aucun résultat n'a pu être généré.")


def perform_comprehensive_quality_analysis(img_array, reference_img=None):
    """Effectue une analyse complète de la qualité d'image."""
    results = {}
    
    try:
        # Métriques de base améliorées
        results['Nettete_Laplacian'] = calculate_sharpness(img_array)
        results['Nettete_Sobel'] = calculate_sobel_sharpness(img_array)
        results['Nettete_Variance'] = calculate_variance_sharpness(img_array)
        results['Niveau_Bruit'] = calculate_noise_level(img_array)
        results['Contraste_RMS'] = calculate_rms_contrast(img_array)
        results['Contraste_Michelson'] = calculate_michelson_contrast(img_array)
        
        # Métriques de distorsion
        results['Blur_Detection'] = detect_blur_level(img_array)
        results['Artefacts_Score'] = detect_artifacts(img_array)
        
        # Score de qualité composite avec gestion des valeurs nulles
        sharpness_score = np.mean([
            results.get('Nettete_Laplacian', 0),
            results.get('Nettete_Sobel', 0),
            results.get('Nettete_Variance', 0)
        ])
        
        noise_level = results.get('Niveau_Bruit', 1)
        noise_penalty = 1 / (1 + max(noise_level, 0.001))  # Éviter division par zéro
        
        contrast_score = np.mean([
            results.get('Contraste_RMS', 0),
            results.get('Contraste_Michelson', 0)
        ])
        
        # Normalisation du score final
        quality_raw = sharpness_score * noise_penalty * contrast_score
        results['Score_Qualite_Global'] = min(max(quality_raw / 1000, 0), 1)  # Normaliser entre 0 et 1
        
        # Comparaison avec référence si disponible
        if reference_img is not None:
            try:
                results['SSIM_Reference'] = calculate_ssim(img_array, reference_img)
                results['MSE_Reference'] = calculate_mse(img_array, reference_img)
                results['PSNR_Reference'] = calculate_psnr(img_array, reference_img)
            except Exception as e:
                # En cas d'erreur dans la comparaison, on met des valeurs par défaut
                results['SSIM_Reference'] = 0.0
                results['MSE_Reference'] = 0.0
                results['PSNR_Reference'] = 0.0
        
    except Exception as e:
        # En cas d'erreur générale, on retourne des valeurs par défaut
        for key in ['Nettete_Laplacian', 'Nettete_Sobel', 'Nettete_Variance', 'Niveau_Bruit',
                   'Contraste_RMS', 'Contraste_Michelson', 'Blur_Detection', 'Artefacts_Score']:
            results[key] = 0.0
        results['Score_Qualite_Global'] = 0.0
    
    return results


def perform_advanced_texture_analysis(img_array):
    """Effectue une analyse avancée de texture."""
    results = {}
    
    try:
        # Métriques de texture classiques
        results['Entropie'] = calculate_entropy(img_array)
        results['Homogeneite'] = calculate_homogeneity(img_array)
        results['Energie'] = calculate_energy(img_array)
        
        # Métriques GLCM (Gray Level Co-occurrence Matrix) avec gestion d'erreurs
        try:
            results['Contraste_GLCM'] = calculate_glcm_contrast(img_array)
        except Exception:
            results['Contraste_GLCM'] = 0.0
            
        try:
            results['Dissimilarite_GLCM'] = calculate_glcm_dissimilarity(img_array)
        except Exception:
            results['Dissimilarite_GLCM'] = 0.0
            
        try:
            results['Correlation_GLCM'] = calculate_glcm_correlation(img_array)
        except Exception:
            results['Correlation_GLCM'] = 0.0
        
        # Métriques de texture locale
        try:
            results['Variance_Locale'] = calculate_local_variance(img_array)
        except Exception:
            results['Variance_Locale'] = 0.0
            
        try:
            results['Ecart_Type_Local'] = calculate_local_std(img_array)
        except Exception:
            results['Ecart_Type_Local'] = 0.0
        
        # Analyse de motifs
        try:
            results['Repetitivite_Texture'] = calculate_texture_repetitiveness(img_array)
        except Exception:
            results['Repetitivite_Texture'] = 0.0
            
        try:
            results['Regularite_Texture'] = calculate_texture_regularity(img_array)
        except Exception:
            results['Regularite_Texture'] = 0.0
        
    except Exception as e:
        # En cas d'erreur générale, on retourne des valeurs par défaut
        for key in ['Entropie', 'Homogeneite', 'Energie', 'Contraste_GLCM', 'Dissimilarite_GLCM',
                   'Correlation_GLCM', 'Variance_Locale', 'Ecart_Type_Local', 'Repetitivite_Texture',
                   'Regularite_Texture']:
            results[key] = 0.0
    
    return results


def perform_comprehensive_edge_analysis(img_array):
    """Effectue une analyse complète des contours et formes."""
    results = {}
    
    # Détection de contours avec différents opérateurs
    results['Contours_Sobel'] = calculate_edge_density(img_array)
    results['Contours_Canny'] = calculate_canny_edges(img_array)
    results['Contours_Prewitt'] = calculate_prewitt_edges(img_array)
    
    # Force et orientation des contours
    results['Force_Contours'] = calculate_edge_strength(img_array)
    results['Orientation_Dominante'] = calculate_dominant_orientation(img_array)
    results['Coherence_Orientation'] = calculate_orientation_coherence(img_array)
    
    # Analyse de formes
    results['Circularite'] = calculate_circularity(img_array)
    results['Rectangularite'] = calculate_rectangularity(img_array)
    results['Complexite_Forme'] = calculate_shape_complexity(img_array)
    
    return results


def perform_deep_statistical_analysis(img_array):
    """Effectue une analyse statistique approfondie."""
    results = {}
    
    # Statistiques de base étendues
    results['Moyenne'] = np.mean(img_array)
    results['Mediane'] = np.median(img_array)
    results['Ecart_Type'] = np.std(img_array)
    results['Asymetrie'] = stats.skew(img_array.flatten())
    results['Aplatissement'] = stats.kurtosis(img_array.flatten())
    
    # Statistiques de distribution
    results['Percentile_5'] = np.percentile(img_array, 5)
    results['Percentile_95'] = np.percentile(img_array, 95)
    results['Etendue_Dynamique'] = np.max(img_array) - np.min(img_array)
    results['Coefficient_Variation'] = results['Ecart_Type'] / results['Moyenne'] if results['Moyenne'] > 0 else 0
    
    # Tests de normalité
    results['Test_Normalite_Shapiro'] = perform_shapiro_test(img_array.flatten())
    results['Test_Normalite_KS'] = perform_ks_test(img_array.flatten())
    
    # Analyse de distribution locale
    results['Entropie_Locale'] = calculate_local_entropy(img_array)
    results['Uniformite_Distribution'] = calculate_distribution_uniformity(img_array)
    
    return results


def perform_color_analysis(img_array_color):
    """Effectue une analyse des couleurs (si image couleur)."""
    results = {}
    
    # Séparation des canaux
    if len(img_array_color.shape) == 3:
        r, g, b = img_array_color[:,:,0], img_array_color[:,:,1], img_array_color[:,:,2]
        
        # Dominance des couleurs
        results['Dominance_Rouge'] = np.mean(r)
        results['Dominance_Vert'] = np.mean(g)
        results['Dominance_Bleu'] = np.mean(b)
        
        # Saturation et luminosité
        results['Saturation_Moyenne'] = calculate_average_saturation(img_array_color)
        results['Luminosite_Moyenne'] = calculate_average_brightness(img_array_color)
        
        # Diversité des couleurs
        results['Nombre_Couleurs_Uniques'] = len(np.unique(img_array_color.reshape(-1, 3), axis=0))
        results['Entropie_Couleur'] = calculate_color_entropy(img_array_color)
        
        # Contraste colorimétrique
        results['Contraste_Couleur'] = calculate_color_contrast(img_array_color)
    
    return results


def perform_frequency_analysis(img_array):
    """Effectue une analyse dans le domaine fréquentiel."""
    results = {}
    
    # Transformée de Fourier
    fft = np.fft.fft2(img_array)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.log(np.abs(fft_shift) + 1)
    
    # Analyse spectrale
    results['Energie_Spectrale'] = np.sum(np.abs(fft)**2)
    results['Frequence_Dominante'] = find_dominant_frequency(fft)
    results['Repartition_Frequences'] = calculate_frequency_distribution(magnitude_spectrum)
    
    # Analyse de périodicité
    results['Periodicite_Horizontale'] = detect_horizontal_periodicity(img_array)
    results['Periodicite_Verticale'] = detect_vertical_periodicity(img_array)
    
    return results


def perform_medical_metrics_analysis(img_array):
    """Effectue une analyse spécialisée pour l'imagerie médicale."""
    results = {}
    
    # Métriques de densité (simulées pour radiographie)
    results['Densite_Moyenne'] = np.mean(img_array) / 255.0
    results['Densite_Ecart_Type'] = np.std(img_array) / 255.0
    
    # Analyse de régions
    results['Regions_Sombres_Pct'] = np.sum(img_array < 85) / img_array.size * 100
    results['Regions_Claires_Pct'] = np.sum(img_array > 170) / img_array.size * 100
    results['Regions_Moyennes_Pct'] = 100 - results['Regions_Sombres_Pct'] - results['Regions_Claires_Pct']
    
    # Symétrie (important pour radiographie thoracique)
    results['Symetrie_Horizontale'] = calculate_horizontal_symmetry(img_array)
    results['Symetrie_Verticale'] = calculate_vertical_symmetry(img_array)
    
    # Détection de patterns pathologiques (simplifié)
    results['Score_Texture_Anormale'] = detect_abnormal_texture_patterns(img_array)
    results['Homogeneite_Regionale'] = calculate_regional_homogeneity(img_array)
    
    return results


def perform_ai_complexity_analysis(img_array):
    """Effectue une analyse de complexité basée sur des métriques IA."""
    results = {}
    
    # Complexité de Kolmogorov (approximée)
    results['Complexite_Kolmogorov'] = approximate_kolmogorov_complexity(img_array)
    
    # Dimension fractale
    results['Dimension_Fractale'] = calculate_fractal_dimension(img_array)
    
    # Entropie de Rényi
    results['Entropie_Renyi'] = calculate_renyi_entropy(img_array)
    
    # Score de prédictibilité
    results['Predictibilite'] = calculate_predictability_score(img_array)
    
    # Complexité visuelle
    results['Complexite_Visuelle'] = calculate_visual_complexity(img_array)
    
    return results


def display_global_comparison_enhanced(all_results: Dict, selected_categories: List[str]):
    """Affiche une comparaison globale améliorée avec de nombreux graphiques."""
    st.markdown("### 🔄 **Analyse Comparative Complète**")
    
    # Créer un DataFrame de comparaison avec toutes les métriques
    comparison_data = []
    detailed_data = []
    
    for category, data in all_results.items():
        results = data['results']
        count = data['count']
        
        if results['sizes']:  # S'assurer qu'il y a des données
            # Données de résumé pour comparaison
            comparison_data.append({
                'Catégorie': category,
                'Nombre_Images': count,
                'Largeur_Moyenne': np.mean([size[0] for size in results['sizes']]),
                'Hauteur_Moyenne': np.mean([size[1] for size in results['sizes']]),
                'Intensité_Moyenne': np.mean(results['means']),
                'Écart_Type_Intensité': np.std(results['means']),
                'Taille_Fichier_Moyenne_KB': np.mean(results['file_sizes']) / 1024,
                'Entropie_Moyenne': np.mean(results.get('entropy', [0])),
                'Netteté_Moyenne': np.mean(results.get('sharpness', [0])),
                'Contraste_Moyen': np.mean(results.get('contrast_ratio', [0])),
                'Densité_Contours': np.mean(results.get('edge_density', [0])),
                'Luminosité_Moyenne': np.mean(results.get('brightness', [0]))
            })
            
            # Données détaillées pour analyses statistiques
            for i in range(len(results['sizes'])):
                detailed_data.append({
                    'Catégorie': category,
                    'Largeur': results['sizes'][i][0],
                    'Hauteur': results['sizes'][i][1],
                    'Intensité': results['means'][i],
                    'Écart_Type': results['stds'][i],
                    'Taille_Fichier_KB': results['file_sizes'][i] / 1024,
                    'Entropie': results.get('entropy', [0])[i] if i < len(results.get('entropy', [])) else 0,
                    'Netteté': results.get('sharpness', [0])[i] if i < len(results.get('sharpness', [])) else 0,
                    'Contraste': results.get('contrast_ratio', [0])[i] if i < len(results.get('contrast_ratio', [])) else 0
                })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        detailed_df = pd.DataFrame(detailed_data)
        
        # 1. Tableau de comparaison principal
        st.markdown("#### 📊 **Résumé Comparatif**")
        st.dataframe(comparison_df, use_container_width=True)
        
        # 2. Métriques principales en colonnes
        st.markdown("#### 🎯 **Métriques Clés**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_images = comparison_df['Nombre_Images'].sum()
            st.metric("📷 Total Images", f"{total_images:,}")
            
        with col2:
            avg_resolution = comparison_df['Largeur_Moyenne'].mean() * comparison_df['Hauteur_Moyenne'].mean()
            st.metric("📐 Résolution Moy.", f"{avg_resolution/1000:.1f}K px")
            
        with col3:
            avg_file_size = comparison_df['Taille_Fichier_Moyenne_KB'].mean()
            st.metric("💾 Taille Moy.", f"{avg_file_size:.1f} KB")
            
        with col4:
            category_with_best_quality = comparison_df.loc[comparison_df['Netteté_Moyenne'].idxmax(), 'Catégorie']
            st.metric("🏆 Meilleure Netteté", category_with_best_quality)
        
        # 3. Graphiques de comparaison avancés
        st.markdown("#### 📈 **Visualisations Comparatives**")
        
        # Graphique radar multi-métriques
        fig_radar = create_radar_chart(comparison_df)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Graphiques en sous-plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot des intensités par catégorie
            fig_box = px.box(
                detailed_df,
                x='Catégorie',
                y='Intensité',
                title="📦 Distribution des Intensités par Catégorie",
                color='Catégorie'
            )
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Violin plot des tailles de fichiers
            fig_violin = px.violin(
                detailed_df,
                x='Catégorie',
                y='Taille_Fichier_KB',
                title="🎻 Distribution des Tailles de Fichier",
                color='Catégorie',
                box=True
            )
            fig_violin.update_layout(showlegend=False)
            st.plotly_chart(fig_violin, use_container_width=True)
        
        # 4. Heatmap des corrélations
        st.markdown("#### 🔥 **Heatmap des Corrélations**")
        numeric_cols = ['Intensité', 'Écart_Type', 'Taille_Fichier_KB', 'Entropie', 'Netteté', 'Contraste']
        available_cols = [col for col in numeric_cols if col in detailed_df.columns]
        
        if len(available_cols) > 1:
            corr_matrix = detailed_df[available_cols].corr()
            fig_heatmap = px.imshow(
                corr_matrix,
                title="Matrice de Corrélation des Métriques",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # 5. Analyses statistiques
        st.markdown("#### 📊 **Analyses Statistiques**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Test ANOVA pour les intensités
            st.markdown("**🧪 Test ANOVA - Intensités**")
            groups = [detailed_df[detailed_df['Catégorie'] == cat]['Intensité'].values 
                     for cat in selected_categories]
            
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                st.write(f"**F-statistique:** {f_stat:.3f}")
                st.write(f"**p-value:** {p_value:.3e}")
                
                if p_value < 0.05:
                    st.success("✅ Différences significatives entre catégories")
                else:
                    st.info("ℹ️ Pas de différence significative")
            except:
                st.warning("⚠️ Impossible de calculer l'ANOVA")
        
        with col2:
            # Statistiques descriptives par catégorie
            st.markdown("**📈 Statistiques par Catégorie**")
            intensity_stats = detailed_df.groupby('Catégorie')['Intensité'].agg(['mean', 'std', 'min', 'max'])
            st.dataframe(intensity_stats, use_container_width=True)
        
        # 6. Graphiques 3D si disponibles
        if len(available_cols) >= 3:
            st.markdown("#### 🌐 **Visualisation 3D**")
            fig_3d = px.scatter_3d(
                detailed_df,
                x=available_cols[0],
                y=available_cols[1], 
                z=available_cols[2],
                color='Catégorie',
                title="Analyse 3D Multi-Métriques",
                opacity=0.7
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        
        # 7. Insights automatiques
        st.markdown("#### 🧠 **Insights Automatiques**")
        generate_insights(comparison_df, detailed_df)
        
        return comparison_df, detailed_df
    
    else:
        st.warning("⚠️ Aucune donnée valide trouvée pour la comparaison")
        return None, None


def create_radar_chart(comparison_df):
    """Crée un graphique radar pour la comparaison multi-métriques."""
    # Normaliser les données pour le radar (0-1)
    radar_data = comparison_df.copy()
    metrics = ['Intensité_Moyenne', 'Netteté_Moyenne', 'Contraste_Moyen', 'Entropie_Moyenne', 'Luminosité_Moyenne']
    available_metrics = [m for m in metrics if m in radar_data.columns]
    
    for metric in available_metrics:
        if radar_data[metric].max() != radar_data[metric].min():
            radar_data[metric + '_norm'] = (radar_data[metric] - radar_data[metric].min()) / (radar_data[metric].max() - radar_data[metric].min())
        else:
            radar_data[metric + '_norm'] = 0.5
    
    fig = go.Figure()
    
    for _, row in radar_data.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[m + '_norm'] for m in available_metrics],
            theta=[m.replace('_Moyenne', '').replace('_Moyen', '') for m in available_metrics],
            fill='toself',
            name=row['Catégorie']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="🕷️ Profil Radar Multi-Métriques des Catégories"
    )
    
    return fig


def generate_insights(comparison_df, detailed_df):
    """Génère des insights automatiques basés sur les données."""
    insights = []
    
    # Insight 1: Catégorie avec le plus d'images
    max_images_cat = comparison_df.loc[comparison_df['Nombre_Images'].idxmax()]
    insights.append(f"📷 **{max_images_cat['Catégorie']}** contient le plus d'images ({max_images_cat['Nombre_Images']:,})")
    
    # Insight 2: Meilleure qualité d'image
    if 'Netteté_Moyenne' in comparison_df.columns:
        best_quality_cat = comparison_df.loc[comparison_df['Netteté_Moyenne'].idxmax()]
        insights.append(f"🏆 **{best_quality_cat['Catégorie']}** a la meilleure netteté moyenne ({best_quality_cat['Netteté_Moyenne']:.2f})")
    
    # Insight 3: Plus grande variabilité
    most_variable_cat = comparison_df.loc[comparison_df['Écart_Type_Intensité'].idxmax()]
    insights.append(f"📊 **{most_variable_cat['Catégorie']}** présente la plus grande variabilité d'intensité")
    
    # Insight 4: Taille de fichier
    largest_files_cat = comparison_df.loc[comparison_df['Taille_Fichier_Moyenne_KB'].idxmax()]
    insights.append(f"💾 **{largest_files_cat['Catégorie']}** a les fichiers les plus volumineux ({largest_files_cat['Taille_Fichier_Moyenne_KB']:.1f} KB)")
    
    # Insight 5: Analyse de la distribution
    categories_count = len(comparison_df)
    if categories_count >= 3:
        insights.append(f"🔍 Analyse comparative sur **{categories_count} catégories** avec des caractéristiques distinctes")
    
    # Affichage des insights
    for insight in insights:
        st.info(insight)
    
    # Recommandations
    st.markdown("#### 💡 **Recommandations**")
    
    if 'Netteté_Moyenne' in comparison_df.columns:
        low_quality_cats = comparison_df[comparison_df['Netteté_Moyenne'] < comparison_df['Netteté_Moyenne'].mean()]
        if not low_quality_cats.empty:
            st.warning(f"⚠️ Considérer l'amélioration de la qualité pour: {', '.join(low_quality_cats['Catégorie'].tolist())}")
    
    # Équilibrage du dataset
    min_images = comparison_df['Nombre_Images'].min()
    max_images = comparison_df['Nombre_Images'].max()
    if max_images / min_images > 2:
        st.warning(f"⚖️ Dataset déséquilibré détecté (ratio {max_images/min_images:.1f}:1)")
        st.info("💡 Considérer l'augmentation de données ou le rééchantillonnage")


def show_global_analysis(data_dir, categories):
    """Affiche l'analyse globale des images avec des graphiques améliorés."""
    st.markdown("## 🔍 **Analyse Globale des Images**")
    
    if not categories:
        st.warning("⚠️ Aucune catégorie sélectionnée. Veuillez sélectionner au moins une catégorie.")
        return
    
    # Configuration de l'analyse
    analysis_config = get_analysis_mode_config()
    
    # Affichage du mode d'analyse sélectionné
    st.info(f"🎯 **Mode d'analyse**: {analysis_config['name']} | "
            f"📊 **Échantillon**: {analysis_config['sample_size'] if analysis_config['sample_size'] else 'Complet'} | "
            f"🚀 **Amélioré**: {'✅' if analysis_config['enhanced'] else '❌'}")
    
    # Sélection des catégories à analyser
    selected_categories = st.multiselect(
        "📂 Sélectionnez les catégories à analyser:",
        options=categories,
        default=categories,
        key="global_analysis_categories"
    )
    
    if not selected_categories:
        st.warning("⚠️ Veuillez sélectionner au moins une catégorie.")
        return
    
    # Estimation du temps de traitement
    total_estimated_time = estimate_processing_time(data_dir, selected_categories, analysis_config)
    if total_estimated_time > 30:
        st.warning(f"⏱️ Temps estimé: {total_estimated_time:.1f}s. Considérez le mode échantillon pour un traitement plus rapide.")
    
    if st.button("🚀 Lancer l'analyse globale", key="start_global_analysis"):
        st.markdown("---")
        
        # Conteneur pour les métriques en temps réel
        metrics_container = st.container()
        with metrics_container:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                processed_metric = st.empty()
            with col2:
                time_metric = st.empty()
            with col3:
                speed_metric = st.empty()
            with col4:
                eta_metric = st.empty()
        
        # Conteneur pour la barre de progression
        progress_container = st.container()
        
        start_time = time.time()
        total_processed = 0
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Analyse de chaque catégorie
            all_results = {}
            
            for i, category in enumerate(selected_categories):
                category_start_time = time.time()
                status_text.text(f"🔄 Analyse en cours: {category}")
                
                category_dir = data_dir / category / "images"
                
                if category_dir.exists():
                    # Récupérer les fichiers d'images
                    image_files = list(category_dir.glob("*.png")) + list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.jpeg"))
                    
                    # Limiter selon la configuration
                    if analysis_config['sample_size']:
                        image_files = image_files[:analysis_config['sample_size']]
                    
                    if image_files:
                        if analysis_config['enhanced']:
                            results = analyze_images_batch_enhanced(image_files, batch_size=10)
                            count = len(image_files)
                        else:
                            results = analyze_images_batch(image_files, batch_size=10)
                            count = len(image_files)
                        
                        all_results[category] = {'results': results, 'count': count}
                        total_processed += count
                    else:
                        st.warning(f"⚠️ Aucune image trouvée dans: {category_dir}")
                
                    # Mise à jour des métriques en temps réel
                    elapsed_time = time.time() - start_time
                    processing_speed = total_processed / elapsed_time if elapsed_time > 0 else 0
                    
                    processed_metric.metric("📷 Images traitées", f"{total_processed:,}")
                    time_metric.metric("⏱️ Temps écoulé", f"{elapsed_time:.1f}s")
                    speed_metric.metric("🚀 Vitesse", f"{processing_speed:.1f} img/s")
                    
                    # ETA pour les catégories restantes
                    remaining_categories = len(selected_categories) - (i + 1)
                    if remaining_categories > 0 and processing_speed > 0:
                        eta = remaining_categories * (elapsed_time / (i + 1))
                        eta_metric.metric("⏳ ETA", f"{eta:.1f}s")
                    else:
                        eta_metric.metric("⏳ ETA", "Terminé!")
                    
                else:
                    st.warning(f"⚠️ Dossier non trouvé: {category_dir}")
                
                # Mise à jour de la barre de progression
                progress_bar.progress((i + 1) / len(selected_categories))
            
            # Effacer la barre de progression
            progress_container.empty()
        
        # Temps total de traitement
        total_time = time.time() - start_time
        
        if all_results:
            # Métriques finales
            st.success(f"✅ Analyse terminée en {total_time:.1f}s pour {len(selected_categories)} catégorie(s)")
            
            final_col1, final_col2, final_col3 = st.columns(3)
            with final_col1:
                st.metric("🎯 Images analysées", f"{total_processed:,}")
            with final_col2:
                st.metric("⚡ Vitesse moyenne", f"{total_processed/total_time:.1f} img/s")
            with final_col3:
                st.metric("📊 Catégories", len(selected_categories))
            
            st.markdown("---")
            
            # Affichage de la comparaison améliorée
            comparison_df, detailed_df = display_global_comparison_enhanced(all_results, selected_categories)
            
            # Option de téléchargement des résultats
            if comparison_df is not None:
                st.markdown("#### � **Téléchargement des Résultats**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_summary = comparison_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Résumé (CSV)",
                        data=csv_summary,
                        file_name=f"analyse_globale_resume_{int(time.time())}.csv",
                        mime="text/csv",
                        key="download_summary"
                    )
                
                with col2:
                    if detailed_df is not None:
                        csv_detailed = detailed_df.to_csv(index=False)
                        st.download_button(
                            label="📋 Détails (CSV)",
                            data=csv_detailed,
                            file_name=f"analyse_globale_details_{int(time.time())}.csv",
                            mime="text/csv",
                            key="download_detailed"
                        )
                
                with col3:
                    # Rapport JSON complet
                    json_report = {
                        'metadata': {
                            'timestamp': time.time(),
                            'analysis_config': analysis_config,
                            'categories': selected_categories,
                            'processing_time': total_time,
                            'total_images': total_processed
                        },
                        'summary': comparison_df.to_dict('records'),
                        'detailed_results': all_results
                    }
                    
                    json_str = json.dumps(json_report, indent=2, default=str)
                    st.download_button(
                        label="📄 Rapport JSON",
                        data=json_str,
                        file_name=f"rapport_analyse_{int(time.time())}.json",
                        mime="application/json",
                        key="download_json"
                    )
        
        else:
            st.error("❌ Aucune donnée n'a pu être analysée.")
    
    # Informations sur l'analyse
    st.markdown("---")
    st.markdown("### ℹ️ **À propos de cette analyse**")
    
    with st.expander("📖 Explication des métriques"):
        st.markdown("""
        **Métriques de base:**
        - **Largeur/Hauteur**: Dimensions de l'image en pixels
        - **Intensité moyenne**: Luminosité moyenne de l'image (0-255)
        - **Écart-type**: Variation de l'intensité (contraste)
        - **Taille fichier**: Taille du fichier image en KB
        
        **Métriques avancées (mode amélioré):**
        - **Entropie**: Mesure de la complexité/information dans l'image
        - **Netteté**: Mesure de la définition des contours (Laplacian variance)
        - **Contraste**: Ratio entre zones claires et sombres
        - **Densité des contours**: Quantité de détails/textures détectées
        - **Luminosité**: Clarté générale de l'image
        - **Asymétrie/Kurtosis**: Distribution statistique des intensités
        """)
    
    with st.expander("📊 Interprétation des graphiques"):
        st.markdown("""
        - **Graphique radar**: Compare les profils multi-métriques des catégories
        - **Box plots**: Montrent la distribution et les valeurs aberrantes
        - **Violin plots**: Révèlent la forme de la distribution des données
        - **Heatmap**: Visualise les corrélations entre métriques
        - **Graphique 3D**: Explore les relations dans l'espace multidimensionnel
        - **Tests statistiques**: ANOVA pour détecter les différences significatives
        """)
    
    with st.expander("🔬 Tests statistiques"):
        st.markdown("""
        - **Test ANOVA**: Détermine s'il existe des différences significatives entre catégories
        - **p-value < 0.05**: Indique des différences statistiquement significatives
        - **Matrice de corrélation**: Révèle les relations entre différentes métriques
        - **F-statistique**: Mesure de la variance entre groupes vs variance intra-groupe
        """)
    
    with st.expander("⚡ Optimisation des performances"):
        st.markdown("""
        - **Mode Rapide**: Échantillon de 100 images, métriques de base
        - **Mode Équilibré**: Échantillon de 500 images, métriques avancées
        - **Mode Complet**: Toutes les images, analyse exhaustive
        - **Mode Personnalisé**: Configuration manuelle selon vos besoins
        
        💡 **Conseil**: Utilisez le mode rapide pour l'exploration, le mode complet pour l'analyse finale.
        """)


def estimate_processing_time(data_dir, categories, config):
    """Estime le temps de traitement nécessaire."""
    total_images = 0
    
    for category in categories:
        category_dir = data_dir / category / "images"
        if category_dir.exists():
            image_files = list(category_dir.glob("*.png")) + list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.jpeg"))
            if config['sample_size']:
                total_images += min(len(image_files), config['sample_size'])
            else:
                total_images += len(image_files)
    
    # Estimation basée sur des benchmarks (images/seconde)
    base_speed = 50  # images/seconde pour analyse de base
    enhanced_speed = 20  # images/seconde pour analyse améliorée
    
    speed = enhanced_speed if config['enhanced'] else base_speed
    return total_images / speed


# Fonction principale pour l'onglet images
def show_image_analysis(data_dir, categories):
    """Affiche l'analyse détaillée des images avec métriques avancées."""
    st.markdown("## 🖼️ **Analyse Avancée des Images**")
    
    # Tabs pour différents types d'analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Analyse de Base",
        "📊 Analyse Globale", 
        "🔬 Métriques Avancées", 
        "🎲 Échantillons Aléatoires",
        "🏥 Analyse Radiologique"
    ])
    
    with tab1:
        show_basic_image_analysis(data_dir, categories)
    
    with tab2:
        show_global_analysis(data_dir, categories)
    
    with tab3:
        show_advanced_image_metrics(data_dir, categories)
    
    with tab4:
        show_random_image_grid(data_dir, categories)
    
    with tab5:
        show_radiological_analysis(data_dir, categories)


def show_images_tab(data_dir, categories):
    """Affiche l'onglet images."""
    show_image_analysis(data_dir, categories)

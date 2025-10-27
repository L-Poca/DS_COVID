"""
Widget pour les configurations communes entre les pages de modélisation.
Contient les sidebars, paramètres généraux et configurations partagées.
"""

import streamlit as st
from pathlib import Path


def create_data_source_config():
    """
    Widget pour configurer la source des données.
    
    Returns:
        dict: Configuration de la source des données
    """
    data_source = st.selectbox(
        "Source des données",
        ["Dataset COVID-19 (Réel)", "Données simulées (test)", "Upload personnalisé"]
    )
    
    config = {"data_source": data_source}
    
    # Paramètres spécifiques au dataset COVID
    if data_source == "Dataset COVID-19 (Réel)":
        st.subheader("🦠 Paramètres COVID-19")
        
        # Import des constantes (à adapter selon votre structure)
        try:
            from src.features.Data_Loaders.covid_data_loader import CLASS_NAMES
            default_classes = list(CLASS_NAMES)
        except ImportError:
            default_classes = ["Normal", "Lung_Opacity", "COVID", "Viral Pneumonia"]
        
        config.update({
            "selected_classes": st.multiselect(
                "Classes à inclure",
                options=default_classes,
                default=default_classes,
                help="Sélectionnez les classes médicales à inclure"
            ),
            "max_images_per_class": st.number_input(
                "Max images par classe",
                min_value=10,
                max_value=5000,
                value=500,
                help="Limitez le nombre d'images par classe pour un traitement plus rapide"
            ),
            "target_size": st.selectbox(
                "Taille des images",
                [(128, 128), (224, 224), (256, 256)],
                index=1,
                help="Taille de redimensionnement des images"
            ),
            "color_mode": st.selectbox(
                "Mode couleur",
                ["rgb", "grayscale"],
                index=0,
                help="Mode couleur des images"
            )
        })
    
    elif data_source == "Données simulées (test)":
        config.update({
            "n_samples": st.slider("Nombre d'échantillons", 100, 5000, 1000),
            "n_features": st.selectbox("Taille des features", [1024, 4096, 16384], index=2),
            "n_classes": st.selectbox("Nombre de classes", [2, 4], index=1)
        })
    
    return config


def create_general_parameters():
    """
    Widget pour les paramètres généraux de modélisation.
    
    Returns:
        dict: Paramètres généraux
    """
    st.subheader("📊 Paramètres Généraux")
    
    return {
        "test_size": st.slider("Taille du jeu de test (%)", 10, 50, 20) / 100,
        "validation_size": st.slider("Taille du jeu de validation (%)", 5, 25, 10) / 100,
        "random_state": st.number_input("Seed aléatoire", 1, 9999, 42)
    }


def create_pipeline_type_selector():
    """
    Widget pour sélectionner le type de pipeline.
    
    Returns:
        str: Type de pipeline sélectionné
    """
    return st.selectbox(
        "Type de pipeline",
        ["Sklearn (ML Classique)", "TensorFlow (Deep Learning)", "Augmentation de données"]
    )


def create_model_selection_widget():
    """
    Widget pour la sélection de modèle dans les pages d'évaluation et prédiction.
    
    Returns:
        tuple: (selected_model, selected_model_name, model_info)
    """
    st.subheader("🤖 Sélection du Modèle")
    
    model_source = st.radio(
        "Source du modèle",
        ["Session courante", "Modèle sauvegardé"],
        help="Choisir entre les modèles de la session ou charger un modèle sauvegardé"
    )
    
    selected_model = None
    selected_model_name = None
    model_info = {}
    
    if model_source == "Session courante":
        if 'training_results' in st.session_state:
            results = st.session_state['training_results']
            model_names = list(results.keys())
            
            if model_names:
                selected_model_name = st.selectbox(
                    "Choisissez un modèle:",
                    options=model_names,
                    help="Modèles disponibles de la session d'entraînement"
                )
                
                # Afficher les informations du modèle sélectionné
                if selected_model_name:
                    result = results[selected_model_name]
                    model_info = result
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Précision Test", f"{result.get('test_accuracy', 0):.4f}")
                    with col2:
                        st.metric("F1-Score", f"{result.get('test_f1', 0):.4f}")
                    with col3:
                        st.metric("Score CV", f"{result.get('cv_mean', 0):.4f}")
                    
                    # Récupérer le modèle
                    selected_model = result.get('pipeline')
            else:
                st.warning("⚠️ Aucun modèle disponible en session. Entraînez d'abord un modèle.")
        else:
            st.warning("⚠️ Aucun modèle en session. Allez dans la page Entraînement.")
    
    else:  # Modèle sauvegardé
        st.info("🚧 Chargement de modèles sauvegardés à implémenter")
    
    return selected_model, selected_model_name, model_info


def display_data_statistics(data_stats, total_images=None):
    """
    Widget pour afficher les statistiques des données.
    
    Args:
        data_stats (dict): Statistiques par classe
        total_images (int, optional): Nombre total d'images
    """
    st.subheader("📊 Statistiques des Données")
    
    if data_stats:
        stats_data = []
        calculated_total = 0
        
        for class_name, stats in data_stats.items():
            stats_data.append({
                "Classe": class_name,
                "Images": stats.get("n_images", 0),
                "Masques": stats.get("n_masks", 0),
                "Disponible": "✅" if stats.get("images_available", False) else "❌"
            })
            if stats.get("images_available", False):
                calculated_total += stats.get("n_images", 0)
        
        stats_df = st.dataframe(stats_data, use_container_width=True)
        
        total_to_display = total_images if total_images is not None else calculated_total
        
        if total_to_display == 0:
            st.error("❌ Aucune donnée trouvée. Vérifiez le chemin des données.")
            return False
        else:
            st.success(f"✅ {total_to_display} images disponibles au total")
            return True
    
    return False


def create_training_options():
    """
    Widget pour les options d'entraînement communes.
    
    Returns:
        dict: Options d'entraînement
    """
    st.subheader("🎛️ Options d'Entraînement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_comparison = st.checkbox("Activer la comparaison", True)
        save_results = st.checkbox("Sauvegarder les résultats", True)
    
    with col2:
        use_gpu = st.checkbox("Utiliser GPU", True, help="Utiliser le GPU si disponible")
        verbose = st.checkbox("Mode verbeux", False, help="Afficher plus de détails")
    
    return {
        "enable_comparison": enable_comparison,
        "save_results": save_results,
        "use_gpu": use_gpu,
        "verbose": verbose
    }


def create_tensorflow_specific_params():
    """
    Widget pour les paramètres spécifiques à TensorFlow.
    
    Returns:
        dict: Paramètres TensorFlow
    """
    st.subheader("🔧 Paramètres TensorFlow")
    
    return {
        "epochs": st.slider("Nombre d'époques", 5, 100, 20),
        "batch_size": st.selectbox("Batch size", [8, 16, 32, 64], index=2),
        "learning_rate": st.selectbox("Learning rate", [0.001, 0.01, 0.1], index=0),
        "validation_split": st.slider("Split validation", 0.1, 0.3, 0.2)
    }


def show_configuration_summary(config):
    """
    Widget pour afficher un résumé de la configuration.
    
    Args:
        config (dict): Configuration complète
    """
    with st.expander("📋 Résumé de la Configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Source des données:**")
            st.write(f"- Type: {config.get('data_source', 'Non défini')}")
            
            if config.get('selected_classes'):
                st.write(f"- Classes: {', '.join(config['selected_classes'])}")
            
            if config.get('n_samples'):
                st.write(f"- Échantillons: {config['n_samples']}")
        
        with col2:
            st.write("**Paramètres d'entraînement:**")
            st.write(f"- Test size: {config.get('test_size', 0)*100:.0f}%")
            st.write(f"- Random state: {config.get('random_state', 42)}")
            
            if config.get('epochs'):
                st.write(f"- Époques: {config['epochs']}")
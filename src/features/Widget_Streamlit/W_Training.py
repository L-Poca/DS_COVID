"""
Widgets spécifiques à la page Training (Entraînement).
Contient les composants pour la configuration des pipelines, l'entraînement et l'affichage des résultats.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import json


def create_pipeline_configuration_tab(project_root, pipeline_type):
    """
    Widget pour l'onglet configuration des pipelines.
    
    Args:
        project_root (Path): Chemin racine du projet
        pipeline_type (str): Type de pipeline sélectionné
        
    Returns:
        tuple: (selected_configs, manager_instance)
    """
    st.header("Configuration des Pipelines")
    
    selected_configs = []
    manager = None
    
    try:
        if pipeline_type == "Sklearn (ML Classique)":
            from src.features.Pipelines.Pipeline_Sklearn import PipelineManager
            
            config_path = project_root / "src" / "features" / "Pipelines" / "Pipeline_Sklearn_config.json"
            if not config_path.exists():
                st.error(f"❌ Fichier de configuration non trouvé: {config_path}")
                return selected_configs, manager
            
            manager = PipelineManager(str(config_path))
            configs = manager.get_available_configs()
            
            st.subheader("📋 Pipelines Sklearn Disponibles")
            if configs:
                config_df = pd.DataFrame(configs)
                st.dataframe(config_df, use_container_width=True)
                
                selected_configs = st.multiselect(
                    "Choisissez les pipelines à entraîner:",
                    options=[config['name'] for config in configs],
                    default=['basic_rf', 'medical_specialized'] if any(c['name'] in ['basic_rf', 'medical_specialized'] for c in configs) else [configs[0]['name']] if configs else [],
                    help="Sélectionnez un ou plusieurs pipelines pour comparaison"
                )
        
        elif pipeline_type == "TensorFlow (Deep Learning)":
            from src.features.Pipelines.Pipeline_TensorFlow import TensorFlowPipelineManager
            
            config_path = project_root / "src" / "features" / "Pipelines" / "Pipeline_TensorFlow_config.json"
            if not config_path.exists():
                st.error(f"❌ Fichier de configuration TensorFlow non trouvé: {config_path}")
                return selected_configs, manager
            
            manager = TensorFlowPipelineManager(str(config_path))
            configs = manager.get_available_configs()
            
            st.subheader("📋 Modèles TensorFlow Disponibles")
            if configs:
                config_df = pd.DataFrame(configs)
                st.dataframe(config_df, use_container_width=True)
                
                selected_configs = st.multiselect(
                    "Choisissez les modèles à entraîner:",
                    options=[config['name'] for config in configs],
                    default=['basic_cnn', 'vgg16_transfer'] if any(c['name'] in ['basic_cnn', 'vgg16_transfer'] for c in configs) else [configs[0]['name']] if configs else [],
                    help="Sélectionnez un ou plusieurs modèles pour comparaison"
                )
        
        elif pipeline_type == "Augmentation de données":
            from src.features.Pipelines.Pipeline_DataAugmentation import DataAugmentationPipeline
            
            config_path = project_root / "src" / "features" / "Pipelines" / "Pipeline_DataAugmentation_config.json"
            if not config_path.exists():
                st.error(f"❌ Fichier de configuration d'augmentation non trouvé: {config_path}")
                return selected_configs, manager
            
            manager = DataAugmentationPipeline(str(config_path))
            strategies = manager.get_available_strategies()
            
            st.subheader("📋 Stratégies d'Augmentation Disponibles")
            if strategies:
                strategy_df = pd.DataFrame(strategies)
                st.dataframe(strategy_df, use_container_width=True)
                
                selected_configs = st.multiselect(
                    "Choisissez les stratégies d'augmentation:",
                    options=[strategy['name'] for strategy in strategies],
                    default=['medical_basic', 'covid_specific'] if any(s['name'] in ['medical_basic', 'covid_specific'] for s in strategies) else [strategies[0]['name']] if strategies else [],
                    help="Sélectionnez une ou plusieurs stratégies d'augmentation"
                )
        
        # Afficher les détails des configurations sélectionnées
        if selected_configs:
            st.success(f"✅ {len(selected_configs)} configuration(s) sélectionnée(s)")
            
            with st.expander("🔍 Détails des configurations sélectionnées"):
                for config_name in selected_configs:
                    st.write(f"**{config_name}**")
                    # Ici vous pouvez ajouter plus de détails selon vos besoins
                    st.write("Configuration détaillée à implémenter selon vos besoins spécifiques")
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de la configuration: {e}")
        selected_configs = []
        manager = None
    
    return selected_configs, manager


def create_training_tab(selected_configs, pipeline_type, data_config, general_params, tf_params=None):
    """
    Widget pour l'onglet d'entraînement.
    
    Args:
        selected_configs (list): Configurations sélectionnées
        pipeline_type (str): Type de pipeline
        data_config (dict): Configuration des données
        general_params (dict): Paramètres généraux
        tf_params (dict, optional): Paramètres TensorFlow
        
    Returns:
        dict: Résultats d'entraînement ou None
    """
    st.header("Entraînement des Modèles")
    
    if not selected_configs:
        st.warning("⚠️ Veuillez sélectionner au moins un pipeline dans l'onglet Configuration")
        return None
    
    # Affichage des paramètres d'entraînement
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Données")
        if data_config["data_source"] == "Dataset COVID-19 (Réel)":
            st.info(f"Classes sélectionnées: {', '.join(data_config.get('selected_classes', []))}")
            st.info(f"Max images/classe: {data_config.get('max_images_per_class', 'N/A')}")
            st.info(f"Taille images: {data_config.get('target_size', 'N/A')}")
            st.info(f"Mode couleur: {data_config.get('color_mode', 'N/A')}")
        elif data_config["data_source"] == "Données simulées (test)":
            st.info(f"Échantillons: {data_config.get('n_samples', 'N/A')}")
            st.info(f"Features: {data_config.get('n_features', 'N/A')}")
            st.info(f"Classes: {data_config.get('n_classes', 'N/A')}")
    
    with col2:
        st.subheader("🎛️ Paramètres")
        st.info(f"Test size: {general_params.get('test_size', 0)*100:.0f}%")
        st.info(f"Validation size: {general_params.get('validation_size', 0)*100:.0f}%")
        st.info(f"Random state: {general_params.get('random_state', 42)}")
        
        if pipeline_type == "TensorFlow (Deep Learning)" and tf_params:
            st.info(f"Époques: {tf_params.get('epochs', 20)}")
            st.info(f"Batch size: {tf_params.get('batch_size', 32)}")
    
    # Bouton d'entraînement
    training_started = st.button("🚀 Lancer l'Entraînement", type="primary")
    
    if training_started:
        # Placeholder pour la logique d'entraînement
        st.info("🔄 Entraînement en cours...")
        
        # Ici sera implémentée la logique d'entraînement réelle
        # Pour l'instant, on retourne une structure de base
        
        with st.spinner("Entraînement en cours..."):
            # Simulation d'entraînement
            import time
            time.sleep(2)
        
        st.success("✅ Entraînement simulé terminé!")
        
        # Structure de résultats simulée
        results = {}
        for config_name in selected_configs:
            results[config_name] = {
                'test_accuracy': 0.85 + (hash(config_name) % 100) / 1000,  # Simulation
                'test_f1': 0.82 + (hash(config_name) % 80) / 1000,
                'cv_mean': 0.80 + (hash(config_name) % 120) / 1000,
                'cv_std': 0.02 + (hash(config_name) % 30) / 10000,
                'training_time': pd.Timedelta(seconds=30 + (hash(config_name) % 180)),
                'pipeline': f"MockPipeline_{config_name}",  # Mock pipeline
                'timestamp': datetime.now()
            }
        
        return results
    
    return None


def create_results_comparison_tab():
    """
    Widget pour l'onglet de comparaison des résultats.
    
    Returns:
        None
    """
    st.header("Comparaison des Résultats")
    
    if 'training_results' in st.session_state:
        results = st.session_state['training_results']
        
        if results:
            # Tableau de comparaison
            st.subheader("📊 Tableau Comparatif")
            
            comparison_data = []
            for config_name, result in results.items():
                row = {
                    'Pipeline': config_name,
                    'Précision Test': f"{result.get('test_accuracy', 0):.4f}",
                    'F1-Score': f"{result.get('test_f1', 0):.4f}",
                    'Score CV': f"{result.get('cv_mean', 0):.4f}",
                    'Écart-type CV': f"{result.get('cv_std', 0):.4f}",
                    'Temps (s)': f"{result.get('training_time', pd.Timedelta(0)).total_seconds():.1f}"
                }
                
                if 'best_score' in result:
                    row['Meilleur Score'] = f"{result['best_score']:.4f}"
                
                comparison_data.append(row)
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
            
            # Meilleur modèle
            best_model = max(results.items(), key=lambda x: x[1].get('test_accuracy', 0))
            st.success(f"🏆 **Meilleur modèle**: {best_model[0]} avec une précision de {best_model[1].get('test_accuracy', 0):.4f}")
            
            # Graphiques de comparaison
            st.subheader("📈 Visualisations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Graphique en barres des précisions
                metrics_df = pd.DataFrame({
                    'Pipeline': [r['Pipeline'] for r in comparison_data],
                    'Précision': [float(r['Précision Test']) for r in comparison_data]
                })
                st.bar_chart(metrics_df.set_index('Pipeline'))
            
            with col2:
                # Temps d'entraînement
                time_df = pd.DataFrame({
                    'Pipeline': [r['Pipeline'] for r in comparison_data],
                    'Temps': [float(r['Temps (s)']) for r in comparison_data]
                })
                st.bar_chart(time_df.set_index('Pipeline'))
            
            # Bouton de téléchargement des résultats
            create_results_download_widget(df_comparison)
    
    else:
        st.info("ℹ️ Aucun résultat d'entraînement disponible. Lancez d'abord un entraînement dans l'onglet précédent.")


def create_results_download_widget(df_comparison):
    """
    Widget pour télécharger les résultats.
    
    Args:
        df_comparison (pd.DataFrame): DataFrame des résultats à télécharger
    """
    csv_data = df_comparison.to_csv(index=False)
    
    st.download_button(
        label="📥 Télécharger les résultats (CSV)",
        data=csv_data,
        file_name=f"covid_pipeline_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def save_training_results_to_session(results, pipeline_type, comparison_data):
    """
    Sauvegarde les résultats d'entraînement dans la session Streamlit.
    
    Args:
        results (dict): Résultats d'entraînement
        pipeline_type (str): Type de pipeline utilisé
        comparison_data (dict): Données de comparaison
    """
    # Sauvegarder dans la session
    st.session_state['training_results'] = results
    st.session_state['pipeline_type'] = pipeline_type
    st.session_state['comparison_data'] = comparison_data
    
    st.success("✅ Résultats sauvegardés dans la session!")


def create_data_verification_widget(project_root, data_config):
    """
    Widget pour vérifier la disponibilité des données.
    
    Args:
        project_root (Path): Chemin racine du projet
        data_config (dict): Configuration des données
        
    Returns:
        bool: True si les données sont disponibles
    """
    if data_config["data_source"] == "Dataset COVID-19 (Réel)":
        st.subheader("📊 Vérification des Données")
        
        with st.spinner("🔍 Vérification de la disponibilité des données..."):
            try:
                from src.features.Data_Loaders.covid_data_loader import check_data_availability
                data_stats = check_data_availability(project_root)
            except ImportError:
                st.error("❌ Impossible d'importer check_data_availability")
                return False
            except Exception as e:
                st.error(f"❌ Erreur lors de la vérification des données: {e}")
                return False
        
        # Afficher les statistiques avec le widget de configuration commune
        from .W_Configuration_Commune import display_data_statistics
        return display_data_statistics(data_stats)
    
    return True  # Pour les autres sources de données
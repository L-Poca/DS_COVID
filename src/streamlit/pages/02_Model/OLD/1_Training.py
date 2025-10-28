import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Ajouter le répertoire racine au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))  # Ajouter le dossier streamlit

# Calcul correct du project_root
current_file = Path(__file__)  # pages/02_Model/1_Training.py
current_dir = current_file.parent  # pages/02_Model/
streamlit_dir = current_dir.parent.parent  # streamlit/
src_dir = streamlit_dir.parent  # src/
project_root = src_dir.parent  # DS_COVID/

sys.path.append(str(project_root))

# Imports des widgets modulaires
try:
    from src.features.Widget_Streamlit import (
        create_data_source_config,
        create_general_parameters,
        create_pipeline_type_selector,
        create_training_options,
        create_tensorflow_specific_params,
        show_configuration_summary,
        create_pipeline_configuration_tab,
        create_training_tab,
        create_results_comparison_tab,
        create_data_verification_widget,
        save_training_results_to_session
    )
    WIDGETS_OK = True
except ImportError as e:
    st.error(f"❌ Erreur d'import des widgets: {e}")
    WIDGETS_OK = False

try:
    from src.features.Pipelines.Pipeline_Sklearn import PipelineManager
    from src.features.Pipelines.Pipeline_DataAugmentation import DataAugmentationPipeline
    from src.features.Pipelines.Pipeline_TensorFlow import TensorFlowPipelineManager
    from src.features.Data_Loaders.covid_data_loader import (
        load_covid_dataset, check_data_availability, prepare_data_for_sklearn,
        prepare_data_for_tensorflow, CLASS_NAMES, COVID_CLASSES, get_data_paths
    )
    IMPORTS_OK = True
except ImportError as e:
    st.error(f"❌ Erreur d'import: {e}")
    IMPORTS_OK = False

st.title("🏋️ Entraînement du Modèle")

st.markdown("""
Cette page permet d'entraîner différents modèles de classification pour la détection COVID-19 
en utilisant des pipelines sklearn configurables.
""")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Sélection du type de données
    data_source = st.selectbox(
        "Source des données",
        ["Dataset COVID-19 (Réel)", "Données simulées (test)", "Upload personnalisé"]
    )
    
    # Pipeline type
    pipeline_type = st.selectbox(
        "Type de pipeline",
        ["Sklearn (ML Classique)", "TensorFlow (Deep Learning)", "Augmentation de données"]
    )
    
    # Paramètres généraux
    test_size = st.slider("Taille du jeu de test (%)", 10, 50, 20) / 100
    validation_size = st.slider("Taille du jeu de validation (%)", 5, 25, 10) / 100
    random_state = st.number_input("Seed aléatoire", 1, 9999, 42)
    
    # Paramètres spécifiques au dataset COVID
    if data_source == "Dataset COVID-19 (Réel)":
        st.subheader("🦠 Paramètres COVID-19")
        selected_classes = st.multiselect(
            "Classes à inclure",
            options=list(CLASS_NAMES),
            default=CLASS_NAMES,
            help="Sélectionnez les classes médicales à inclure"
        )
        
        max_images_per_class = st.number_input(
            "Max images par classe",
            min_value=10,
            max_value=5000,
            value=500,
            help="Limitez le nombre d'images par classe pour un traitement plus rapide"
        )
        
        target_size = st.selectbox(
            "Taille des images",
            [(128, 128), (224, 224), (256, 256)],
            index=1,
            help="Taille de redimensionnement des images"
        )
        
        color_mode = st.selectbox(
            "Mode couleur",
            ["rgb", "grayscale"],
            index=0 if pipeline_type == "TensorFlow (Deep Learning)" else 1
        )

# Interface principale
tab1, tab2, tab3 = st.tabs(["� Configuration", "🚀 Entraînement", "📊 Résultats"])

with tab1:
    st.header("Configuration des Pipelines")
    
    if not IMPORTS_OK:
        st.error("❌ Impossible de charger les modules nécessaires")
        st.stop()
    
    # Vérification de la disponibilité des données
    if data_source == "Dataset COVID-19 (Réel)":
        st.subheader("📊 Vérification des Données")
        
        with st.spinner("🔍 Vérification de la disponibilité des données..."):
            data_stats = check_data_availability(project_root)
        
        # Afficher les statistiques
        stats_data = []
        total_images = 0
        
        for class_name, stats in data_stats.items():
            stats_data.append({
                "Classe": class_name,
                "Images": stats["n_images"],
                "Masques": stats["n_masks"],
                "Disponible": "✅" if stats["images_available"] else "❌"
            })
            if stats["images_available"]:
                total_images += stats["n_images"]
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        if total_images == 0:
            st.error("❌ Aucune donnée COVID-19 trouvée. Vérifiez le chemin des données.")
            st.info("📁 Données attendues dans: data/raw/COVID-19_Radiography_Dataset/")
            st.stop()
        else:
            st.success(f"✅ {total_images} images disponibles au total")
    
    # Charger le gestionnaire de pipelines selon le type
    try:
        if pipeline_type == "Sklearn (ML Classique)":
            config_path = project_root / "src" / "features" / "Pipelines" / "Pipeline_Sklearn_config.json"
            if not config_path.exists():
                st.error(f"❌ Fichier de configuration non trouvé: {config_path}")
                st.stop()
            
            manager = PipelineManager(str(config_path))
            configs = manager.get_available_configs()
            
            st.subheader("📋 Pipelines Sklearn Disponibles")
            config_df = pd.DataFrame(configs)
            st.dataframe(config_df, use_container_width=True)
            
            selected_configs = st.multiselect(
                "Choisissez les pipelines à entraîner:",
                options=[config['name'] for config in configs],
                default=['basic_rf', 'medical_specialized'] if configs else [],
                help="Sélectionnez un ou plusieurs pipelines pour comparaison"
            )
        
        elif pipeline_type == "TensorFlow (Deep Learning)":
            config_path = project_root / "src" / "features" / "Pipelines" / "Pipeline_TensorFlow_config.json"
            if not config_path.exists():
                st.error(f"❌ Fichier de configuration TensorFlow non trouvé: {config_path}")
                st.stop()
            
            tf_manager = TensorFlowPipelineManager(str(config_path))
            configs = tf_manager.get_available_configs()
            
            st.subheader("📋 Modèles TensorFlow Disponibles")
            config_df = pd.DataFrame(configs)
            st.dataframe(config_df, use_container_width=True)
            
            selected_configs = st.multiselect(
                "Choisissez les modèles à entraîner:",
                options=[config['name'] for config in configs],
                default=['basic_cnn', 'vgg16_transfer'] if configs else [],
                help="Sélectionnez un ou plusieurs modèles pour comparaison"
            )
            
            # Paramètres spécifiques TensorFlow
            st.subheader("🔧 Paramètres TensorFlow")
            epochs = st.slider("Nombre d'époques", 5, 100, 20)
            batch_size = st.selectbox("Batch size", [8, 16, 32, 64], index=2)
        
        elif pipeline_type == "Augmentation de données":
            config_path = project_root / "src" / "features" / "Pipelines" / "Pipeline_DataAugmentation_config.json"
            if not config_path.exists():
                st.error(f"❌ Fichier de configuration d'augmentation non trouvé: {config_path}")
                st.stop()
            
            aug_manager = DataAugmentationPipeline(str(config_path))
            strategies = aug_manager.get_available_strategies()
            
            st.subheader("📋 Stratégies d'Augmentation Disponibles")
            strategy_df = pd.DataFrame(strategies)
            st.dataframe(strategy_df, use_container_width=True)
            
            selected_configs = st.multiselect(
                "Choisissez les stratégies d'augmentation:",
                options=[strategy['name'] for strategy in strategies],
                default=['medical_basic', 'covid_specific'] if strategies else [],
                help="Sélectionnez une ou plusieurs stratégies d'augmentation"
            )
        
        if selected_configs:
            st.success(f"✅ {len(selected_configs)} configuration(s) sélectionnée(s)")
            
            # Afficher les détails des configurations sélectionnées
            with st.expander("🔍 Détails des configurations sélectionnées"):
                for config_name in selected_configs:
                    if pipeline_type == "Sklearn (ML Classique)":
                        config_details = next((c for c in configs if c['name'] == config_name), None)
                        if config_details:
                            st.write(f"**{config_name}**: {config_details['description']}")
                            st.write(f"- GridSearch: {'✅' if config_details['grid_search'] else '❌'}")
                            st.write(f"- CV Folds: {config_details['cv_folds']}")
                    
                    elif pipeline_type == "TensorFlow (Deep Learning)":
                        config_details = next((c for c in configs if c['name'] == config_name), None)
                        if config_details:
                            st.write(f"**{config_name}**: {config_details['description']}")
                            st.write(f"- Architecture: {config_details['architecture_type']}")
                            st.write(f"- Input Shape: {config_details['input_shape']}")
                    
                    elif pipeline_type == "Augmentation de données":
                        strategy_details = next((s for s in strategies if s['name'] == config_name), None)
                        if strategy_details:
                            st.write(f"**{config_name}**: {strategy_details['description']}")
                            st.write(f"- Type: {strategy_details['type']}")
                            st.write(f"- Cible: {strategy_details['target_count']}")
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de la configuration: {e}")
        selected_configs = []

with tab2:
    st.header("Entraînement des Modèles")
    
    if not selected_configs:
        st.warning("⚠️ Veuillez sélectionner au moins un pipeline dans l'onglet Configuration")
    else:
        # Paramètres d'entraînement
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Données")
            if data_source == "Dataset COVID-19 (Réel)":
                st.info(f"Classes sélectionnées: {', '.join(selected_classes)}")
                st.info(f"Max images/classe: {max_images_per_class}")
                st.info(f"Taille images: {target_size}")
                st.info(f"Mode couleur: {color_mode}")
            elif data_source == "Données simulées (test)":
                n_samples = st.slider("Nombre d'échantillons", 100, 5000, 1000)
                n_features = st.selectbox("Taille des features", [1024, 4096, 16384], index=2)
                n_classes = st.selectbox("Nombre de classes", [2, 4], index=1)
            
        with col2:
            st.subheader("🎛️ Paramètres")
            enable_comparison = st.checkbox("Activer la comparaison", True)
            save_results = st.checkbox("Sauvegarder les résultats", True)
            
            if pipeline_type == "TensorFlow (Deep Learning)":
                use_gpu = st.checkbox("Utiliser GPU", True, help="Utiliser le GPU si disponible")
        
        # Bouton d'entraînement
        if st.button("🚀 Lancer l'Entraînement", type="primary"):
            # Initialiser la variable dataset
            dataset = None
            
            # Préparer les données selon la source
            if data_source == "Dataset COVID-19 (Réel)":
                with st.spinner("🔄 Chargement du dataset COVID-19..."):
                    try:
                        dataset = load_covid_dataset(
                            classes=selected_classes,
                            max_images_per_class=max_images_per_class,
                            target_size=target_size,
                            color_mode=color_mode,
                            test_size=test_size,
                            validation_size=validation_size,
                            random_state=random_state,
                            project_root=project_root
                        )
                        
                        st.success(f"✅ Dataset chargé: {dataset['metadata']['total_samples']} images")
                        
                        # Afficher la distribution
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Train", len(dataset['X_train']))
                        with col2:
                            st.metric("Validation", len(dataset['X_val']))
                        with col3:
                            st.metric("Test", len(dataset['X_test']))
                        
                        # Préparer selon le type de pipeline
                        if pipeline_type == "Sklearn (ML Classique)":
                            X_train = prepare_data_for_sklearn(dataset['X_train'])
                            X_test = prepare_data_for_sklearn(dataset['X_test'])
                            X_val = prepare_data_for_sklearn(dataset['X_val'])
                            y_train, y_test, y_val = dataset['y_train'], dataset['y_test'], dataset['y_val']
                        
                        elif pipeline_type == "TensorFlow (Deep Learning)":
                            tf_dataset = prepare_data_for_tensorflow(dataset)
                            X_train = tf_dataset['X_train']
                            X_test = tf_dataset['X_test']
                            X_val = tf_dataset['X_val']
                            y_train = tf_dataset['y_train_categorical']
                            y_test = tf_dataset['y_test_categorical']
                            y_val = tf_dataset['y_val_categorical']
                        
                        elif pipeline_type == "Augmentation de données":
                            # Pour l'augmentation, on utilise les chemins des dossiers
                            data_paths = get_data_paths(project_root)
                            X_train, X_test, X_val, y_train, y_test, y_val = None, None, None, None, None, None
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors du chargement: {e}")
                        st.stop()
                        
            elif data_source == "Données simulées (test)":
                with st.spinner("🔄 Génération des données simulées..."):
                    np.random.seed(random_state)
                    X = np.random.rand(n_samples, n_features)
                    y = np.random.randint(0, n_classes, n_samples)
                    
                    # Division train/test
                    split_idx = int((1 - test_size) * n_samples)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    st.success(f"✅ Données générées: {X_train.shape[0]} train, {X_test.shape[0]} test")
                
                # Entraînement selon le type de pipeline
                results_container = st.container()
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = {}
                
                if pipeline_type == "Sklearn (ML Classique)":
                    # Entraînement Sklearn
                    for i, config_name in enumerate(selected_configs):
                        status_text.text(f"🏋️ Entraînement Sklearn: {config_name}...")
                        progress_bar.progress((i) / len(selected_configs))
                        
                        with results_container:
                            with st.expander(f"📈 Résultats - {config_name}", expanded=True):
                                try:
                                    start_time = time.time()
                                    
                                    result = manager.train_pipeline(
                                        config_name,
                                        X_train, y_train,
                                        X_test, y_test
                                    )
                                    
                                    end_time = time.time()
                                    training_duration = end_time - start_time
                                    
                                    results[config_name] = result
                                    
                                    # Affichage des métriques
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Précision Test", f"{result.get('test_accuracy', 0):.3f}")
                                    with col2:
                                        st.metric("F1-Score", f"{result.get('test_f1', 0):.3f}")
                                    with col3:
                                        st.metric("Score CV", f"{result.get('cv_mean', 0):.3f}")
                                    
                                    if 'best_params' in result:
                                        st.write("🔧 **Meilleurs paramètres:**")
                                        st.json(result['best_params'])
                                    
                                    st.success(f"✅ {config_name} terminé en {training_duration:.1f}s")
                                    
                                except Exception as e:
                                    st.error(f"❌ Erreur avec {config_name}: {str(e)}")
                
                elif pipeline_type == "TensorFlow (Deep Learning)":
                    # Entraînement TensorFlow
                    for i, config_name in enumerate(selected_configs):
                        status_text.text(f"🧠 Entraînement TensorFlow: {config_name}...")
                        progress_bar.progress((i) / len(selected_configs))
                        
                        with results_container:
                            with st.expander(f"📈 Résultats - {config_name}", expanded=True):
                                try:
                                    start_time = time.time()
                                    
                                    # Créer le modèle
                                    model = tf_manager.create_model(config_name)
                                    
                                    # Configurer les générateurs de données (simulation)
                                    # En réalité, il faudrait créer les générateurs appropriés
                                    # Pour l'instant, on utilise les données directement
                                    
                                    # Configuration simplifiée pour TensorFlow
                                    # (En production, utiliser les vrais générateurs de données)
                                    
                                    # Entraînement simplifié (à adapter selon vos besoins)
                                    history = model.fit(
                                        X_train, y_train,
                                        validation_data=(X_test, y_test),
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        verbose=0
                                    )
                                    
                                    # Évaluation
                                    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                                    
                                    end_time = time.time()
                                    training_duration = end_time - start_time
                                    
                                    result = {
                                        'model': model,
                                        'history': history,
                                        'test_accuracy': test_accuracy,
                                        'test_loss': test_loss,
                                        'training_time': training_duration,
                                        'config_name': config_name
                                    }
                                    
                                    results[config_name] = result
                                    
                                    # Affichage des métriques
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Précision Test", f"{test_accuracy:.3f}")
                                    with col2:
                                        st.metric("Loss Test", f"{test_loss:.3f}")
                                    with col3:
                                        st.metric("Époques", epochs)
                                    
                                    st.success(f"✅ {config_name} terminé en {training_duration:.1f}s")
                                    
                                except Exception as e:
                                    st.error(f"❌ Erreur avec {config_name}: {str(e)}")
                
                elif pipeline_type == "Augmentation de données":
                    # Augmentation de données
                    for i, config_name in enumerate(selected_configs):
                        status_text.text(f"🔄 Augmentation: {config_name}...")
                        progress_bar.progress((i) / len(selected_configs))
                        
                        with results_container:
                            with st.expander(f"📈 Résultats - {config_name}", expanded=True):
                                try:
                                    start_time = time.time()
                                    
                                    # Configuration des dossiers
                                    input_base = project_root / "data" / "raw" / "COVID-19_Radiography_Dataset" / "COVID-19_Radiography_Dataset"
                                    output_base = project_root / "data" / "processed" / "augmented" / config_name
                                    
                                    # Traitement de toutes les classes sélectionnées
                                    aug_results = []
                                    for class_name in selected_classes:
                                        input_dir = input_base / class_name / "images"
                                        output_dir = output_base / class_name
                                        
                                        if input_dir.exists():
                                            class_result = aug_manager.process_class_directory(
                                                class_name=class_name,
                                                input_dir=str(input_dir),
                                                output_dir=str(output_dir),
                                                aug_config_name=config_name,
                                                target_count=None
                                            )
                                            aug_results.append(class_result)
                                    
                                    end_time = time.time()
                                    training_duration = end_time - start_time
                                    
                                    result = {
                                        'augmentation_results': aug_results,
                                        'training_time': training_duration,
                                        'config_name': config_name,
                                        'total_generated': sum([r.get('generated_count', 0) for r in aug_results])
                                    }
                                    
                                    results[config_name] = result
                                    
                                    # Affichage des métriques
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Classes traitées", len(aug_results))
                                    with col2:
                                        st.metric("Images générées", result['total_generated'])
                                    with col3:
                                        st.metric("Temps (s)", f"{training_duration:.1f}")
                                    
                                    st.success(f"✅ {config_name} terminé")
                                    
                                except Exception as e:
                                    st.error(f"❌ Erreur avec {config_name}: {str(e)}")
                
                # Finalisation
                progress_bar.progress(1.0)
                status_text.text("✅ Entraînement terminé!")
                
                # Sauvegarde des résultats dans la session
                st.session_state['training_results'] = results
                st.session_state['pipeline_type'] = pipeline_type
                st.session_state['comparison_data'] = {
                    'configs': selected_configs,
                    'pipeline_type': pipeline_type,
                    'data_source': data_source,
                    'data_info': dataset.get('metadata', {}) if data_source == "Dataset COVID-19 (Réel)" and dataset is not None else {
                        'n_samples': n_samples if data_source == "Données simulées (test)" else 0,
                        'n_features': n_features if data_source == "Données simulées (test)" else 0,
                        'n_classes': n_classes if data_source == "Données simulées (test)" else len(selected_classes),
                        'test_size': test_size
                    }
                }
                
            else:
                st.info("🚧 Chargement de données réelles non encore implémenté")

with tab3:
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
                    'Temps (s)': f"{result['training_time'].total_seconds():.1f}"
                }
                
                if 'best_score' in result:
                    row['Meilleur GridSearch'] = f"{result['best_score']:.4f}"
                
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
            csv_data = df_comparison.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les résultats (CSV)",
                data=csv_data,
                file_name=f"covid_pipeline_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("ℹ️ Aucun résultat d'entraînement disponible. Lancez d'abord un entraînement dans l'onglet précédent.")

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time
from PIL import Image
import io
import joblib
from datetime import datetime

# Calcul correct du project_root
current_file = Path(__file__)  # pages/02_Model/3_Prediction.py
current_dir = current_file.parent  # pages/02_Model/
streamlit_dir = current_dir.parent.parent  # streamlit/
src_dir = streamlit_dir.parent  # src/
project_root = src_dir.parent  # DS_COVID/

sys.path.append(str(project_root))

try:
    from src.features.Pipelines.Pipeline_Sklearn import PipelineManager
except ImportError:
    st.error("❌ Impossible d'importer PipelineManager. Vérifiez le chemin d'accès.")

st.title("🔮 Prédiction COVID-19")

st.markdown("""
Cette page permet de faire des prédictions avec les modèles entraînés.
Vous pouvez utiliser des données simulées ou charger vos propres données.
""")

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Mode de prédiction
    prediction_mode = st.selectbox(
        "Mode de prédiction",
        ["Données simulées", "Upload fichier", "Saisie manuelle", "Batch prédiction"]
    )
    
    # Sélection du modèle
    st.subheader("🤖 Modèle")
    
    model_source = st.radio(
        "Source du modèle",
        ["Session courante", "Modèle sauvegardé"],
        help="Choisir entre les modèles de la session ou charger un modèle sauvegardé"
    )

# Interface principale
tab1, tab2, tab3 = st.tabs(["🎯 Prédiction Simple", "📊 Batch Prédiction", "📈 Analyse"])

with tab1:
    st.header("Prédiction Simple")
    
    # Sélection du modèle
    selected_model = None
    
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
    
    # Interface de prédiction selon le mode
    if selected_model is not None:
        st.divider()
        
        if prediction_mode == "Données simulées":
            st.subheader("🎲 Génération de Données Test")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_samples_pred = st.slider("Nombre d'échantillons", 1, 100, 5)
                seed_pred = st.number_input("Seed aléatoire", 1, 9999, 123)
            
            with col2:
                # Récupérer les dimensions des features depuis la session
                comparison_data = st.session_state.get('comparison_data', {})
                data_info = comparison_data.get('data_info', {})
                default_features = data_info.get('n_features', 16384)
                default_classes = data_info.get('n_classes', 4)
                
                st.info(f"📊 Features: {default_features}")
                st.info(f"🏷️ Classes: {default_classes}")
            
            if st.button("🚀 Générer et Prédire", type="primary"):
                with st.spinner("🔄 Génération des données et prédiction..."):
                    try:
                        # Générer les données
                        np.random.seed(seed_pred)
                        X_pred = np.random.rand(n_samples_pred, default_features)
                        
                        # Faire les prédictions
                        predictions = selected_model.predict(X_pred)
                        probabilities = selected_model.predict_proba(X_pred)
                        
                        # Affichage des résultats
                        st.success("✅ Prédictions terminées!")
                        
                        # Tableau des résultats
                        results_df = pd.DataFrame({
                            'Échantillon': range(1, n_samples_pred + 1),
                            'Prédiction': predictions,
                            'Probabilité Max': [max(prob) for prob in probabilities],
                            'Confiance': [f"{max(prob):.2%}" for prob in probabilities]
                        })
                        
                        # Ajouter les probabilités par classe
                        for i in range(probabilities.shape[1]):
                            results_df[f'Prob Classe {i}'] = [f"{prob[i]:.3f}" for prob in probabilities]
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Statistiques
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Échantillons traités", n_samples_pred)
                        
                        with col2:
                            confidence_mean = np.mean([max(prob) for prob in probabilities])
                            st.metric("Confiance moyenne", f"{confidence_mean:.2%}")
                        
                        with col3:
                            unique_preds = len(np.unique(predictions))
                            st.metric("Classes prédites", unique_preds)
                        
                        # Graphique de distribution des prédictions
                        pred_counts = pd.Series(predictions).value_counts().sort_index()
                        
                        st.subheader("📊 Distribution des Prédictions")
                        st.bar_chart(pred_counts)
                        
                        # Graphique de confiance
                        st.subheader("🎯 Distribution de la Confiance")
                        confidence_scores = [max(prob) for prob in probabilities]
                        confidence_df = pd.DataFrame({'Confiance': confidence_scores})
                        st.histogram(confidence_df['Confiance'], bins=20)
                        
                        # Sauvegarder les résultats
                        st.session_state['last_predictions'] = {
                            'model_name': selected_model_name,
                            'predictions': predictions,
                            'probabilities': probabilities,
                            'timestamp': datetime.now(),
                            'n_samples': n_samples_pred
                        }
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
        
        elif prediction_mode == "Upload fichier":
            st.subheader("📁 Upload de Fichier")
            
            uploaded_file = st.file_uploader(
                "Choisir un fichier CSV",
                type=['csv'],
                help="Le fichier doit contenir les features en colonnes"
            )
            
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    st.success(f"✅ Fichier chargé: {df_upload.shape[0]} lignes, {df_upload.shape[1]} colonnes")
                    
                    # Aperçu des données
                    st.subheader("👀 Aperçu des Données")
                    st.dataframe(df_upload.head(), use_container_width=True)
                    
                    if st.button("🔮 Faire les Prédictions"):
                        with st.spinner("🔄 Prédiction en cours..."):
                            try:
                                # Préparer les données
                                X_upload = df_upload.values
                                
                                # Prédictions
                                predictions = selected_model.predict(X_upload)
                                probabilities = selected_model.predict_proba(X_upload)
                                
                                # Créer le DataFrame de résultats
                                results_df = df_upload.copy()
                                results_df['Prediction'] = predictions
                                results_df['Confidence'] = [max(prob) for prob in probabilities]
                                
                                st.success("✅ Prédictions terminées!")
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Option de téléchargement
                                csv_results = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Télécharger les Résultats",
                                    data=csv_results,
                                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                                
                            except Exception as e:
                                st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
                
                except Exception as e:
                    st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")
        
        elif prediction_mode == "Saisie manuelle":
            st.subheader("✏️ Saisie Manuelle des Features")
            
            # Interface simplifiée pour la saisie manuelle
            st.info("🚧 Saisie manuelle à implémenter selon le type de données spécifique")
            
            # Exemple d'interface
            with st.expander("🔧 Interface de saisie (exemple)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    feature1 = st.number_input("Feature 1", value=0.5)
                    feature2 = st.number_input("Feature 2", value=0.3)
                
                with col2:
                    feature3 = st.number_input("Feature 3", value=0.8)
                    feature4 = st.number_input("Feature 4", value=0.2)
                
                if st.button("🔮 Prédire"):
                    st.info("Interface de saisie manuelle à adapter selon vos données spécifiques")
    
    else:
        st.info("ℹ️ Sélectionnez un modèle pour commencer les prédictions.")

with tab2:
    st.header("Prédiction en Lot (Batch)")
    
    if selected_model is not None:
        st.subheader("📦 Traitement par Lots")
        
        batch_mode = st.selectbox(
            "Mode de traitement",
            ["Dossier de fichiers", "Multiple fichiers CSV"]
        )
        
        if batch_mode == "Multiple fichiers CSV":
            uploaded_files = st.file_uploader(
                "Choisir plusieurs fichiers CSV",
                type=['csv'],
                accept_multiple_files=True,
                help="Sélectionnez plusieurs fichiers pour traitement en lot"
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} fichier(s) chargé(s)")
                
                if st.button("🚀 Traiter Tous les Fichiers"):
                    batch_results = []
                    progress_bar = st.progress(0)
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        st.write(f"🔄 Traitement de {uploaded_file.name}...")
                        
                        try:
                            df_batch = pd.read_csv(uploaded_file)
                            X_batch = df_batch.values
                            
                            predictions = selected_model.predict(X_batch)
                            probabilities = selected_model.predict_proba(X_batch)
                            
                            batch_results.append({
                                'filename': uploaded_file.name,
                                'n_samples': len(predictions),
                                'predictions': predictions,
                                'avg_confidence': np.mean([max(prob) for prob in probabilities])
                            })
                            
                            progress_bar.progress((i + 1) / len(uploaded_files))
                            
                        except Exception as e:
                            st.error(f"❌ Erreur avec {uploaded_file.name}: {str(e)}")
                    
                    # Résumé des résultats
                    if batch_results:
                        st.success("✅ Traitement par lots terminé!")
                        
                        summary_df = pd.DataFrame([
                            {
                                'Fichier': r['filename'],
                                'Échantillons': r['n_samples'],
                                'Confiance Moyenne': f"{r['avg_confidence']:.2%}"
                            }
                            for r in batch_results
                        ])
                        
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Sauvegarder les résultats de batch
                        st.session_state['batch_results'] = batch_results
        
        else:
            st.info("🚧 Traitement de dossier à implémenter")
    
    else:
        st.info("ℹ️ Sélectionnez un modèle pour le traitement en lot.")

with tab3:
    st.header("Analyse des Prédictions")
    
    # Analyse des dernières prédictions
    if 'last_predictions' in st.session_state:
        last_pred = st.session_state['last_predictions']
        
        st.subheader(f"📈 Dernière Prédiction ({last_pred['model_name']})")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Échantillons", last_pred['n_samples'])
        
        with col2:
            confidence_mean = np.mean([max(prob) for prob in last_pred['probabilities']])
            st.metric("Confiance Moyenne", f"{confidence_mean:.2%}")
        
        with col3:
            timestamp = last_pred['timestamp']
            st.metric("Horodatage", timestamp.strftime("%H:%M:%S"))
        
        # Analyse de la distribution des classes
        st.subheader("🏷️ Distribution des Classes Prédites")
        
        pred_counts = pd.Series(last_pred['predictions']).value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.bar_chart(pred_counts)
        
        with col2:
            # Tableau de fréquences
            freq_df = pd.DataFrame({
                'Classe': pred_counts.index,
                'Nombre': pred_counts.values,
                'Pourcentage': [f"{(count/sum(pred_counts.values)*100):.1f}%" 
                              for count in pred_counts.values]
            })
            st.dataframe(freq_df, use_container_width=True)
        
        # Analyse de confiance par classe
        st.subheader("🎯 Confiance par Classe")
        
        confidence_by_class = {}
        for i, pred in enumerate(last_pred['predictions']):
            prob = last_pred['probabilities'][i]
            confidence = max(prob)
            
            if pred not in confidence_by_class:
                confidence_by_class[pred] = []
            confidence_by_class[pred].append(confidence)
        
        conf_stats = []
        for classe, confidences in confidence_by_class.items():
            conf_stats.append({
                'Classe': classe,
                'Confiance Moy.': np.mean(confidences),
                'Confiance Min': np.min(confidences),
                'Confiance Max': np.max(confidences),
                'Écart-type': np.std(confidences)
            })
        
        conf_df = pd.DataFrame(conf_stats)
        st.dataframe(conf_df, use_container_width=True)
        
        # Histogramme des confidences
        st.subheader("📊 Distribution des Confidences")
        
        all_confidences = [max(prob) for prob in last_pred['probabilities']]
        confidence_hist_df = pd.DataFrame({'Confiance': all_confidences})
        
        st.histogram(confidence_hist_df['Confiance'], bins=20)
        
        # Échantillons peu confiants
        st.subheader("⚠️ Échantillons à Faible Confiance")
        
        low_confidence_threshold = st.slider(
            "Seuil de confiance",
            0.0, 1.0, 0.7,
            help="Afficher les échantillons en dessous de ce seuil"
        )
        
        low_conf_indices = [i for i, conf in enumerate(all_confidences) 
                           if conf < low_confidence_threshold]
        
        if low_conf_indices:
            st.warning(f"⚠️ {len(low_conf_indices)} échantillon(s) avec confiance < {low_confidence_threshold}")
            
            low_conf_df = pd.DataFrame({
                'Index': low_conf_indices,
                'Prédiction': [last_pred['predictions'][i] for i in low_conf_indices],
                'Confiance': [all_confidences[i] for i in low_conf_indices]
            })
            
            st.dataframe(low_conf_df, use_container_width=True)
        
        else:
            st.success(f"✅ Tous les échantillons ont une confiance ≥ {low_confidence_threshold}")
    
    else:
        st.info("""
        ℹ️ **Analyse des Prédictions**
        
        Effectuez d'abord des prédictions pour voir l'analyse détaillée.
        
        L'analyse inclura:
        - Distribution des classes prédites
        - Statistiques de confiance par classe
        - Identification des échantillons peu confiants
        - Visualisations interactives
        """)
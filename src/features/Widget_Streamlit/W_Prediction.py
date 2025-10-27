"""
Widgets spécifiques à la page Prediction.
Contient les composants pour la sélection des modèles, les différents modes de prédiction 
et l'analyse des résultats de prédiction.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def create_prediction_config_sidebar():
    """
    Widget pour la configuration de prédiction dans la sidebar.
    
    Returns:
        dict: Configuration de prédiction
    """
    st.header("⚙️ Configuration")
    
    # Mode de prédiction
    prediction_mode = st.selectbox(
        "Mode de prédiction",
        ["Données simulées", "Upload fichier", "Saisie manuelle", "Batch prédiction"]
    )
    
    return {"prediction_mode": prediction_mode}


def create_simple_prediction_tab(selected_model, selected_model_name, prediction_mode):
    """
    Widget pour l'onglet prédiction simple.
    
    Args:
        selected_model: Modèle sélectionné pour la prédiction
        selected_model_name (str): Nom du modèle sélectionné
        prediction_mode (str): Mode de prédiction choisi
        
    Returns:
        dict: Résultats de prédiction ou None
    """
    st.header("Prédiction Simple")
    
    if selected_model is None:
        st.info("ℹ️ Sélectionnez un modèle pour commencer les prédictions.")
        return None
    
    st.divider()
    
    if prediction_mode == "Données simulées":
        return create_simulated_data_prediction(selected_model, selected_model_name)
    
    elif prediction_mode == "Upload fichier":
        return create_file_upload_prediction(selected_model, selected_model_name)
    
    elif prediction_mode == "Saisie manuelle":
        return create_manual_input_prediction(selected_model, selected_model_name)
    
    else:
        st.info("ℹ️ Mode de prédiction non supporté dans cet onglet.")
        return None


def create_simulated_data_prediction(selected_model, selected_model_name):
    """
    Widget pour la prédiction avec des données simulées.
    
    Args:
        selected_model: Modèle sélectionné
        selected_model_name (str): Nom du modèle
        
    Returns:
        dict: Résultats de prédiction ou None
    """
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
                
                # Faire les prédictions (simulation pour cet exemple)
                predictions = np.random.randint(0, default_classes, n_samples_pred)
                probabilities = np.random.dirichlet(np.ones(default_classes), n_samples_pred)
                
                # Affichage des résultats
                return display_prediction_results(predictions, probabilities, selected_model_name, n_samples_pred)
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
                return None
    
    return None


def create_file_upload_prediction(selected_model, selected_model_name):
    """
    Widget pour la prédiction avec upload de fichier.
    
    Args:
        selected_model: Modèle sélectionné
        selected_model_name (str): Nom du modèle
        
    Returns:
        dict: Résultats de prédiction ou None
    """
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
                    # Simulation de prédiction
                    n_samples = len(df_upload)
                    n_classes = 4  # Exemple
                    
                    predictions = np.random.randint(0, n_classes, n_samples)
                    probabilities = np.random.dirichlet(np.ones(n_classes), n_samples)
                    
                    return display_prediction_results(predictions, probabilities, selected_model_name, n_samples)
        
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")
    
    return None


def create_manual_input_prediction(selected_model, selected_model_name):
    """
    Widget pour la prédiction avec saisie manuelle.
    
    Args:
        selected_model: Modèle sélectionné
        selected_model_name (str): Nom du modèle
        
    Returns:
        dict: Résultats de prédiction ou None
    """
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
            # Simulation d'une prédiction simple
            features = [feature1, feature2, feature3, feature4]
            predictions = np.array([np.random.randint(0, 4)])  # 1 échantillon
            probabilities = np.random.dirichlet(np.ones(4), 1)
            
            return display_prediction_results(predictions, probabilities, selected_model_name, 1)
    
    return None


def display_prediction_results(predictions, probabilities, model_name, n_samples):
    """
    Widget pour afficher les résultats de prédiction.
    
    Args:
        predictions (np.array): Prédictions
        probabilities (np.array): Probabilités
        model_name (str): Nom du modèle
        n_samples (int): Nombre d'échantillons
        
    Returns:
        dict: Résultats sauvegardés
    """
    st.success("✅ Prédictions terminées!")
    
    # Tableau des résultats
    results_df = pd.DataFrame({
        'Échantillon': range(1, n_samples + 1),
        'Prédiction': predictions,
        'Probabilité Max': [max(prob) for prob in probabilities],
        'Confiance': [f"{max(prob):.2%}" for prob in probabilities]
    })
    
    # Ajouter les probabilités par classe
    for i in range(probabilities.shape[1]):
        results_df[f'Classe_{i}_Prob'] = [prob[i] for prob in probabilities]
    
    st.dataframe(results_df, use_container_width=True)
    
    # Statistiques
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Échantillons", n_samples)
    
    with col2:
        confidence_mean = np.mean([max(prob) for prob in probabilities])
        st.metric("Confiance Moy.", f"{confidence_mean:.2%}")
    
    with col3:
        pred_mode = pd.Series(predictions).mode().iloc[0] if len(predictions) > 0 else 0
        st.metric("Classe Dominante", pred_mode)
    
    # Graphiques
    create_prediction_charts(predictions, probabilities)
    
    # Sauvegarder les résultats
    results = {
        'model_name': model_name,
        'predictions': predictions,
        'probabilities': probabilities,
        'timestamp': datetime.now(),
        'n_samples': n_samples
    }
    
    st.session_state['last_predictions'] = results
    
    return results


def create_prediction_charts(predictions, probabilities):
    """
    Widget pour créer les graphiques de prédiction.
    
    Args:
        predictions (np.array): Prédictions
        probabilities (np.array): Probabilités
    """
    # Graphique de distribution des prédictions
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    
    st.subheader("📊 Distribution des Prédictions")
    st.bar_chart(pred_counts)
    
    # Graphique de confiance
    st.subheader("🎯 Distribution de la Confiance")
    confidence_scores = [max(prob) for prob in probabilities]
    confidence_df = pd.DataFrame({'Confiance': confidence_scores})
    
    # Créer un histogramme avec pandas
    hist_data = pd.cut(confidence_df['Confiance'], bins=20).value_counts().sort_index()
    st.bar_chart(hist_data)


def create_batch_prediction_tab(selected_model, selected_model_name):
    """
    Widget pour l'onglet prédiction en lot (batch).
    
    Args:
        selected_model: Modèle sélectionné
        selected_model_name (str): Nom du modèle
        
    Returns:
        list: Résultats de batch ou None
    """
    st.header("Prédiction en Lot (Batch)")
    
    if selected_model is None:
        st.info("ℹ️ Sélectionnez un modèle pour le traitement en lot.")
        return None
    
    st.subheader("📦 Traitement par Lots")
    
    batch_mode = st.selectbox(
        "Mode de traitement",
        ["Multiple fichiers CSV", "Dossier de fichiers"]
    )
    
    if batch_mode == "Multiple fichiers CSV":
        return create_multiple_files_batch_prediction(selected_model, selected_model_name)
    
    else:  # Dossier de fichiers
        st.info("🚧 Traitement de dossier à implémenter")
        return None


def create_multiple_files_batch_prediction(selected_model, selected_model_name):
    """
    Widget pour la prédiction en batch avec plusieurs fichiers.
    
    Args:
        selected_model: Modèle sélectionné
        selected_model_name (str): Nom du modèle
        
    Returns:
        list: Résultats de batch ou None
    """
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
                    # Charger le fichier
                    df = pd.read_csv(uploaded_file)
                    
                    # Simulation de prédiction
                    n_samples = len(df)
                    n_classes = 4
                    predictions = np.random.randint(0, n_classes, n_samples)
                    probabilities = np.random.dirichlet(np.ones(n_classes), n_samples)
                    
                    # Ajouter aux résultats
                    batch_results.append({
                        'file_name': uploaded_file.name,
                        'n_samples': n_samples,
                        'predictions': predictions,
                        'probabilities': probabilities,
                        'mean_confidence': np.mean([max(prob) for prob in probabilities])
                    })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                except Exception as e:
                    st.error(f"❌ Erreur avec {uploaded_file.name}: {str(e)}")
            
            # Résumé des résultats
            if batch_results:
                create_batch_results_summary(batch_results)
            
            return batch_results
    
    return None


def create_batch_results_summary(batch_results):
    """
    Widget pour afficher le résumé des résultats de batch.
    
    Args:
        batch_results (list): Résultats du traitement en lot
    """
    st.subheader("📊 Résumé du Traitement en Lot")
    
    # Tableau de résumé
    summary_data = []
    for result in batch_results:
        summary_data.append({
            'Fichier': result['file_name'],
            'Échantillons': result['n_samples'],
            'Confiance Moyenne': f"{result['mean_confidence']:.2%}",
            'Classe Dominante': pd.Series(result['predictions']).mode().iloc[0]
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Statistiques globales
    total_samples = sum([r['n_samples'] for r in batch_results])
    overall_confidence = np.mean([r['mean_confidence'] for r in batch_results])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Fichiers Traités", len(batch_results))
    with col2:
        st.metric("Total Échantillons", total_samples)
    with col3:
        st.metric("Confiance Globale", f"{overall_confidence:.2%}")


def create_prediction_analysis_tab():
    """
    Widget pour l'onglet analyse des prédictions.
    """
    st.header("Analyse des Prédictions")
    
    # Analyse des dernières prédictions
    if 'last_predictions' in st.session_state:
        last_pred = st.session_state['last_predictions']
        create_last_prediction_analysis(last_pred)
    else:
        create_analysis_info_placeholder()


def create_last_prediction_analysis(last_pred):
    """
    Widget pour analyser les dernières prédictions.
    
    Args:
        last_pred (dict): Dernières prédictions sauvegardées
    """
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
    create_class_distribution_analysis(last_pred)
    
    # Analyse de confiance par classe
    create_confidence_by_class_analysis(last_pred)
    
    # Échantillons peu confiants
    create_low_confidence_analysis(last_pred)


def create_class_distribution_analysis(last_pred):
    """
    Widget pour analyser la distribution des classes prédites.
    
    Args:
        last_pred (dict): Dernières prédictions sauvegardées
    """
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


def create_confidence_by_class_analysis(last_pred):
    """
    Widget pour analyser la confiance par classe.
    
    Args:
        last_pred (dict): Dernières prédictions sauvegardées
    """
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
    
    # Créer un histogramme avec pandas
    hist_data = pd.cut(confidence_hist_df['Confiance'], bins=20).value_counts().sort_index()
    st.bar_chart(hist_data)


def create_low_confidence_analysis(last_pred):
    """
    Widget pour analyser les échantillons à faible confiance.
    
    Args:
        last_pred (dict): Dernières prédictions sauvegardées
    """
    st.subheader("⚠️ Échantillons à Faible Confiance")
    
    low_confidence_threshold = st.slider(
        "Seuil de confiance",
        0.0, 1.0, 0.7,
        help="Afficher les échantillons en dessous de ce seuil"
    )
    
    all_confidences = [max(prob) for prob in last_pred['probabilities']]
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


def create_analysis_info_placeholder():
    """
    Widget d'information pour l'analyse quand aucune prédiction n'est disponible.
    """
    st.info("""
    ℹ️ **Analyse des Prédictions**
    
    Effectuez d'abord des prédictions pour voir l'analyse détaillée.
    
    L'analyse inclura:
    - Distribution des classes prédites
    - Statistiques de confiance par classe
    - Identification des échantillons peu confiants
    - Visualisations interactives
    """)
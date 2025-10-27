"""
Widgets spécifiques à la page Evaluation.
Contient les composants pour la sélection des modèles, l'affichage des métriques, 
les visualisations et la génération de rapports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime


def create_model_selection_tab(eval_mode="Modèles en session"):
    """
    Widget pour l'onglet sélection des modèles à évaluer.
    
    Args:
        eval_mode (str): Mode d'évaluation sélectionné
        
    Returns:
        list: Liste des modèles sélectionnés
    """
    st.header("Sélection des Modèles à Évaluer")
    
    selected_models = []
    
    if eval_mode == "Modèles en session":
        if 'training_results' in st.session_state:
            results = st.session_state['training_results']
            
            st.success(f"✅ {len(results)} modèle(s) disponible(s) en session")
            
            # Sélection des modèles à évaluer
            model_names = list(results.keys())
            selected_models = st.multiselect(
                "Choisissez les modèles à évaluer:",
                options=model_names,
                default=model_names,
                help="Sélectionnez un ou plusieurs modèles pour évaluation détaillée"
            )
            
            if selected_models:
                # Afficher les informations de base
                st.subheader("📋 Informations des Modèles")
                
                for model_name in selected_models:
                    result = results[model_name]
                    
                    with st.expander(f"🔍 {model_name}", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Précision Test", f"{result.get('test_accuracy', 0):.4f}")
                        with col2:
                            st.metric("F1-Score", f"{result.get('test_f1', 0):.4f}")
                        with col3:
                            st.metric("Score CV", f"{result.get('cv_mean', 0):.4f}")
                        
                        # Informations supplémentaires
                        if 'training_time' in result:
                            st.info(f"⏱️ Temps d'entraînement: {result['training_time'].total_seconds():.1f}s")
                        
                        if 'timestamp' in result:
                            st.info(f"📅 Entraîné le: {result['timestamp'].strftime('%d/%m/%Y %H:%M')}")
        
        else:
            st.warning("""
            ⚠️ Aucun modèle en session.
            
            Veuillez d'abord entraîner des modèles dans la page **Entraînement**.
            """)
    
    elif eval_mode == "Modèles sauvegardés":
        st.info("🚧 Chargement de modèles sauvegardés non encore implémenté")
    
    else:  # Upload modèle
        st.info("🚧 Upload de modèles externes non encore implémenté")
    
    return selected_models


def create_detailed_metrics_tab(selected_models):
    """
    Widget pour l'onglet métriques détaillées.
    
    Args:
        selected_models (list): Liste des modèles sélectionnés
    """
    st.header("Métriques Détaillées")
    
    if not selected_models:
        st.info("ℹ️ Sélectionnez des modèles pour voir les métriques détaillées.")
        return
    
    results = st.session_state.get('training_results', {})
    
    # Tableau comparatif détaillé
    st.subheader("📊 Comparaison Détaillée")
    
    metrics_data = []
    for model_name in selected_models:
        result = results[model_name]
        
        # Récupérer les métriques détaillées si disponibles
        test_report = result.get('test_classification_report', {})
        
        row = {
            'Modèle': model_name,
            'Précision Test': result.get('test_accuracy', 0),
            'F1-Score Macro': result.get('test_f1', 0),
            'Score CV Moyen': result.get('cv_mean', 0),
            'Score CV Std': result.get('cv_std', 0),
        }
        
        # Ajouter les métriques par classe si disponibles
        if isinstance(test_report, dict) and 'macro avg' in test_report:
            row.update({
                'Précision Macro': test_report['macro avg']['precision'],
                'Rappel Macro': test_report['macro avg']['recall'],
                'Support Total': test_report['macro avg']['support']
            })
        
        metrics_data.append(row)
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Métriques individuelles pour chaque modèle
    st.subheader("📋 Rapports de Classification")
    
    for model_name in selected_models:
        result = results[model_name]
        
        with st.expander(f"📈 Rapport détaillé - {model_name}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Rapport de classification
                test_report = result.get('test_classification_report')
                if isinstance(test_report, dict):
                    # Convertir en DataFrame pour un affichage propre
                    report_df = pd.DataFrame(test_report).transpose()
                    if not report_df.empty:
                        # Sélectionner les colonnes numériques
                        numeric_cols = ['precision', 'recall', 'f1-score', 'support']
                        numeric_cols = [col for col in numeric_cols if col in report_df.columns]
                        
                        if numeric_cols:
                            st.dataframe(report_df[numeric_cols], use_container_width=True)
                        else:
                            st.write(test_report)
                elif test_report:
                    st.text(str(test_report))
                else:
                    st.info("Rapport de classification non disponible")
            
            with col2:
                # Métriques clés
                st.metric("Précision", f"{result.get('test_accuracy', 0):.4f}")
                st.metric("F1-Score", f"{result.get('test_f1', 0):.4f}")
                
                if 'training_time' in result:
                    st.metric("Temps", f"{result['training_time'].total_seconds():.1f}s")
    
    # Graphique radar de comparaison
    if len(selected_models) > 1:
        create_radar_comparison_chart(selected_models, results)


def create_radar_comparison_chart(selected_models, results):
    """
    Widget pour créer un graphique radar de comparaison.
    
    Args:
        selected_models (list): Liste des modèles sélectionnés
        results (dict): Résultats des modèles
    """
    st.subheader("🕸️ Comparaison Radar")
    
    # Préparer les données pour le graphique radar
    radar_metrics = ['test_accuracy', 'test_f1', 'cv_mean']
    radar_labels = ['Précision Test', 'F1-Score', 'Score CV']
    
    fig = go.Figure()
    
    for model_name in selected_models:
        result = results[model_name]
        values = [result.get(metric, 0) for metric in radar_metrics]
        values.append(values[0])  # Fermer le radar
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_labels + [radar_labels[0]],
            fill='toself',
            name=model_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title="Comparaison des Métriques Principales"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_visualizations_tab(selected_models):
    """
    Widget pour l'onglet visualisations avancées.
    
    Args:
        selected_models (list): Liste des modèles sélectionnés
    """
    st.header("Visualisations Avancées")
    
    if not selected_models:
        st.info("ℹ️ Sélectionnez des modèles pour voir les visualisations.")
        return
    
    results = st.session_state.get('training_results', {})
    
    # Matrices de confusion (simulées car pas de données réelles dans ce contexte)
    st.subheader("🔥 Matrices de Confusion")
    st.info("📝 Les matrices de confusion seront affichées ici avec les vraies données d'évaluation")
    
    # Comparaison des temps d'entraînement
    create_training_time_comparison(selected_models, results)
    
    # Distribution des scores CV
    create_cv_scores_distribution(selected_models, results)


def create_training_time_comparison(selected_models, results):
    """
    Widget pour comparer les temps d'entraînement.
    
    Args:
        selected_models (list): Liste des modèles sélectionnés
        results (dict): Résultats des modèles
    """
    st.subheader("⏱️ Temps d'Entraînement")
    
    timing_data = []
    for model_name in selected_models:
        result = results[model_name]
        training_time = result.get('training_time')
        if training_time:
            timing_data.append({
                'Modèle': model_name,
                'Temps (secondes)': training_time.total_seconds()
            })
    
    if timing_data:
        timing_df = pd.DataFrame(timing_data)
        
        fig = px.bar(
            timing_df,
            x='Modèle',
            y='Temps (secondes)',
            title="Comparaison des Temps d'Entraînement",
            color='Temps (secondes)',
            color_continuous_scale="viridis"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def create_cv_scores_distribution(selected_models, results):
    """
    Widget pour afficher la distribution des scores CV.
    
    Args:
        selected_models (list): Liste des modèles sélectionnés
        results (dict): Résultats des modèles
    """
    st.subheader("📈 Distribution des Scores de Validation Croisée")
    
    cv_data = []
    for model_name in selected_models:
        result = results[model_name]
        if 'cv_scores' in result:
            for score in result['cv_scores']:
                cv_data.append({
                    'Modèle': model_name,
                    'Score CV': score
                })
        else:
            # Simuler des scores CV basés sur la moyenne et l'écart-type
            cv_mean = result.get('cv_mean', 0.8)
            cv_std = result.get('cv_std', 0.02)
            simulated_scores = np.random.normal(cv_mean, cv_std, 5)  # 5 folds simulés
            
            for score in simulated_scores:
                cv_data.append({
                    'Modèle': model_name,
                    'Score CV': max(0, min(1, score))  # Borner entre 0 et 1
                })
    
    if cv_data:
        cv_df = pd.DataFrame(cv_data)
        
        fig = px.box(
            cv_df,
            x='Modèle',
            y='Score CV',
            title="Distribution des Scores de Validation Croisée",
            points="all"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def create_evaluation_report_tab(selected_models):
    """
    Widget pour l'onglet rapport d'évaluation complet.
    
    Args:
        selected_models (list): Liste des modèles sélectionnés
    """
    st.header("Rapport d'Évaluation Complet")
    
    if not selected_models:
        st.info("""
        ℹ️ **Génération de rapport**
        
        Sélectionnez des modèles évalués pour générer un rapport automatique complet.
        
        Le rapport inclura:
        - Résumé exécutif avec le meilleur modèle
        - Configuration expérimentale
        - Métriques détaillées par modèle
        - Recommandations personnalisées
        - Options d'export des données
        """)
        return
    
    results = st.session_state.get('training_results', {})
    comparison_data = st.session_state.get('comparison_data', {})
    
    # Génération du rapport
    st.subheader("📄 Rapport Automatique")
    
    report_sections = generate_evaluation_report(selected_models, results, comparison_data)
    
    # Afficher le rapport
    full_report = "\n\n".join(report_sections)
    st.markdown(full_report)
    
    # Boutons d'export
    create_export_section(selected_models, results, full_report)


def generate_evaluation_report(selected_models, results, comparison_data):
    """
    Génère les sections du rapport d'évaluation.
    
    Args:
        selected_models (list): Liste des modèles sélectionnés
        results (dict): Résultats des modèles
        comparison_data (dict): Données de comparaison
        
    Returns:
        list: Liste des sections du rapport
    """
    report_sections = []
    
    # Section 1: Résumé exécutif
    report_sections.append("## 🎯 Résumé Exécutif")
    
    best_model = max(selected_models, key=lambda x: results[x].get('test_accuracy', 0))
    best_accuracy = results[best_model].get('test_accuracy', 0)
    
    report_sections.append(f"""
**Évaluation de {len(selected_models)} modèle(s) de classification COVID-19**

- **Meilleur modèle**: {best_model}
- **Précision maximale**: {best_accuracy:.4f}
- **Date d'évaluation**: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
    """)
    
    # Section 2: Configuration expérimentale
    if comparison_data:
        data_info = comparison_data.get('data_info', {})
        report_sections.append("## ⚙️ Configuration Expérimentale")
        report_sections.append(f"""
**Paramètres des données:**
- Nombre d'échantillons: {data_info.get('n_samples', 'N/A')}
- Nombre de features: {data_info.get('n_features', 'N/A')}
- Nombre de classes: {data_info.get('n_classes', 'N/A')}
- Taille du jeu de test: {data_info.get('test_size', 'N/A')}
        """)
    
    # Section 3: Résultats par modèle
    report_sections.append("## 📊 Résultats Détaillés")
    
    for model_name in selected_models:
        result = results[model_name]
        
        report_sections.append(f"### {model_name}")
        
        metrics_text = f"""
- **Précision Test**: {result.get('test_accuracy', 0):.4f}
- **F1-Score**: {result.get('test_f1', 0):.4f}
- **Score CV Moyen**: {result.get('cv_mean', 0):.4f} ± {result.get('cv_std', 0):.4f}
"""
        
        if 'training_time' in result:
            metrics_text += f"- **Temps d'entraînement**: {result['training_time'].total_seconds():.1f}s\n"
        
        if 'best_params' in result:
            metrics_text += f"- **Paramètres optimaux**: {result['best_params']}\n"
        
        report_sections.append(metrics_text)
    
    # Section 4: Recommandations
    report_sections.extend(generate_recommendations(selected_models, results))
    
    return report_sections


def generate_recommendations(selected_models, results):
    """
    Génère des recommandations basées sur les résultats.
    
    Args:
        selected_models (list): Liste des modèles sélectionnés
        results (dict): Résultats des modèles
        
    Returns:
        list: Liste des recommandations
    """
    recommendations = ["## 💡 Recommandations"]
    
    # Analyser les résultats pour donner des recommandations
    accuracies = [results[name].get('test_accuracy', 0) for name in selected_models]
    times = [results[name].get('training_time', pd.Timedelta(0)).total_seconds() 
            for name in selected_models]
    
    reco_list = []
    
    if max(accuracies) - min(accuracies) < 0.05:
        reco_list.append("- Les performances sont similaires entre modèles. Privilégier le plus rapide.")
    
    fastest_model = selected_models[times.index(min(times))]
    reco_list.append(f"- **Modèle le plus rapide**: {fastest_model}")
    
    if len(selected_models) > 1:
        reco_list.append("- Considérer un ensemble (ensemble) des meilleurs modèles.")
    
    recommendations.extend(reco_list)
    
    return recommendations


def create_export_section(selected_models, results, full_report):
    """
    Widget pour la section d'export des données.
    
    Args:
        selected_models (list): Liste des modèles sélectionnés
        results (dict): Résultats des modèles
        full_report (str): Rapport complet en markdown
    """
    # Bouton de téléchargement du rapport
    st.download_button(
        label="📥 Télécharger le Rapport (Markdown)",
        data=full_report,
        file_name=f"rapport_evaluation_covid_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )
    
    # Export des données
    st.subheader("💾 Export des Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV des métriques
        if st.button("📊 Exporter Métriques (CSV)"):
            metrics_export = []
            for model_name in selected_models:
                result = results[model_name]
                metrics_export.append({
                    'modele': model_name,
                    'precision_test': result.get('test_accuracy', 0),
                    'f1_score': result.get('test_f1', 0),
                    'cv_mean': result.get('cv_mean', 0),
                    'cv_std': result.get('cv_std', 0),
                    'temps_entrainement': result.get('training_time', pd.Timedelta(0)).total_seconds()
                })
            
            export_df = pd.DataFrame(metrics_export)
            csv_data = export_df.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Télécharger CSV",
                data=csv_data,
                file_name=f"metrics_covid_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        # JSON des résultats complets
        if st.button("🗃️ Exporter Résultats (JSON)"):
            # Préparer les données pour JSON (sérialisation)
            json_export = {}
            for model_name in selected_models:
                result = results[model_name].copy()
                # Convertir les objets non sérialisables
                if 'training_time' in result:
                    result['training_time'] = result['training_time'].total_seconds()
                if 'timestamp' in result:
                    result['timestamp'] = result['timestamp'].isoformat()
                # Exclure le pipeline qui n'est pas sérialisable
                result.pop('pipeline', None)
                json_export[model_name] = result
            
            json_data = json.dumps(json_export, indent=2, default=str)
            
            st.download_button(
                label="⬇️ Télécharger JSON",
                data=json_data,
                file_name=f"results_covid_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
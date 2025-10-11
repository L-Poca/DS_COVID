import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Calcul correct du project_root
current_file = Path(__file__)  # pages/02_Model/2_Evaluation.py
current_dir = current_file.parent  # pages/02_Model/
streamlit_dir = current_dir.parent.parent  # streamlit/
src_dir = streamlit_dir.parent  # src/
project_root = src_dir.parent  # DS_COVID/

sys.path.append(str(project_root))

try:
    from src.features.Pipelines.Pipeline_Sklearn import PipelineManager
except ImportError:
    st.error("❌ Impossible d'importer PipelineManager. Vérifiez le chemin d'accès.")

st.title("📊 Évaluation des Modèles")

st.markdown("""
Cette page permet d'évaluer en détail les modèles entraînés et de comparer leurs performances
avec différentes métriques et visualisations.
""")

# Configuration sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Mode d'évaluation
    eval_mode = st.selectbox(
        "Mode d'évaluation",
        ["Modèles en session", "Modèles sauvegardés", "Upload modèle"]
    )
    
    # Options d'affichage
    st.subheader("📈 Visualisations")
    show_confusion_matrix = st.checkbox("Matrice de confusion", True)
    show_roc_curves = st.checkbox("Courbes ROC", True)
    show_precision_recall = st.checkbox("Précision-Rappel", True)
    show_feature_importance = st.checkbox("Importance des features", False)

# Interface principale
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Sélection Modèles", "📊 Métriques", "📈 Visualisations", "📋 Rapport"])

with tab1:
    st.header("Sélection des Modèles à Évaluer")
    
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
                            st.metric("F1-Score", f"{result.get('test_f1', 0):.4f}")
                        
                        with col2:
                            st.metric("Score CV", f"{result.get('cv_mean', 0):.4f}")
                            st.metric("Écart-type CV", f"{result.get('cv_std', 0):.4f}")
                        
                        with col3:
                            training_time = result.get('training_time')
                            if training_time:
                                st.metric("Temps entraînement", f"{training_time.total_seconds():.1f}s")
                        
                        # Paramètres du modèle
                        if 'best_params' in result:
                            st.write("🔧 **Meilleurs paramètres:**")
                            st.json(result['best_params'])
        else:
            st.warning("""
            ⚠️ Aucun modèle en session.
            
            Veuillez d'abord entraîner des modèles dans la page **Entraînement**.
            """)
            selected_models = []
    
    elif eval_mode == "Modèles sauvegardés":
        st.info("🚧 Chargement de modèles sauvegardés non encore implémenté")
        selected_models = []
    
    else:  # Upload modèle
        st.info("🚧 Upload de modèles externes non encore implémenté")
        selected_models = []

with tab2:
    st.header("Métriques Détaillées")
    
    if eval_mode == "Modèles en session" and 'training_results' in st.session_state and selected_models:
        results = st.session_state['training_results']
        
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
                        # Convertir en DataFrame pour affichage
                        report_df = pd.DataFrame(test_report).transpose()
                        
                        # Nettoyer et formater
                        if 'support' in report_df.columns:
                            report_df['support'] = report_df['support'].fillna(0).astype(int)
                        
                        # Arrondir les valeurs numériques
                        numeric_cols = ['precision', 'recall', 'f1-score']
                        for col in numeric_cols:
                            if col in report_df.columns:
                                report_df[col] = report_df[col].round(4)
                        
                        st.dataframe(report_df, use_container_width=True)
                    
                    elif isinstance(test_report, str):
                        st.code(test_report)
                
                with col2:
                    # Métriques principales en cards
                    st.metric("🎯 Précision", f"{result.get('test_accuracy', 0):.4f}")
                    st.metric("📊 F1-Score", f"{result.get('test_f1', 0):.4f}")
                    
                    if 'best_score' in result:
                        st.metric("🏆 Best Grid Score", f"{result['best_score']:.4f}")
        
        # Graphique radar de comparaison
        if len(selected_models) > 1:
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
    
    else:
        st.info("ℹ️ Sélectionnez des modèles pour voir les métriques détaillées.")

with tab3:
    st.header("Visualisations Avancées")
    
    if eval_mode == "Modèles en session" and 'training_results' in st.session_state and selected_models:
        results = st.session_state['training_results']
        
        # Matrices de confusion
        if show_confusion_matrix:
            st.subheader("🔥 Matrices de Confusion")
            
            # Organiser en colonnes selon le nombre de modèles
            if len(selected_models) == 1:
                cols = [st.container()]
            elif len(selected_models) == 2:
                cols = st.columns(2)
            else:
                cols = st.columns(min(3, len(selected_models)))
            
            for i, model_name in enumerate(selected_models):
                result = results[model_name]
                
                with cols[i % len(cols)]:
                    st.write(f"**{model_name}**")
                    
                    if 'confusion_matrix' in result:
                        cm = result['confusion_matrix']
                        
                        # Créer une heatmap interactive avec Plotly
                        fig = px.imshow(
                            cm,
                            labels=dict(x="Prédiction", y="Réalité", color="Nombre"),
                            x=[f"Classe {i}" for i in range(cm.shape[1])],
                            y=[f"Classe {i}" for i in range(cm.shape[0])],
                            color_continuous_scale="Blues",
                            title=f"Matrice - {model_name}"
                        )
                        
                        # Ajouter les valeurs dans les cellules
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                fig.add_annotation(
                                    x=j, y=i,
                                    text=str(cm[i, j]),
                                    showarrow=False,
                                    font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
                                )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Matrice de confusion non disponible")
        
        # Comparaison des temps d'entraînement
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
        
        # Distribution des scores CV
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
    
    else:
        st.info("ℹ️ Sélectionnez des modèles pour voir les visualisations.")

with tab4:
    st.header("Rapport d'Évaluation Complet")
    
    if eval_mode == "Modèles en session" and 'training_results' in st.session_state and selected_models:
        results = st.session_state['training_results']
        comparison_data = st.session_state.get('comparison_data', {})
        
        # Génération du rapport
        st.subheader("📄 Rapport Automatique")
        
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
        report_sections.append("## 💡 Recommandations")
        
        # Analyser les résultats pour donner des recommandations
        accuracies = [results[name].get('test_accuracy', 0) for name in selected_models]
        times = [results[name].get('training_time', pd.Timedelta(0)).total_seconds() 
                for name in selected_models]
        
        recommendations = []
        
        if max(accuracies) - min(accuracies) < 0.05:
            recommendations.append("- Les performances sont similaires entre modèles. Privilégier le plus rapide.")
        
        fastest_model = selected_models[times.index(min(times))]
        recommendations.append(f"- **Modèle le plus rapide**: {fastest_model}")
        
        if len(selected_models) > 1:
            recommendations.append("- Considérer un ensemble (ensemble) des meilleurs modèles.")
        
        report_sections.extend(recommendations)
        
        # Afficher le rapport
        full_report = "\n\n".join(report_sections)
        st.markdown(full_report)
        
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
                        'model': model_name,
                        'test_accuracy': result.get('test_accuracy', 0),
                        'test_f1': result.get('test_f1', 0),
                        'cv_mean': result.get('cv_mean', 0),
                        'cv_std': result.get('cv_std', 0),
                        'training_time_s': result.get('training_time', pd.Timedelta(0)).total_seconds()
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
                    
                    # Convertir les objets non-sérialisables
                    if 'training_time' in result:
                        result['training_time'] = result['training_time'].total_seconds()
                    
                    if 'confusion_matrix' in result:
                        result['confusion_matrix'] = result['confusion_matrix'].tolist()
                    
                    json_export[model_name] = result
                
                import json
                json_data = json.dumps(json_export, indent=2, default=str)
                
                st.download_button(
                    label="⬇️ Télécharger JSON",
                    data=json_data,
                    file_name=f"results_covid_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    else:
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



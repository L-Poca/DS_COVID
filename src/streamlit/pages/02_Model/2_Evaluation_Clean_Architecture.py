"""
Page d'évaluation refactorisée avec Clean Architecture
Permet d'évaluer les modèles entraînés avec différentes métriques
"""

import streamlit as st
import sys
import os
from pathlib import Path
from typing import Optional, Dict, List, Any
import traceback

# Ajout du chemin vers les modules
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.features.Widget_Streamlit.core.di_container import AppContainerFactory, ComponentFactory
    from src.features.Widget_Streamlit.core.entities.training_config import EvaluationConfig
    from src.features.Widget_Streamlit.core.entities.model_result import EvaluationResult
    from src.features.Widget_Streamlit.core.interfaces.i_state_manager import StateScope
    from src.features.Widget_Streamlit.core.services.evaluation_service import EvaluationService
    from src.features.Widget_Streamlit.core.state.streamlit_state_manager import StreamlitStateManager
    from src.features.Widget_Streamlit.core.components.forms import EvaluationForm
except ImportError as e:
    st.error(f"Erreur d'import des modules Clean Architecture: {e}")
    st.error("Vérifiez que tous les modules de l'architecture sont correctement installés")
    st.stop()


class EvaluationPageController:
    """
    Contrôleur pour la page d'évaluation utilisant la Clean Architecture.
    
    Responsabilités:
    - Orchestrer l'interaction entre UI et services métier
    - Gérer l'état de la page d'évaluation
    - Coordonner l'affichage des résultats d'évaluation
    """
    
    def __init__(self):
        """Initialise le contrôleur avec injection automatique des dépendances."""
        try:
            # Injection automatique des dépendances
            self._container = AppContainerFactory.create_container()
            self._evaluation_service = self._container.get_evaluation_service()
            self._state_manager = self._container.get_state_manager()
            
            # Création des composants UI
            self._evaluation_form = ComponentFactory.create_evaluation_form(
                self._state_manager,
                self._container.get_sklearn_pipeline_manager(),
                self._container.get_tensorflow_pipeline_manager()
            )
            
        except Exception as e:
            st.error(f"Erreur d'initialisation du contrôleur d'évaluation: {e}")
            st.error("Détails:", str(traceback.format_exc()))
            raise
    
    def render(self):
        """Point d'entrée principal pour le rendu de la page d'évaluation."""
        try:
            st.title("🔍 Évaluation des Modèles")
            st.markdown("---")
            
            # Vérification des prérequis
            if not self._check_prerequisites():
                return
            
            # Configuration d'évaluation
            evaluation_config = self._render_evaluation_configuration()
            if evaluation_config is None:
                return
            
            # Évaluation du modèle
            if st.button("🚀 Lancer l'Évaluation", type="primary", use_container_width=True):
                self._handle_start_evaluation(evaluation_config)
            
            # Affichage des résultats
            self._render_evaluation_results()
            
        except Exception as e:
            st.error(f"Erreur lors du rendu de la page d'évaluation: {e}")
            st.error("Détails:", str(traceback.format_exc()))
    
    def _check_prerequisites(self) -> bool:
        """
        Vérifie que les prérequis pour l'évaluation sont remplis.
        
        Returns:
            bool: True si les prérequis sont OK, False sinon
        """
        try:
            # Vérification des données d'entraînement
            training_results = self._state_manager.get_state(
                "training_results", 
                scope=StateScope.SESSION
            )
            
            if not training_results:
                st.warning("⚠️ Aucun modèle entraîné trouvé")
                st.info("Veuillez d'abord entraîner un modèle dans la page Training")
                return False
            
            # Vérification de la disponibilité des données
            data_config = self._state_manager.get_state(
                "current_data_config",
                scope=StateScope.SESSION
            )
            
            if not data_config:
                st.warning("⚠️ Configuration des données manquante")
                st.info("Veuillez d'abord configurer les données dans la page Training")
                return False
            
            # Affichage des informations sur le modèle disponible
            with st.expander("📊 Informations sur le modèle entraîné", expanded=False):
                st.json({
                    "Pipeline Type": training_results.get("pipeline_type", "Unknown"),
                    "Pipeline Name": training_results.get("pipeline_name", "Unknown"),
                    "Training Score": training_results.get("training_score", "N/A"),
                    "Training Time": training_results.get("training_time", "N/A")
                })
            
            return True
            
        except Exception as e:
            st.error(f"Erreur lors de la vérification des prérequis: {e}")
            return False
    
    def _render_evaluation_configuration(self) -> Optional[EvaluationConfig]:
        """
        Affiche et gère le formulaire de configuration d'évaluation.
        
        Returns:
            Optional[EvaluationConfig]: Configuration d'évaluation si valide, None sinon
        """
        try:
            st.subheader("⚙️ Configuration de l'Évaluation")
            
            # Récupération de la configuration actuelle
            current_config = self._state_manager.get_state(
                "current_evaluation_config",
                scope=StateScope.WIDGET,
                default_value=None
            )
            
            # Rendu du formulaire d'évaluation
            evaluation_config = self._evaluation_form.render_evaluation_config_form(
                current_config=current_config,
                key_prefix="eval"
            )
            
            # Sauvegarde de la configuration
            if evaluation_config:
                self._state_manager.set_state(
                    "current_evaluation_config",
                    evaluation_config,
                    scope=StateScope.WIDGET
                )
            
            return evaluation_config
            
        except Exception as e:
            st.error(f"Erreur lors du rendu de la configuration d'évaluation: {e}")
            return None
    
    def _handle_start_evaluation(self, config: EvaluationConfig):
        """
        Gère le processus d'évaluation du modèle.
        
        Args:
            config: Configuration d'évaluation
        """
        try:
            # Indicateur de progression
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            with progress_placeholder.container():
                progress_bar = st.progress(0)
                progress_bar.progress(10, "Initialisation de l'évaluation...")
            
            with status_placeholder.container():
                st.info("🔄 Préparation des données d'évaluation...")
            
            # Préparation des données d'évaluation
            progress_bar.progress(30, "Chargement des données...")
            prepared_data = self._evaluation_service.prepare_evaluation_data(config)
            
            if not prepared_data.is_successful:
                st.error(f"❌ Erreur lors de la préparation des données: {prepared_data.error_message}")
                return
            
            # Évaluation du modèle
            progress_bar.progress(60, "Évaluation en cours...")
            status_placeholder.info("🧮 Calcul des métriques d'évaluation...")
            
            evaluation_result = self._evaluation_service.evaluate_model(config, prepared_data)
            
            if not evaluation_result.is_successful:
                st.error(f"❌ Erreur lors de l'évaluation: {evaluation_result.error_message}")
                return
            
            # Finalisation
            progress_bar.progress(100, "Évaluation terminée!")
            status_placeholder.success("✅ Évaluation terminée avec succès!")
            
            # Sauvegarde des résultats
            self._state_manager.set_state(
                "evaluation_results",
                evaluation_result,
                scope=StateScope.SESSION
            )
            
            # Nettoyage de l'interface
            progress_placeholder.empty()
            status_placeholder.empty()
            
            # Message de succès
            st.success("🎉 Évaluation terminée avec succès!")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'évaluation: {e}")
            st.error("Détails:", str(traceback.format_exc()))
    
    def _render_evaluation_results(self):
        """Affiche les résultats d'évaluation."""
        try:
            # Récupération des résultats
            evaluation_results = self._state_manager.get_state(
                "evaluation_results",
                scope=StateScope.SESSION
            )
            
            if not evaluation_results:
                return
            
            st.subheader("📊 Résultats de l'Évaluation")
            
            # Métriques principales
            if hasattr(evaluation_results, 'metrics') and evaluation_results.metrics:
                self._render_main_metrics(evaluation_results.metrics)
            
            # Matrice de confusion
            if hasattr(evaluation_results, 'confusion_matrix') and evaluation_results.confusion_matrix is not None:
                self._render_confusion_matrix(evaluation_results.confusion_matrix)
            
            # Rapport de classification
            if hasattr(evaluation_results, 'classification_report') and evaluation_results.classification_report:
                self._render_classification_report(evaluation_results.classification_report)
            
            # Courbes ROC/AUC
            if hasattr(evaluation_results, 'roc_data') and evaluation_results.roc_data:
                self._render_roc_curves(evaluation_results.roc_data)
            
            # Données brutes (optionnel)
            with st.expander("🔍 Données brutes d'évaluation", expanded=False):
                st.json(evaluation_results.__dict__ if hasattr(evaluation_results, '__dict__') else str(evaluation_results))
                
        except Exception as e:
            st.error(f"Erreur lors de l'affichage des résultats: {e}")
    
    def _render_main_metrics(self, metrics: Dict[str, Any]):
        """Affiche les métriques principales."""
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                accuracy = metrics.get('accuracy', 0)
                st.metric(
                    label="🎯 Accuracy",
                    value=f"{accuracy:.3f}",
                    delta=f"{accuracy*100:.1f}%"
                )
            
            with col2:
                precision = metrics.get('precision', 0)
                st.metric(
                    label="🔍 Précision",
                    value=f"{precision:.3f}",
                    delta=f"{precision*100:.1f}%"
                )
            
            with col3:
                recall = metrics.get('recall', 0)
                st.metric(
                    label="📈 Rappel",
                    value=f"{recall:.3f}",
                    delta=f"{recall*100:.1f}%"
                )
            
            with col4:
                f1_score = metrics.get('f1_score', 0)
                st.metric(
                    label="⚖️ F1-Score",
                    value=f"{f1_score:.3f}",
                    delta=f"{f1_score*100:.1f}%"
                )
        except Exception as e:
            st.error(f"Erreur affichage métriques: {e}")
    
    def _render_confusion_matrix(self, confusion_matrix):
        """Affiche la matrice de confusion."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            st.subheader("🔄 Matrice de Confusion")
            
            # Création du graphique
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Labels pour COVID-19
            labels = ['Normal', 'COVID-19', 'Lung Opacity', 'Viral Pneumonia']
            
            # Heatmap
            sns.heatmap(
                confusion_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=labels[:confusion_matrix.shape[1]],
                yticklabels=labels[:confusion_matrix.shape[0]],
                ax=ax
            )
            
            ax.set_title('Matrice de Confusion')
            ax.set_xlabel('Prédictions')
            ax.set_ylabel('Valeurs Réelles')
            
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"Erreur affichage matrice de confusion: {e}")
            # Affichage alternatif
            st.write("Matrice de confusion (valeurs brutes):")
            st.write(confusion_matrix)
    
    def _render_classification_report(self, classification_report: Dict[str, Any]):
        """Affiche le rapport de classification."""
        try:
            st.subheader("📋 Rapport de Classification")
            
            # Conversion en DataFrame pour un meilleur affichage
            import pandas as pd
            
            df_report = pd.DataFrame(classification_report).transpose()
            st.dataframe(
                df_report.style.format("{:.3f}"),
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Erreur affichage rapport de classification: {e}")
            st.json(classification_report)
    
    def _render_roc_curves(self, roc_data: Dict[str, Any]):
        """Affiche les courbes ROC."""
        try:
            import matplotlib.pyplot as plt
            
            st.subheader("📈 Courbes ROC")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Affichage des courbes pour chaque classe
            for class_name, data in roc_data.items():
                if 'fpr' in data and 'tpr' in data and 'auc' in data:
                    ax.plot(
                        data['fpr'], 
                        data['tpr'], 
                        label=f"{class_name} (AUC = {data['auc']:.3f})",
                        linewidth=2
                    )
            
            # Ligne de référence
            ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
            
            ax.set_xlabel('Taux de Faux Positifs')
            ax.set_ylabel('Taux de Vrais Positifs')
            ax.set_title('Courbes ROC par Classe')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"Erreur affichage courbes ROC: {e}")
            st.json(roc_data)


def main():
    """
    Point d'entrée principal de la page d'évaluation.
    
    Utilise le pattern Controller pour orchestrer l'interaction
    entre les composants de l'architecture Clean.
    """
    try:
        # Configuration de la page Streamlit
        st.set_page_config(
            page_title="Évaluation - COVID Detection",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Création et rendu du contrôleur
        controller = EvaluationPageController()
        controller.render()
        
    except Exception as e:
        st.error("❌ Erreur critique dans la page d'évaluation")
        st.error(f"Détails: {e}")
        st.error("Traceback complet:")
        st.code(traceback.format_exc())
        
        # Fallback vers le widget original
        st.warning("🔄 Tentative de basculement vers l'ancien widget...")
        try:
            from src.features.Widget_Streamlit.W_Evaluation import render_evaluation_widget
            render_evaluation_widget()
        except Exception as fallback_error:
            st.error(f"❌ Erreur aussi dans le fallback: {fallback_error}")


if __name__ == "__main__":
    main()
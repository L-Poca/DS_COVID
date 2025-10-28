"""
Page de prédiction refactorisée avec Clean Architecture
Permet de faire des prédictions sur de nouvelles images avec les modèles entraînés
"""

import streamlit as st
import sys
import os
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import traceback
from PIL import Image
import numpy as np

# Ajout du chemin vers les modules
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.features.Widget_Streamlit.core.di_container import AppContainerFactory, ComponentFactory
    from src.features.Widget_Streamlit.core.entities.model_result import PredictionResult
    from src.features.Widget_Streamlit.core.interfaces.i_state_manager import StateScope
    from src.features.Widget_Streamlit.core.services.prediction_service import PredictionService
    from src.features.Widget_Streamlit.core.state.streamlit_state_manager import StreamlitStateManager
    from src.features.Widget_Streamlit.core.components.forms import PredictionForm
except ImportError as e:
    st.error(f"Erreur d'import des modules Clean Architecture: {e}")
    st.error("Vérifiez que tous les modules de l'architecture sont correctement installés")
    st.stop()


class PredictionPageController:
    """
    Contrôleur pour la page de prédiction utilisant la Clean Architecture.
    
    Responsabilités:
    - Orchestrer l'interaction entre UI et services métier
    - Gérer l'état de la page de prédiction
    - Coordonner l'affichage des résultats de prédiction
    """
    
    def __init__(self):
        """Initialise le contrôleur avec injection automatique des dépendances."""
        try:
            # Injection automatique des dépendances
            self._container = AppContainerFactory.create_container()
            self._prediction_service = self._container.get_prediction_service()
            self._state_manager = self._container.get_state_manager()
            
            # Création des composants UI
            self._prediction_form = ComponentFactory.create_prediction_form(
                self._state_manager
            )
            
        except Exception as e:
            st.error(f"Erreur d'initialisation du contrôleur de prédiction: {e}")
            st.error("Détails:", str(traceback.format_exc()))
            raise
    
    def render(self):
        """Point d'entrée principal pour le rendu de la page de prédiction."""
        try:
            st.title("🔮 Prédiction COVID-19")
            st.markdown("---")
            
            # Vérification des prérequis
            if not self._check_prerequisites():
                return
            
            # Interface de prédiction
            col1, col2 = st.columns([2, 1])
            
            with col1:
                self._render_prediction_interface()
            
            with col2:
                self._render_model_info()
            
            # Affichage des résultats
            self._render_prediction_results()
            
        except Exception as e:
            st.error(f"Erreur lors du rendu de la page de prédiction: {e}")
            st.error("Détails:", str(traceback.format_exc()))
    
    def _check_prerequisites(self) -> bool:
        """
        Vérifie que les prérequis pour la prédiction sont remplis.
        
        Returns:
            bool: True si les prérequis sont OK, False sinon
        """
        try:
            # Vérification des modèles entraînés
            training_results = self._state_manager.get_state(
                "training_results", 
                scope=StateScope.SESSION
            )
            
            if not training_results:
                st.warning("⚠️ Aucun modèle entraîné trouvé")
                st.info("Veuillez d'abord entraîner un modèle dans la page Training")
                
                # Bouton pour rediriger vers la page d'entraînement
                if st.button("🚀 Aller à la page Training", type="primary"):
                    st.switch_page("pages/02_Model/1_Training_Clean_Architecture.py")
                
                return False
            
            return True
            
        except Exception as e:
            st.error(f"Erreur lors de la vérification des prérequis: {e}")
            return False
    
    def _render_prediction_interface(self):
        """Affiche l'interface de prédiction."""
        try:
            st.subheader("📤 Upload d'Image")
            
            # Configuration de prédiction
            prediction_config = self._prediction_form.render_prediction_config_form()
            
            # Upload de fichier
            uploaded_files = st.file_uploader(
                "Sélectionnez les images à analyser",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                accept_multiple_files=True,
                help="Formats supportés: JPG, JPEG, PNG, BMP, TIFF"
            )
            
            # Affichage des images uploadées
            if uploaded_files:
                self._display_uploaded_images(uploaded_files)
                
                # Bouton de prédiction
                if st.button("🔍 Analyser les Images", type="primary", use_container_width=True):
                    self._handle_prediction(uploaded_files, prediction_config)
            
            # Option de prédiction par lot
            with st.expander("📁 Prédiction par Lot", expanded=False):
                st.info("🚧 Fonctionnalité en développement")
                st.write("Permettra d'analyser un dossier entier d'images")
                
                batch_folder = st.text_input(
                    "Chemin vers le dossier d'images",
                    placeholder="C:/path/to/your/images/folder",
                    disabled=True
                )
                
                if st.button("📂 Analyser le Dossier", disabled=True):
                    st.warning("Fonctionnalité non encore implémentée")
                    
        except Exception as e:
            st.error(f"Erreur dans l'interface de prédiction: {e}")
    
    def _render_model_info(self):
        """Affiche les informations sur le modèle actuel."""
        try:
            st.subheader("🤖 Modèle Actuel")
            
            # Récupération des informations du modèle
            training_results = self._state_manager.get_state("training_results")
            
            if training_results:
                # Informations principales
                st.metric(
                    "Type de Pipeline",
                    training_results.get("pipeline_type", "Unknown")
                )
                
                st.metric(
                    "Nom du Pipeline", 
                    training_results.get("pipeline_name", "Unknown")
                )
                
                # Score d'entraînement
                training_score = training_results.get("training_score")
                if training_score:
                    st.metric(
                        "Score d'Entraînement",
                        f"{training_score:.3f}",
                        delta=f"{training_score*100:.1f}%"
                    )
                
                # Métriques d'évaluation si disponibles
                evaluation_results = self._state_manager.get_state("evaluation_results")
                if evaluation_results and hasattr(evaluation_results, 'metrics'):
                    st.write("**Métriques d'Évaluation:**")
                    metrics = evaluation_results.metrics
                    
                    if 'accuracy' in metrics:
                        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                    if 'f1_score' in metrics:
                        st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
                
                # Classes prédites
                data_config = self._state_manager.get_state("current_data_config")
                if data_config and hasattr(data_config, 'selected_classes'):
                    st.write("**Classes Détectées:**")
                    for i, class_name in enumerate(data_config.selected_classes):
                        st.write(f"• {class_name}")
            
        except Exception as e:
            st.error(f"Erreur affichage informations modèle: {e}")
    
    def _display_uploaded_images(self, uploaded_files: List):
        """
        Affiche les images uploadées dans une grille.
        
        Args:
            uploaded_files: Liste des fichiers uploadés
        """
        try:
            st.subheader(f"📸 Images Uploadées ({len(uploaded_files)})")
            
            # Grille d'images
            cols = st.columns(min(len(uploaded_files), 4))
            
            for idx, uploaded_file in enumerate(uploaded_files):
                col_idx = idx % len(cols)
                
                with cols[col_idx]:
                    # Chargement et affichage de l'image
                    image = Image.open(uploaded_file)
                    st.image(
                        image, 
                        caption=uploaded_file.name,
                        use_column_width=True
                    )
                    
                    # Informations sur l'image
                    st.caption(f"Taille: {image.size[0]}x{image.size[1]}")
                    st.caption(f"Mode: {image.mode}")
                    
        except Exception as e:
            st.error(f"Erreur affichage images: {e}")
    
    def _handle_prediction(self, uploaded_files: List, prediction_config: Dict):
        """
        Gère le processus de prédiction sur les images uploadées.
        
        Args:
            uploaded_files: Liste des fichiers uploadés
            prediction_config: Configuration de prédiction
        """
        try:
            # Indicateur de progression
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            total_images = len(uploaded_files)
            
            with progress_placeholder.container():
                progress_bar = st.progress(0)
                progress_bar.progress(0, f"Initialisation... (0/{total_images})")
            
            # Résultats de prédiction
            all_predictions = []
            
            # Traitement image par image
            for idx, uploaded_file in enumerate(uploaded_files):
                
                # Mise à jour du progrès
                progress = (idx + 1) / total_images
                progress_bar.progress(
                    progress, 
                    f"Analyse de {uploaded_file.name}... ({idx + 1}/{total_images})"
                )
                
                with status_placeholder.container():
                    st.info(f"🔄 Traitement de l'image: {uploaded_file.name}")
                
                try:
                    # Chargement de l'image
                    image = Image.open(uploaded_file)
                    
                    # Prédiction
                    prediction_result = self._prediction_service.predict_single_image(
                        image, 
                        prediction_config
                    )
                    
                    if prediction_result.is_successful:
                        prediction_result.image_name = uploaded_file.name
                        all_predictions.append(prediction_result)
                    else:
                        st.error(f"❌ Erreur pour {uploaded_file.name}: {prediction_result.error_message}")
                        
                except Exception as img_error:
                    st.error(f"❌ Erreur traitement {uploaded_file.name}: {img_error}")
                    continue
            
            # Finalisation
            progress_bar.progress(1.0, "Analyse terminée!")
            status_placeholder.success(f"✅ {len(all_predictions)}/{total_images} images analysées avec succès!")
            
            # Sauvegarde des résultats
            if all_predictions:
                self._state_manager.set_state(
                    "prediction_results",
                    all_predictions,
                    scope=StateScope.SESSION
                )
            
            # Nettoyage de l'interface
            progress_placeholder.empty()
            status_placeholder.empty()
            
            # Message de succès
            if all_predictions:
                st.success(f"🎉 Prédictions terminées! {len(all_predictions)} images analysées.")
                st.balloons()
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {e}")
            st.error("Détails:", str(traceback.format_exc()))
    
    def _render_prediction_results(self):
        """Affiche les résultats de prédiction."""
        try:
            # Récupération des résultats
            prediction_results = self._state_manager.get_state(
                "prediction_results",
                scope=StateScope.SESSION
            )
            
            if not prediction_results:
                return
            
            st.subheader("📊 Résultats des Prédictions")
            
            # Résumé global
            self._render_prediction_summary(prediction_results)
            
            # Résultats détaillés
            self._render_detailed_predictions(prediction_results)
            
            # Export des résultats
            self._render_export_options(prediction_results)
                
        except Exception as e:
            st.error(f"Erreur lors de l'affichage des résultats: {e}")
    
    def _render_prediction_summary(self, prediction_results: List[PredictionResult]):
        """
        Affiche un résumé des prédictions.
        
        Args:
            prediction_results: Liste des résultats de prédiction
        """
        try:
            st.subheader("📈 Résumé Global")
            
            # Comptage par classe prédite
            class_counts = {}
            total_confidence = 0
            
            for result in prediction_results:
                predicted_class = result.predicted_class
                confidence = result.confidence
                
                class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
                total_confidence += confidence
            
            # Métriques globales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Images Analysées", len(prediction_results))
            
            with col2:
                avg_confidence = total_confidence / len(prediction_results) if prediction_results else 0
                st.metric("Confiance Moyenne", f"{avg_confidence:.3f}")
            
            with col3:
                covid_count = class_counts.get("COVID", 0)
                st.metric("Cas COVID Détectés", covid_count)
            
            with col4:
                normal_count = class_counts.get("Normal", 0)
                st.metric("Cas Normaux", normal_count)
            
            # Graphique de distribution
            if class_counts:
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 6))
                classes = list(class_counts.keys())
                counts = list(class_counts.values())
                
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
                bars = ax.bar(classes, counts, color=colors[:len(classes)])
                
                ax.set_title('Distribution des Prédictions par Classe')
                ax.set_xlabel('Classes')
                ax.set_ylabel('Nombre d\'Images')
                
                # Ajout des valeurs sur les barres
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           str(count), ha='center', va='bottom')
                
                st.pyplot(fig)
                plt.close()
            
        except Exception as e:
            st.error(f"Erreur affichage résumé: {e}")
    
    def _render_detailed_predictions(self, prediction_results: List[PredictionResult]):
        """
        Affiche les résultats détaillés de chaque prédiction.
        
        Args:
            prediction_results: Liste des résultats de prédiction
        """
        try:
            st.subheader("🔍 Résultats Détaillés")
            
            # Filtre par classe
            all_classes = list(set([r.predicted_class for r in prediction_results]))
            selected_classes = st.multiselect(
                "Filtrer par classe prédite",
                options=all_classes,
                default=all_classes,
                key="prediction_class_filter"
            )
            
            # Filtre par confiance
            min_confidence = st.slider(
                "Confiance minimale",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key="prediction_confidence_filter"
            )
            
            # Filtrage des résultats
            filtered_results = [
                r for r in prediction_results 
                if r.predicted_class in selected_classes and r.confidence >= min_confidence
            ]
            
            st.write(f"**{len(filtered_results)} résultats affichés** (sur {len(prediction_results)} total)")
            
            # Affichage en grille
            for idx, result in enumerate(filtered_results):
                
                with st.expander(f"📸 {result.image_name} - {result.predicted_class} ({result.confidence:.3f})", expanded=idx < 5):
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Métriques principales
                        st.metric("Classe Prédite", result.predicted_class)
                        st.metric("Confiance", f"{result.confidence:.3f}")
                        
                        if hasattr(result, 'processing_time'):
                            st.metric("Temps de Traitement", f"{result.processing_time:.2f}s")
                    
                    with col2:
                        # Probabilités par classe
                        if hasattr(result, 'class_probabilities') and result.class_probabilities:
                            st.write("**Probabilités par classe:**")
                            
                            import pandas as pd
                            prob_df = pd.DataFrame(
                                list(result.class_probabilities.items()),
                                columns=['Classe', 'Probabilité']
                            )
                            prob_df = prob_df.sort_values('Probabilité', ascending=False)
                            
                            # Graphique en barres
                            import matplotlib.pyplot as plt
                            
                            fig, ax = plt.subplots(figsize=(8, 4))
                            bars = ax.barh(prob_df['Classe'], prob_df['Probabilité'])
                            
                            # Coloration selon la probabilité
                            colors = ['red' if p > 0.5 else 'orange' if p > 0.3 else 'green' 
                                     for p in prob_df['Probabilité']]
                            for bar, color in zip(bars, colors):
                                bar.set_color(color)
                            
                            ax.set_xlabel('Probabilité')
                            ax.set_title(f'Probabilités - {result.image_name}')
                            ax.set_xlim(0, 1)
                            
                            st.pyplot(fig)
                            plt.close()
            
        except Exception as e:
            st.error(f"Erreur affichage détails: {e}")
    
    def _render_export_options(self, prediction_results: List[PredictionResult]):
        """
        Affiche les options d'export des résultats.
        
        Args:
            prediction_results: Liste des résultats de prédiction
        """
        try:
            st.subheader("💾 Export des Résultats")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export CSV
                if st.button("📄 Exporter CSV", use_container_width=True):
                    csv_data = self._create_csv_export(prediction_results)
                    st.download_button(
                        "💾 Télécharger CSV",
                        data=csv_data,
                        file_name=f"predictions_covid_{st.session_state.get('export_timestamp', 'results')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                # Export JSON
                if st.button("📋 Exporter JSON", use_container_width=True):
                    json_data = self._create_json_export(prediction_results)
                    st.download_button(
                        "💾 Télécharger JSON",
                        data=json_data,
                        file_name=f"predictions_covid_{st.session_state.get('export_timestamp', 'results')}.json",
                        mime="application/json"
                    )
            
            with col3:
                # Export Rapport
                if st.button("📊 Générer Rapport", use_container_width=True):
                    st.info("🚧 Génération de rapport en développement")
            
        except Exception as e:
            st.error(f"Erreur options export: {e}")
    
    def _create_csv_export(self, prediction_results: List[PredictionResult]) -> str:
        """Crée un export CSV des résultats."""
        try:
            import pandas as pd
            
            data = []
            for result in prediction_results:
                row = {
                    'image_name': result.image_name,
                    'predicted_class': result.predicted_class,
                    'confidence': result.confidence,
                    'processing_time': getattr(result, 'processing_time', None)
                }
                
                # Ajout des probabilités par classe
                if hasattr(result, 'class_probabilities') and result.class_probabilities:
                    for class_name, prob in result.class_probabilities.items():
                        row[f'prob_{class_name}'] = prob
                
                data.append(row)
            
            df = pd.DataFrame(data)
            return df.to_csv(index=False)
            
        except Exception as e:
            st.error(f"Erreur création CSV: {e}")
            return ""
    
    def _create_json_export(self, prediction_results: List[PredictionResult]) -> str:
        """Crée un export JSON des résultats."""
        try:
            import json
            
            data = {
                'metadata': {
                    'total_predictions': len(prediction_results),
                    'export_timestamp': str(st.session_state.get('export_timestamp', 'unknown')),
                    'model_info': self._state_manager.get_state("training_results")
                },
                'predictions': []
            }
            
            for result in prediction_results:
                prediction_data = {
                    'image_name': result.image_name,
                    'predicted_class': result.predicted_class,
                    'confidence': result.confidence,
                    'class_probabilities': getattr(result, 'class_probabilities', {}),
                    'processing_time': getattr(result, 'processing_time', None)
                }
                data['predictions'].append(prediction_data)
            
            return json.dumps(data, indent=2, default=str)
            
        except Exception as e:
            st.error(f"Erreur création JSON: {e}")
            return ""


def main():
    """
    Point d'entrée principal de la page de prédiction.
    
    Utilise le pattern Controller pour orchestrer l'interaction
    entre les composants de l'architecture Clean.
    """
    try:
        # Configuration de la page Streamlit
        st.set_page_config(
            page_title="Prédiction - COVID Detection",
            page_icon="🔮",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Création et rendu du contrôleur
        controller = PredictionPageController()
        controller.render()
        
    except Exception as e:
        st.error("❌ Erreur critique dans la page de prédiction")
        st.error(f"Détails: {e}")
        st.error("Traceback complet:")
        st.code(traceback.format_exc())
        
        # Fallback vers le widget original
        st.warning("🔄 Tentative de basculement vers l'ancien widget...")
        try:
            from src.features.Widget_Streamlit.W_Prediction import render_prediction_widget
            render_prediction_widget()
        except Exception as fallback_error:
            st.error(f"❌ Erreur aussi dans le fallback: {fallback_error}")


if __name__ == "__main__":
    main()
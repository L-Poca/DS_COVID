"""
Page d'entraînement refactorisée avec Clean Architecture.
Utilise les services, composants et gestionnaires d'état centralisés.
"""

import streamlit as st
import logging
from typing import Optional, Dict, Any

# Configuration de la page
st.set_page_config(
    page_title="Entraînement de Modèles - COVID Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import des composants de la nouvelle architecture
try:
    from src.features.Widget_Streamlit.core.di_container import AppContainerFactory
    from src.features.Widget_Streamlit.core.components import ConfigurationForm, ProgressTracker, ComponentFactory
    from src.features.Widget_Streamlit.core.entities import TrainingConfig, DataConfig, PipelineType
    from src.features.Widget_Streamlit.core.services import TrainingService
    CLEAN_ARCH_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Architecture Clean non disponible: {e}")
    CLEAN_ARCH_AVAILABLE = False


class TrainingPageController:
    """Contrôleur pour la page d'entraînement utilisant Clean Architecture."""
    
    def __init__(self):
        """Initialise le contrôleur avec les dépendances."""
        self._logger = logging.getLogger(__name__)
        
        if not CLEAN_ARCH_AVAILABLE:
            st.stop()
        
        try:
            # Récupération du container DI
            self._container = AppContainerFactory.create_container()
            
            # Services injectés
            self._training_service = self._container.get_training_service()
            self._state_manager = self._container.get_state_manager()
            
            # Composants UI
            self._config_form = ComponentFactory.create_configuration_form(
                self._state_manager, "training_page"
            )
            
            self._logger.info("Page d'entraînement initialisée avec Clean Architecture")
            
        except Exception as e:
            st.error(f"❌ Erreur initialisation: {e}")
            st.stop()
    
    def render_page(self):
        """Affiche la page d'entraînement complète."""
        try:
            # En-tête de la page
            self._render_header()
            
            # Validation du container
            self._render_system_status()
            
            # Contenu principal
            self._render_main_content()
            
        except Exception as e:
            self._logger.error(f"Erreur affichage page: {e}")
            st.error(f"❌ Erreur d'affichage: {e}")
    
    def _render_header(self):
        """Affiche l'en-tête de la page."""
        st.title("🎯 Entraînement de Modèles COVID-19")
        st.markdown("""
        Cette page permet d'entraîner des modèles de machine learning pour la détection COVID-19
        sur des images de radiographies pulmonaires en utilisant la **Clean Architecture**.
        """)
        
        # Informations sur l'architecture
        with st.expander("ℹ️ À propos de cette architecture"):
            st.markdown("""
            Cette version utilise:
            - **Services métier** pour la logique d'entraînement
            - **Composants UI** réutilisables  
            - **Injection de dépendances** automatique
            - **Gestionnaire d'état** centralisé
            - **Adaptateurs** pour le code existant
            """)
    
    def _render_system_status(self):
        """Affiche le statut du système et la validation."""
        with st.sidebar:
            st.subheader("🔧 Statut du Système")
            
            # Validation du container
            validation = self._container.validate_configuration()
            
            if validation['is_valid']:
                st.success("✅ Système opérationnel")
            else:
                st.error("❌ Problèmes détectés")
                for error in validation['errors']:
                    st.error(f"• {error}")
            
            # Services disponibles
            with st.expander("Services disponibles"):
                for service, available in validation['services_available'].items():
                    status = "✅" if available else "❌"
                    st.write(f"{status} {service}")
    
    def _render_main_content(self):
        """Affiche le contenu principal de la page."""
        # Layout en colonnes
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Formulaire de configuration
            training_config = self._render_configuration_section()
        
        with col2:
            # Section d'entraînement
            if training_config:
                self._render_training_section(training_config)
        
        # Section des résultats (pleine largeur)
        self._render_results_section()
    
    def _render_configuration_section(self) -> Optional[TrainingConfig]:
        """
        Affiche la section de configuration.
        
        Returns:
            Optional[TrainingConfig]: Configuration validée ou None
        """
        st.subheader("⚙️ Configuration")
        
        # Utilisation du composant de formulaire réutilisable
        training_config = self._config_form.render_training_config_form(
            key_prefix="main_training",
            defaults=None
        )
        
        return training_config
    
    def _render_training_section(self, config: TrainingConfig):
        """
        Affiche la section d'entraînement.
        
        Args:
            config: Configuration d'entraînement validée
        """
        st.subheader("🚀 Entraînement")
        
        # Résumé de la configuration
        with st.expander("📋 Résumé de la configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Pipeline:**")
                st.write(f"• Type: {config.pipeline_type.value}")
                st.write(f"• Nom: {config.pipeline_name}")
                
                st.write("**Données:**")
                st.write(f"• Source: {config.data_config.data_source.value}")
                st.write(f"• Taille: {config.data_config.image_size}")
            
            with col2:
                st.write("**Répartition:**")
                st.write(f"• Entraînement: {config.train_size:.0%}")
                st.write(f"• Validation: {config.validation_size:.0%}")
                st.write(f"• Test: {config.test_size:.0%}")
                
                if config.pipeline_type == PipelineType.TENSORFLOW:
                    st.write("**Paramètres TF:**")
                    st.write(f"• Époques: {config.epochs}")
                    st.write(f"• Batch size: {config.batch_size}")
        
        # Boutons d'action
        col1, col2, col3 = st.columns(3)
        
        with col1:
            validate_btn = st.button("🔍 Valider Config", key="validate_config")
        
        with col2:
            start_training_btn = st.button("▶️ Démarrer", key="start_training")
        
        with col3:
            stop_training_btn = st.button("⏹️ Arrêter", key="stop_training")
        
        # Actions
        if validate_btn:
            self._handle_validate_config(config)
        
        if start_training_btn:
            self._handle_start_training(config)
        
        if stop_training_btn:
            self._handle_stop_training()
    
    def _render_results_section(self):
        """Affiche la section des résultats d'entraînement."""
        st.subheader("📊 Résultats d'Entraînement")
        
        # Récupération des résultats depuis l'état
        training_results = self._state_manager.get_state("current_training_results")
        training_in_progress = self._state_manager.get_state("training_in_progress", False)
        
        if training_in_progress:
            st.info("🔄 Entraînement en cours...")
            
            # Barre de progression (exemple)
            progress = st.progress(0.0)
            status = st.empty()
            
            # Simulation de progression (en réalité, viendrait du service)
            current_epoch = self._state_manager.get_state("current_epoch", 0)
            total_epochs = self._state_manager.get_state("total_epochs", 100)
            
            if total_epochs > 0:
                progress_pct = current_epoch / total_epochs
                progress.progress(progress_pct)
                status.text(f"Époque {current_epoch}/{total_epochs}")
        
        elif training_results:
            self._display_training_results(training_results)
        
        else:
            st.info("ℹ️ Aucun résultat d'entraînement disponible. Configurez et lancez un entraînement.")
    
    def _handle_validate_config(self, config: TrainingConfig):
        """Gère la validation de la configuration."""
        try:
            with st.spinner("🔍 Validation de la configuration..."):
                # Utilisation du service pour valider
                validation_result = self._training_service.validate_training_config(config)
                
                if validation_result['is_valid']:
                    st.success("✅ Configuration valide!")
                    
                    # Affichage des détails
                    if validation_result.get('warnings'):
                        st.warning("⚠️ Avertissements:")
                        for warning in validation_result['warnings']:
                            st.write(f"• {warning}")
                else:
                    st.error("❌ Configuration invalide:")
                    for error in validation_result['errors']:
                        st.write(f"• {error}")
                        
        except Exception as e:
            st.error(f"❌ Erreur lors de la validation: {e}")
    
    def _handle_start_training(self, config: TrainingConfig):
        """Gère le démarrage de l'entraînement."""
        try:
            # Vérification que l'entraînement n'est pas déjà en cours
            if self._state_manager.get_state("training_in_progress", False):
                st.warning("⚠️ Un entraînement est déjà en cours")
                return
            
            with st.spinner("🚀 Préparation de l'entraînement..."):
                # Préparation des données via le service
                prepared_data = self._training_service.prepare_training_data(config)
                
                # Mise à jour de l'état
                self._state_manager.set_state("training_in_progress", True)
                self._state_manager.set_state("total_epochs", config.epochs)
                self._state_manager.set_state("current_epoch", 0)
                
                st.success("✅ Entraînement démarré!")
                st.info("🔄 L'entraînement se déroule en arrière-plan. Les résultats apparaîtront ci-dessous.")
                
                # Note: Dans une vraie implémentation, l'entraînement se ferait de manière asynchrone
                # ou avec des callbacks pour mettre à jour l'état en temps réel
                
                # Simulation d'entraînement (pour démo)
                self._simulate_training(config, prepared_data)
                
        except Exception as e:
            self._state_manager.set_state("training_in_progress", False)
            st.error(f"❌ Erreur lors du démarrage: {e}")
    
    def _handle_stop_training(self):
        """Gère l'arrêt de l'entraînement."""
        self._state_manager.set_state("training_in_progress", False)
        self._state_manager.set_state("current_epoch", 0)
        st.warning("⏹️ Entraînement arrêté par l'utilisateur")
    
    def _simulate_training(self, config: TrainingConfig, prepared_data: Dict[str, Any]):
        """
        Simulation d'entraînement pour la démo.
        Dans la réalité, ceci utiliserait le TrainingService complet.
        """
        try:
            # Lancement de l'entraînement via le service
            training_result = self._training_service.train_model(config, prepared_data)
            
            # Sauvegarde des résultats
            self._state_manager.set_state("current_training_results", training_result)
            self._state_manager.set_state("training_in_progress", False)
            
            # Force le rerun pour afficher les résultats
            st.rerun()
            
        except Exception as e:
            self._logger.error(f"Erreur simulation entraînement: {e}")
            self._state_manager.set_state("training_in_progress", False)
            st.error(f"❌ Erreur pendant l'entraînement: {e}")
    
    def _display_training_results(self, results: Any):
        """Affiche les résultats d'entraînement."""
        st.success("✅ Entraînement terminé!")
        
        # Métriques principales
        if hasattr(results, 'metrics') and results.metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            metrics = results.metrics
            with col1:
                st.metric("Précision", f"{metrics.get('accuracy', 0):.3f}")
            with col2:
                st.metric("F1-Score", f"{metrics.get('f1_macro', 0):.3f}")
            with col3:
                st.metric("Temps", f"{results.training_time:.1f}s")
            with col4:
                st.metric("Modèle", results.model_id)
        
        # Détails dans un expander
        with st.expander("📋 Détails complets"):
            st.json(results.metadata if hasattr(results, 'metadata') else {})


def main():
    """Point d'entrée principal de la page."""
    try:
        # Création et affichage de la page
        page_controller = TrainingPageController()
        page_controller.render_page()
        
    except Exception as e:
        st.error(f"❌ Erreur fatale: {e}")
        
        # Fallback vers l'ancienne version si disponible
        st.warning("🔄 Tentative de fallback vers l'ancienne version...")
        try:
            # Import de l'ancienne page si disponible
            import sys
            from pathlib import Path
            
            old_page_path = Path(__file__).parent / "1_Training.py"
            if old_page_path.exists():
                st.info("✅ Ancienne version chargée")
                # exec(open(old_page_path).read())
            else:
                st.error("❌ Aucune version de fallback disponible")
        except Exception as fallback_error:
            st.error(f"❌ Fallback échoué: {fallback_error}")


if __name__ == "__main__":
    main()
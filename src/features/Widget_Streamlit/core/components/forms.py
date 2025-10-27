"""
Composants de formulaire réutilisables pour l'interface Streamlit.
Composants atomiques focalisés sur la saisie de données.
"""

import streamlit as st
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from pathlib import Path
import logging

from ...entities import DataConfig, TrainingConfig, PipelineType, DataSource
from ...interfaces import IStateManager
from ...interfaces.i_state_manager import StateScope


class ConfigurationForm:
    """Formulaire de configuration réutilisable pour les paramètres de training."""
    
    def __init__(self, state_manager: IStateManager, form_id: str = "config_form"):
        """
        Initialise le formulaire de configuration.
        
        Args:
            state_manager: Gestionnaire d'état pour persistance
            form_id: ID unique du formulaire
        """
        self._state_manager = state_manager
        self._form_id = form_id
        self._logger = logging.getLogger(__name__)
        
    def render_data_config_form(self, 
                               key_prefix: str = "data",
                               defaults: Optional[DataConfig] = None) -> DataConfig:
        """
        Affiche un formulaire de configuration des données.
        
        Args:
            key_prefix: Préfixe pour les clés d'état
            defaults: Configuration par défaut
            
        Returns:
            DataConfig: Configuration saisie par l'utilisateur
        """
        st.subheader("📊 Configuration des Données")
        
        with st.form(f"{self._form_id}_data_config"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Source des données
                data_source = st.selectbox(
                    "Source des données",
                    options=[ds.value for ds in DataSource],
                    index=0 if not defaults else [ds.value for ds in DataSource].index(defaults.data_source.value),
                    key=f"{key_prefix}_source",
                    help="Choisissez la source de données COVID-19"
                )
                
                # Taille des images
                image_sizes = [(64, 64), (128, 128), (224, 224), (256, 256), (299, 299)]
                size_options = [f"{w}x{h}" for w, h in image_sizes]
                
                default_size_idx = 2  # 224x224 par défaut
                if defaults and defaults.image_size in image_sizes:
                    default_size_idx = image_sizes.index(defaults.image_size)
                
                selected_size = st.selectbox(
                    "Taille des images",
                    options=size_options,
                    index=default_size_idx,
                    key=f"{key_prefix}_size",
                    help="Taille de redimensionnement des images"
                )
                
                # Classes à inclure
                available_classes = ['COVID', 'Normal', 'Viral Pneumonia']
                selected_classes = st.multiselect(
                    "Classes à inclure",
                    options=available_classes,
                    default=defaults.classes if defaults and defaults.classes else available_classes,
                    key=f"{key_prefix}_classes",
                    help="Sélectionnez les classes médicales à analyser"
                )
            
            with col2:
                # Taille d'échantillon
                sample_size = st.number_input(
                    "Taille d'échantillon (0 = tous)",
                    min_value=0,
                    max_value=10000,
                    value=defaults.sample_size if defaults else 0,
                    key=f"{key_prefix}_sample_size",
                    help="Nombre d'images par classe (0 pour toutes)"
                )
                
                # Options de preprocessing
                normalize = st.checkbox(
                    "Normaliser les images",
                    value=defaults.normalize if defaults else True,
                    key=f"{key_prefix}_normalize",
                    help="Normalise les pixels entre 0 et 1"
                )
                
                balance_classes = st.checkbox(
                    "Équilibrer les classes",
                    value=defaults.balance_classes if defaults else True,
                    key=f"{key_prefix}_balance",
                    help="Assure un nombre égal d'échantillons par classe"
                )
                
                # Seed pour reproductibilité
                random_state = st.number_input(
                    "Seed aléatoire",
                    min_value=0,
                    max_value=9999,
                    value=defaults.random_state if defaults else 42,
                    key=f"{key_prefix}_seed",
                    help="Pour la reproductibilité des résultats"
                )
            
            submitted = st.form_submit_button("✅ Valider Configuration Données")
            
            if submitted:
                # Conversion de la taille sélectionnée
                width, height = map(int, selected_size.split('x'))
                
                # Création de la configuration
                config = DataConfig(
                    data_source=DataSource(data_source),
                    image_size=(width, height),
                    classes=selected_classes if selected_classes else None,
                    sample_size=sample_size,
                    normalize=normalize,
                    balance_classes=balance_classes,
                    random_state=random_state
                )
                
                # Sauvegarde dans l'état
                self._state_manager.set_state(f"{key_prefix}_config", config, StateScope.SESSION)
                st.success("✅ Configuration des données validée!")
                return config
            
        # Retourner la configuration sauvegardée ou par défaut
        saved_config = self._state_manager.get_state(f"{key_prefix}_config", StateScope.SESSION)
        if saved_config:
            return saved_config
        elif defaults:
            return defaults
        else:
            return DataConfig()  # Configuration par défaut
    
    def render_training_config_form(self, 
                                   key_prefix: str = "training",
                                   defaults: Optional[TrainingConfig] = None) -> TrainingConfig:
        """
        Affiche un formulaire de configuration d'entraînement.
        
        Args:
            key_prefix: Préfixe pour les clés d'état
            defaults: Configuration par défaut
            
        Returns:
            TrainingConfig: Configuration saisie par l'utilisateur
        """
        st.subheader("🎯 Configuration d'Entraînement")
        
        with st.form(f"{self._form_id}_training_config"):
            # Configuration des données (inline)
            data_config = self.render_data_config_inline(key_prefix + "_data", 
                                                       defaults.data_config if defaults else None)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Type de pipeline
                pipeline_type = st.selectbox(
                    "Type de Pipeline",
                    options=[pt.value for pt in PipelineType],
                    index=0 if not defaults else [pt.value for pt in PipelineType].index(defaults.pipeline_type.value),
                    key=f"{key_prefix}_pipeline_type",
                    help="Choisissez le framework ML"
                )
                
                # Nom du pipeline (sera dynamique selon le type)
                pipeline_name = st.text_input(
                    "Nom du Pipeline",
                    value=defaults.pipeline_name if defaults else "default_pipeline",
                    key=f"{key_prefix}_pipeline_name",
                    help="Identifiant unique du pipeline"
                )
            
            with col2:
                # Répartition des données
                st.write("**Répartition des données:**")
                train_size = st.slider(
                    "Entraînement (%)",
                    min_value=50,
                    max_value=90,
                    value=int((defaults.train_size if defaults else 0.7) * 100),
                    key=f"{key_prefix}_train_size",
                    help="Pourcentage pour l'entraînement"
                ) / 100
                
                val_size = st.slider(
                    "Validation (%)",
                    min_value=5,
                    max_value=30,
                    value=int((defaults.validation_size if defaults else 0.15) * 100),
                    key=f"{key_prefix}_val_size",
                    help="Pourcentage pour la validation"
                ) / 100
                
                test_size = 1.0 - train_size - val_size
                st.write(f"Test: {test_size:.0%}")
            
            with col3:
                # Paramètres spécifiques selon le type
                if pipeline_type == "tensorflow":
                    epochs = st.number_input(
                        "Nombre d'époques",
                        min_value=1,
                        max_value=200,
                        value=defaults.epochs if defaults else 50,
                        key=f"{key_prefix}_epochs",
                        help="Nombre d'itérations d'entraînement"
                    )
                    
                    batch_size = st.number_input(
                        "Taille de batch",
                        min_value=8,
                        max_value=256,
                        value=defaults.batch_size if defaults else 32,
                        key=f"{key_prefix}_batch_size",
                        help="Nombre d'échantillons par batch"
                    )
                else:
                    epochs = 1  # N/A pour sklearn
                    batch_size = 32  # N/A pour sklearn
                
                # Validation croisée
                cross_validation = st.checkbox(
                    "Validation croisée",
                    value=defaults.cross_validation if defaults else False,
                    key=f"{key_prefix}_cv",
                    help="Active la validation croisée"
                )
                
                cv_folds = 5
                if cross_validation:
                    cv_folds = st.slider(
                        "Nombre de folds",
                        min_value=3,
                        max_value=10,
                        value=defaults.cv_folds if defaults else 5,
                        key=f"{key_prefix}_cv_folds"
                    )
            
            # Paramètres avancés dans un expander
            with st.expander("⚙️ Paramètres Avancés"):
                training_params_str = st.text_area(
                    "Paramètres d'entraînement (JSON)",
                    value="{}",
                    key=f"{key_prefix}_params",
                    help="Paramètres supplémentaires au format JSON"
                )
            
            submitted = st.form_submit_button("🚀 Valider Configuration Entraînement")
            
            if submitted:
                # Validation de la répartition
                if abs(train_size + val_size + test_size - 1.0) > 0.01:
                    st.error("❌ La somme des répartitions doit égaler 100%")
                    return defaults or TrainingConfig(data_config=data_config)
                
                # Parsing des paramètres JSON
                try:
                    import json
                    training_params = json.loads(training_params_str) if training_params_str.strip() else None
                except json.JSONDecodeError:
                    st.error("❌ Paramètres JSON invalides")
                    training_params = None
                
                # Création de la configuration
                config = TrainingConfig(
                    data_config=data_config,
                    pipeline_type=PipelineType(pipeline_type),
                    pipeline_name=pipeline_name,
                    train_size=train_size,
                    validation_size=val_size,
                    test_size=test_size,
                    epochs=epochs,
                    batch_size=batch_size,
                    cross_validation=cross_validation,
                    cv_folds=cv_folds if cross_validation else None,
                    training_params=training_params
                )
                
                # Sauvegarde dans l'état
                self._state_manager.set_state(f"{key_prefix}_config", config, StateScope.SESSION)
                st.success("✅ Configuration d'entraînement validée!")
                return config
        
        # Retourner la configuration sauvegardée ou par défaut
        saved_config = self._state_manager.get_state(f"{key_prefix}_config", StateScope.SESSION)
        if saved_config:
            return saved_config
        elif defaults:
            return defaults
        else:
            return TrainingConfig(data_config=DataConfig())
    
    def render_data_config_inline(self, 
                                 key_prefix: str,
                                 defaults: Optional[DataConfig] = None) -> DataConfig:
        """Version compacte du formulaire de données pour inclusion dans d'autres formes."""
        col1, col2 = st.columns(2)
        
        with col1:
            data_source = st.selectbox(
                "Source",
                options=[ds.value for ds in DataSource],
                index=0,
                key=f"{key_prefix}_source_inline"
            )
            
            size_options = ["224x224", "256x256", "128x128"]
            selected_size = st.selectbox(
                "Taille",
                options=size_options,
                key=f"{key_prefix}_size_inline"
            )
        
        with col2:
            classes = st.multiselect(
                "Classes",
                options=['COVID', 'Normal', 'Viral Pneumonia'],
                default=['COVID', 'Normal', 'Viral Pneumonia'],
                key=f"{key_prefix}_classes_inline"
            )
            
            normalize = st.checkbox(
                "Normaliser",
                value=True,
                key=f"{key_prefix}_normalize_inline"
            )
        
        width, height = map(int, selected_size.split('x'))
        return DataConfig(
            data_source=DataSource(data_source),
            image_size=(width, height),
            classes=classes if classes else None,
            normalize=normalize
        )


class FileUploadForm:
    """Composant de téléchargement de fichiers avec validation."""
    
    def __init__(self, 
                 allowed_extensions: List[str] = None,
                 max_file_size_mb: int = 10):
        """
        Initialise le composant de téléchargement.
        
        Args:
            allowed_extensions: Extensions autorisées
            max_file_size_mb: Taille max en MB
        """
        self.allowed_extensions = allowed_extensions or ['.png', '.jpg', '.jpeg']
        self.max_file_size_mb = max_file_size_mb
        self._logger = logging.getLogger(__name__)
    
    def render_single_file_upload(self, 
                                 label: str = "Choisir un fichier",
                                 key: str = "file_upload",
                                 help_text: str = None) -> Optional[bytes]:
        """
        Affiche un widget de téléchargement pour un seul fichier.
        
        Args:
            label: Texte du widget
            key: Clé unique du widget
            help_text: Texte d'aide
            
        Returns:
            Optional[bytes]: Contenu du fichier ou None
        """
        uploaded_file = st.file_uploader(
            label,
            type=[ext.lstrip('.') for ext in self.allowed_extensions],
            key=key,
            help=help_text or f"Formats acceptés: {', '.join(self.allowed_extensions)}"
        )
        
        if uploaded_file is not None:
            # Validation de la taille
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            
            if file_size_mb > self.max_file_size_mb:
                st.error(f"❌ Fichier trop volumineux: {file_size_mb:.1f}MB (max: {self.max_file_size_mb}MB)")
                return None
            
            # Validation de l'extension
            file_ext = Path(uploaded_file.name).suffix.lower()
            if file_ext not in self.allowed_extensions:
                st.error(f"❌ Extension non autorisée: {file_ext}")
                return None
            
            st.success(f"✅ Fichier validé: {uploaded_file.name} ({file_size_mb:.1f}MB)")
            return uploaded_file.getvalue()
        
        return None
    
    def render_multiple_files_upload(self, 
                                    label: str = "Choisir des fichiers",
                                    key: str = "files_upload",
                                    max_files: int = 10) -> List[bytes]:
        """
        Affiche un widget de téléchargement pour plusieurs fichiers.
        
        Args:
            label: Texte du widget
            key: Clé unique du widget
            max_files: Nombre maximum de fichiers
            
        Returns:
            List[bytes]: Liste des contenus des fichiers valides
        """
        uploaded_files = st.file_uploader(
            label,
            type=[ext.lstrip('.') for ext in self.allowed_extensions],
            key=key,
            accept_multiple_files=True,
            help=f"Max {max_files} fichiers, formats: {', '.join(self.allowed_extensions)}"
        )
        
        valid_files = []
        
        if uploaded_files:
            if len(uploaded_files) > max_files:
                st.error(f"❌ Trop de fichiers: {len(uploaded_files)} (max: {max_files})")
                return []
            
            for uploaded_file in uploaded_files:
                # Validation de chaque fichier
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                file_ext = Path(uploaded_file.name).suffix.lower()
                
                if file_size_mb > self.max_file_size_mb:
                    st.error(f"❌ {uploaded_file.name}: trop volumineux ({file_size_mb:.1f}MB)")
                    continue
                
                if file_ext not in self.allowed_extensions:
                    st.error(f"❌ {uploaded_file.name}: extension non autorisée")
                    continue
                
                valid_files.append(uploaded_file.getvalue())
            
            if valid_files:
                st.success(f"✅ {len(valid_files)} fichier(s) validé(s)")
        
        return valid_files


class ProgressTracker:
    """Composant pour afficher la progression des tâches longues."""
    
    def __init__(self, total_steps: int, description: str = "Progression"):
        """
        Initialise le tracker de progression.
        
        Args:
            total_steps: Nombre total d'étapes
            description: Description de la tâche
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        
        # Éléments Streamlit
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.detail_text = st.empty()
    
    def update(self, step: int, status: str = "", details: str = ""):
        """
        Met à jour la progression.
        
        Args:
            step: Étape actuelle (0-based)
            status: Texte de statut
            details: Détails additionnels
        """
        self.current_step = step
        progress = min(step / self.total_steps, 1.0)
        
        # Mise à jour des éléments
        self.progress_bar.progress(progress)
        
        status_msg = f"{self.description}: {step}/{self.total_steps}"
        if status:
            status_msg += f" - {status}"
        self.status_text.text(status_msg)
        
        if details:
            self.detail_text.text(details)
    
    def complete(self, final_message: str = "Terminé!"):
        """
        Marque la tâche comme terminée.
        
        Args:
            final_message: Message final à afficher
        """
        self.progress_bar.progress(1.0)
        self.status_text.success(f"✅ {final_message}")
        self.detail_text.empty()
    
    def error(self, error_message: str):
        """
        Affiche une erreur.
        
        Args:
            error_message: Message d'erreur
        """
        self.status_text.error(f"❌ {error_message}")
        self.detail_text.empty()


class EvaluationForm:
    """Formulaire pour la configuration d'évaluation des modèles."""
    
    def __init__(self, state_manager: IStateManager):
        self._state_manager = state_manager
    
    def render_evaluation_config_form(
        self,
        current_config: Optional[Dict] = None,
        key_prefix: str = "eval"
    ) -> Optional[Dict]:
        """
        Affiche le formulaire de configuration d'évaluation.
        
        Args:
            current_config: Configuration actuelle à pré-remplir
            key_prefix: Préfixe pour les clés des widgets
            
        Returns:
            Dict si la configuration est valide, None sinon
        """
        try:
            # Test dataset selection
            st.subheader("📊 Données de Test")
            
            col1, col2 = st.columns(2)
            
            with col1:
                test_size = st.slider(
                    "Taille du jeu de test",
                    min_value=0.1,
                    max_value=0.5,
                    value=current_config.get('test_size', 0.3) if current_config else 0.3,
                    step=0.05,
                    key=f"{key_prefix}_test_size",
                    help="Proportion des données utilisées pour les tests"
                )
            
            with col2:
                cross_validation_folds = st.selectbox(
                    "Nombre de plis (Cross-validation)",
                    options=[3, 5, 10],
                    index=1,  # 5 par défaut
                    key=f"{key_prefix}_cv_folds",
                    help="Nombre de plis pour la validation croisée"
                )
            
            # Métriques à calculer
            st.subheader("📈 Métriques d'Évaluation")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                compute_accuracy = st.checkbox(
                    "Accuracy",
                    value=True,
                    key=f"{key_prefix}_accuracy",
                    help="Proportion de prédictions correctes"
                )
                compute_precision = st.checkbox(
                    "Précision",
                    value=True,
                    key=f"{key_prefix}_precision",
                    help="Proportion de vrais positifs parmi les positifs prédits"
                )
            
            with col2:
                compute_recall = st.checkbox(
                    "Rappel (Sensibilité)",
                    value=True,
                    key=f"{key_prefix}_recall",
                    help="Proportion de vrais positifs détectés"
                )
                compute_f1_score = st.checkbox(
                    "F1-Score",
                    value=True,
                    key=f"{key_prefix}_f1",
                    help="Moyenne harmonique de la précision et du rappel"
                )
            
            with col3:
                compute_roc_auc = st.checkbox(
                    "ROC-AUC",
                    value=True,
                    key=f"{key_prefix}_roc_auc",
                    help="Aire sous la courbe ROC"
                )
                compute_confusion_matrix = st.checkbox(
                    "Matrice de Confusion",
                    value=True,
                    key=f"{key_prefix}_confusion_matrix",
                    help="Matrice des prédictions vs réalité"
                )
            
            # Options avancées
            with st.expander("🔧 Options Avancées", expanded=False):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    confidence_threshold = st.slider(
                        "Seuil de Confiance",
                        min_value=0.1,
                        max_value=0.9,
                        value=0.5,
                        step=0.05,
                        key=f"{key_prefix}_confidence_threshold",
                        help="Seuil pour la classification binaire"
                    )
                
                with col2:
                    stratify_split = st.checkbox(
                        "Stratifier le split",
                        value=True,
                        key=f"{key_prefix}_stratify",
                        help="Maintenir les proportions des classes dans le split"
                    )
                
                random_state = st.number_input(
                    "Random State",
                    min_value=0,
                    max_value=9999,
                    value=42,
                    step=1,
                    key=f"{key_prefix}_random_state",
                    help="Graine pour la reproductibilité"
                )
            
            # Création de la configuration
            config = {
                'test_size': test_size,
                'cross_validation_folds': cross_validation_folds,
                'metrics_config': {
                    'accuracy': compute_accuracy,
                    'precision': compute_precision,
                    'recall': compute_recall,
                    'f1_score': compute_f1_score,
                    'roc_auc': compute_roc_auc,
                    'confusion_matrix': compute_confusion_matrix,
                    'confidence_threshold': confidence_threshold
                },
                'stratify_split': stratify_split,
                'random_state': random_state
            }
            
            return config
                
        except Exception as e:
            st.error(f"Erreur dans le formulaire d'évaluation: {e}")
            return None


class ComponentFactory:
    """Factory pour créer les instances des composants UI."""
    
    @staticmethod
    def create_configuration_form(state_manager: IStateManager) -> ConfigurationForm:
        """Crée une instance de ConfigurationForm avec les dépendances."""
        return ConfigurationForm(state_manager)
    
    @staticmethod
    def create_evaluation_form(state_manager: IStateManager) -> EvaluationForm:
        """Crée une instance de EvaluationForm avec les dépendances."""
        return EvaluationForm(state_manager)
    
    @staticmethod
    def create_prediction_form(state_manager: IStateManager) -> 'PredictionForm':
        """Crée une instance de PredictionForm avec les dépendances."""
        return PredictionForm(state_manager)
    
    @staticmethod
    def create_progress_tracker() -> ProgressTracker:
        """Crée une instance de ProgressTracker."""
        return ProgressTracker()


class PredictionForm:
    """Formulaire pour la configuration de prédiction."""
    
    def __init__(self, state_manager: IStateManager):
        self._state_manager = state_manager
    
    def render_prediction_config_form(self, key_prefix: str = "pred") -> Dict[str, Any]:
        """
        Affiche le formulaire de configuration de prédiction.
        
        Args:
            key_prefix: Préfixe pour les clés des widgets
            
        Returns:
            Dict avec la configuration de prédiction
        """
        try:
            st.subheader("⚙️ Configuration de Prédiction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Seuil de confiance
                confidence_threshold = st.slider(
                    "Seuil de Confiance",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.5,
                    step=0.05,
                    key=f"{key_prefix}_confidence_threshold",
                    help="Seuil minimum pour considérer une prédiction comme fiable"
                )
                
                # Préprocessing
                normalize_images = st.checkbox(
                    "Normaliser les Images",
                    value=True,
                    key=f"{key_prefix}_normalize",
                    help="Normalise les valeurs des pixels entre 0 et 1"
                )
            
            with col2:
                # Redimensionnement
                resize_images = st.checkbox(
                    "Redimensionner automatiquement",
                    value=True,
                    key=f"{key_prefix}_resize",
                    help="Redimensionne les images au format du modèle"
                )
                
                # Batch processing
                batch_size = st.selectbox(
                    "Taille de Batch",
                    options=[1, 4, 8, 16, 32],
                    index=2,  # 8 par défaut
                    key=f"{key_prefix}_batch_size",
                    help="Nombre d'images traitées simultanément"
                )
            
            # Options avancées
            with st.expander("🔧 Options Avancées", expanded=False):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Augmentation de données pour prédiction
                    use_tta = st.checkbox(
                        "Test Time Augmentation (TTA)",
                        value=False,
                        key=f"{key_prefix}_tta",
                        help="Utilise plusieurs versions augmentées de l'image pour améliorer la prédiction"
                    )
                    
                    if use_tta:
                        tta_iterations = st.number_input(
                            "Nombre d'itérations TTA",
                            min_value=2,
                            max_value=10,
                            value=5,
                            key=f"{key_prefix}_tta_iterations",
                            help="Nombre de versions augmentées à utiliser"
                        )
                    else:
                        tta_iterations = 1
                
                with col2:
                    # Affichage des probabilités
                    show_probabilities = st.checkbox(
                        "Afficher toutes les probabilités",
                        value=True,
                        key=f"{key_prefix}_show_probs",
                        help="Affiche les probabilités pour toutes les classes"
                    )
                    
                    # Sauvegarde des résultats
                    save_results = st.checkbox(
                        "Sauvegarder automatiquement",
                        value=True,
                        key=f"{key_prefix}_save_results",
                        help="Sauvegarde automatiquement les résultats de prédiction"
                    )
            
            # Configuration résultante
            config = {
                'confidence_threshold': confidence_threshold,
                'normalize_images': normalize_images,
                'resize_images': resize_images,
                'batch_size': batch_size,
                'use_tta': use_tta,
                'tta_iterations': tta_iterations,
                'show_probabilities': show_probabilities,
                'save_results': save_results
            }
            
            return config
            
        except Exception as e:
            st.error(f"Erreur dans le formulaire de prédiction: {e}")
            return {}
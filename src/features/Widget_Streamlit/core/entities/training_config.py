"""
Entités métier pour les configurations d'entraînement.
Définit les structures de données pour la configuration des modèles.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum


class PipelineType(Enum):
    """Types de pipelines supportés."""
    SKLEARN = "Sklearn (ML Classique)"
    TENSORFLOW = "TensorFlow (Deep Learning)"
    DATA_AUGMENTATION = "Augmentation de données"


class DataSource(Enum):
    """Sources de données supportées."""
    COVID_REAL = "Dataset COVID-19 (Réel)"
    SIMULATED = "Données simulées (test)"
    UPLOAD_CUSTOM = "Upload personnalisé"


@dataclass
class DataConfig:
    """Configuration des données d'entraînement."""
    data_source: DataSource
    selected_classes: List[str] = field(default_factory=list)
    max_images_per_class: int = 500
    target_size: Tuple[int, int] = (224, 224)
    color_mode: str = "rgb"
    
    # Pour données simulées
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    n_classes: Optional[int] = None


@dataclass
class GeneralParams:
    """Paramètres généraux d'entraînement."""
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    enable_comparison: bool = True
    save_results: bool = True
    use_gpu: bool = True
    verbose: bool = False


@dataclass
class TensorFlowParams:
    """Paramètres spécifiques à TensorFlow."""
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2


@dataclass
class TrainingConfig:
    """Configuration complète d'entraînement."""
    pipeline_type: PipelineType
    selected_configs: List[str]
    data_config: DataConfig
    general_params: GeneralParams
    tf_params: Optional[TensorFlowParams] = None
    
    def __post_init__(self):
        """Validation après initialisation."""
        if not self.selected_configs:
            raise ValueError("Au moins une configuration doit être sélectionnée")
        
        if self.pipeline_type == PipelineType.TENSORFLOW and self.tf_params is None:
            self.tf_params = TensorFlowParams()


@dataclass
class ValidationConfig:
    """Configuration pour la validation des données."""
    check_data_availability: bool = True
    validate_file_formats: bool = True
    check_class_balance: bool = True
    min_samples_per_class: int = 10


# Fonctions utilitaires pour la conversion

def dict_to_data_config(config_dict: Dict[str, Any]) -> DataConfig:
    """Convertit un dictionnaire en DataConfig."""
    data_source = DataSource(config_dict["data_source"])
    
    return DataConfig(
        data_source=data_source,
        selected_classes=config_dict.get("selected_classes", []),
        max_images_per_class=config_dict.get("max_images_per_class", 500),
        target_size=tuple(config_dict.get("target_size", (224, 224))),
        color_mode=config_dict.get("color_mode", "rgb"),
        n_samples=config_dict.get("n_samples"),
        n_features=config_dict.get("n_features"),
        n_classes=config_dict.get("n_classes")
    )


def dict_to_general_params(params_dict: Dict[str, Any]) -> GeneralParams:
    """Convertit un dictionnaire en GeneralParams."""
    return GeneralParams(
        test_size=params_dict.get("test_size", 0.2),
        validation_size=params_dict.get("validation_size", 0.1),
        random_state=params_dict.get("random_state", 42),
        enable_comparison=params_dict.get("enable_comparison", True),
        save_results=params_dict.get("save_results", True),
        use_gpu=params_dict.get("use_gpu", True),
        verbose=params_dict.get("verbose", False)
    )


def dict_to_tensorflow_params(tf_dict: Dict[str, Any]) -> TensorFlowParams:
    """Convertit un dictionnaire en TensorFlowParams."""
    return TensorFlowParams(
        epochs=tf_dict.get("epochs", 20),
        batch_size=tf_dict.get("batch_size", 32),
        learning_rate=tf_dict.get("learning_rate", 0.001),
        validation_split=tf_dict.get("validation_split", 0.2)
    )


@dataclass
class EvaluationConfig:
    """
    Configuration pour l'évaluation des modèles.
    Définit les paramètres pour évaluer les performances d'un modèle entraîné.
    """
    
    # Configuration de base héritée de l'entraînement
    data_config: DataConfig
    pipeline_type: PipelineType
    pipeline_name: str
    
    # Paramètres d'évaluation
    test_size: float = 0.3
    cross_validation_folds: int = 5
    
    # Configuration des métriques
    metrics_config: Dict[str, Any] = field(default_factory=lambda: {
        'accuracy': True,
        'precision': True,
        'recall': True,
        'f1_score': True,
        'roc_auc': True,
        'confusion_matrix': True,
        'confidence_threshold': 0.5
    })
    
    # Options avancées
    stratify_split: bool = True
    random_state: int = 42
    
    def __post_init__(self):
        """Validation après initialisation."""
        # Validation test_size
        if not 0.1 <= self.test_size <= 0.5:
            raise ValueError("test_size doit être entre 0.1 et 0.5")
        
        # Validation cross_validation_folds
        if self.cross_validation_folds < 2:
            raise ValueError("cross_validation_folds doit être >= 2")
        
        # Validation metrics_config
        if not isinstance(self.metrics_config, dict):
            raise ValueError("metrics_config doit être un dictionnaire")
        
        # Validation confidence_threshold
        threshold = self.metrics_config.get('confidence_threshold', 0.5)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence_threshold doit être entre 0.0 et 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire."""
        return {
            'data_config': self.data_config.to_dict() if self.data_config else None,
            'pipeline_type': self.pipeline_type.value,
            'pipeline_name': self.pipeline_name,
            'test_size': self.test_size,
            'cross_validation_folds': self.cross_validation_folds,
            'metrics_config': self.metrics_config,
            'stratify_split': self.stratify_split,
            'random_state': self.random_state
        }
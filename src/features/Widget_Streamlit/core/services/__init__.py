"""
Module des services métier.
Services encapsulant la logique business pure selon Clean Architecture.
"""

from .training_service import TrainingService
from .evaluation_service import EvaluationService
from .prediction_service import PredictionService

# Export des services principaux
__all__ = [
    'TrainingService',
    'EvaluationService', 
    'PredictionService'
]


# Documentation des responsabilités des services
SERVICE_DOCUMENTATION = {
    'TrainingService': {
        'description': 'Gestion complète de l\'entraînement des modèles',
        'responsibilities': [
            'Validation des configurations d\'entraînement',
            'Préparation et division des données',
            'Orchestration de l\'entraînement selon le type de pipeline',
            'Sauvegarde et chargement des modèles entraînés',
            'Gestion des erreurs et logging'
        ],
        'dependencies': [
            'IPipelineManager (Sklearn & TensorFlow)',
            'IDataLoader',
            'IModelValidator'
        ]
    },
    
    'EvaluationService': {
        'description': 'Évaluation complète et comparaison des modèles',
        'responsibilities': [
            'Évaluation comprehensive des modèles entraînés',
            'Calcul de métriques détaillées (accuracy, F1, ROC, etc.)',
            'Comparaison de performance entre modèles',
            'Génération de rapports d\'évaluation structurés',
            'Détection du surapprentissage et recommandations'
        ],
        'dependencies': [
            'IPipelineManager (tous types)',
            'IDataLoader',
            'IModelValidator', 
            'IMetricsCalculator',
            'IVisualizationService (optionnel)'
        ]
    },
    
    'PredictionService': {
        'description': 'Service de prédiction et inférence en production',
        'responsibilities': [
            'Prédiction sur images individuelles ou lots',
            'Gestion du cache des modèles chargés',
            'Analyse de confiance des prédictions',
            'Support de prédiction depuis fichiers',
            'Préprocessing automatique des images'
        ],
        'dependencies': [
            'IPipelineManager (tous types)',
            'IDataLoader'
        ]
    }
}


def get_service_info(service_name: str = None) -> dict:
    """
    Retourne la documentation d'un service ou de tous les services.
    
    Args:
        service_name: Nom du service (optionnel)
        
    Returns:
        dict: Documentation du/des service(s)
    """
    if service_name:
        return SERVICE_DOCUMENTATION.get(service_name, {})
    return SERVICE_DOCUMENTATION


# Configuration par défaut pour les services
DEFAULT_SERVICE_CONFIG = {
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    },
    'cache': {
        'max_models': 5,  # Nombre max de modèles en cache
        'auto_cleanup': True
    },
    'validation': {
        'default_cv_folds': 5,
        'confidence_threshold': 0.8,
        'performance_threshold': 0.7
    },
    'prediction': {
        'default_batch_size': 32,
        'timeout_seconds': 300,
        'max_image_size_mb': 10
    }
}
"""
Module d'interfaces pour l'architecture Clean.
Exposition des contrats pour le découplage des dépendances.
"""

from .i_pipeline_manager import (
    IPipelineManager,
    ISklearnPipelineManager,
    ITensorFlowPipelineManager,
    IDataAugmentationManager
)

from .i_data_loader import (
    IDataLoader,
    ICovidDataLoader,
    IDataValidator,
    ICacheManager
)

from .i_validation_service import (
    IModelValidator,
    IMetricsCalculator,
    IResultsExporter,
    IVisualizationService
)

from .i_state_manager import (
    IStateManager,
    INavigationManager,
    ISessionManager,
    IConfigManager,
    StateScope
)

# Export des interfaces principales
__all__ = [
    # Pipeline Management
    'IPipelineManager',
    'ISklearnPipelineManager', 
    'ITensorFlowPipelineManager',
    'IDataAugmentationManager',
    
    # Data Management
    'IDataLoader',
    'ICovidDataLoader',
    'IDataValidator',
    'ICacheManager',
    
    # Validation & Metrics
    'IModelValidator',
    'IMetricsCalculator',
    'IResultsExporter',
    'IVisualizationService',
    
    # State Management
    'IStateManager',
    'INavigationManager',
    'ISessionManager',
    'IConfigManager',
    'StateScope'
]


# Documentation des responsabilités
INTERFACE_DOCUMENTATION = {
    'IPipelineManager': 'Gestion générique des pipelines ML',
    'ISklearnPipelineManager': 'Spécialisation pour pipelines Sklearn',
    'ITensorFlowPipelineManager': 'Spécialisation pour pipelines TensorFlow',
    'IDataAugmentationManager': 'Gestion de l\'augmentation de données',
    
    'IDataLoader': 'Chargement générique des données',
    'ICovidDataLoader': 'Spécialisation pour dataset COVID-19',
    'IDataValidator': 'Validation et contrôle qualité des données',
    'ICacheManager': 'Gestion du cache des données lourdes',
    
    'IModelValidator': 'Validation des performances des modèles',
    'IMetricsCalculator': 'Calcul des métriques ML',
    'IResultsExporter': 'Export des résultats vers différents formats',
    'IVisualizationService': 'Génération de graphiques et visualisations',
    
    'IStateManager': 'Gestion centralisée de l\'état application',
    'INavigationManager': 'Gestion de la navigation entre pages',
    'ISessionManager': 'Gestion des sessions utilisateur',
    'IConfigManager': 'Gestion de la configuration application'
}
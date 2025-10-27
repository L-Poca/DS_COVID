"""
Package des widgets Streamlit modulaires pour l'application COVID-19.

Ce package contient tous les widgets réutilisables pour les pages de l'application :
- W_Configuration_Commune : Widgets de configuration partagés
- W_Training : Widgets pour la page d'entraînement
- W_Evaluation : Widgets pour la page d'évaluation
- W_Prediction : Widgets pour la page de prédiction
- W_Vérifications_Front : Widgets de vérification
"""

# Imports des widgets principaux
from .W_Configuration_Commune import (
    create_data_source_config,
    create_general_parameters,
    create_pipeline_type_selector,
    create_model_selection_widget,
    display_data_statistics,
    create_training_options,
    create_tensorflow_specific_params,
    show_configuration_summary
)

from .W_Training import (
    create_pipeline_configuration_tab,
    create_training_tab,
    create_results_comparison_tab,
    create_data_verification_widget,
    save_training_results_to_session
)

from .W_Evaluation import (
    create_model_selection_tab,
    create_detailed_metrics_tab,
    create_visualizations_tab,
    create_evaluation_report_tab
)

from .W_Prediction import (
    create_prediction_config_sidebar,
    create_simple_prediction_tab,
    create_batch_prediction_tab,
    create_prediction_analysis_tab
)

from .W_Vérifications_Front import (
    show_global_status
)

__all__ = [
    # Configuration commune
    'create_data_source_config',
    'create_general_parameters', 
    'create_pipeline_type_selector',
    'create_model_selection_widget',
    'display_data_statistics',
    'create_training_options',
    'create_tensorflow_specific_params',
    'show_configuration_summary',
    
    # Training widgets
    'create_pipeline_configuration_tab',
    'create_training_tab',
    'create_results_comparison_tab',
    'create_data_verification_widget',
    'save_training_results_to_session',
    
    # Evaluation widgets
    'create_model_selection_tab',
    'create_detailed_metrics_tab',
    'create_visualizations_tab', 
    'create_evaluation_report_tab',
    
    # Prediction widgets
    'create_prediction_config_sidebar',
    'create_simple_prediction_tab',
    'create_batch_prediction_tab',
    'create_prediction_analysis_tab',
    
    # Verification widgets
    'show_global_status'
]
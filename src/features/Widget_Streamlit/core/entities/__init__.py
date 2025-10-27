"""
Package des entités métier.
Définit les modèles de données du domaine métier.
"""

from .training_config import (
    PipelineType,
    DataSource,
    DataConfig,
    GeneralParams,
    TensorFlowParams,
    TrainingConfig,
    ValidationConfig,
    dict_to_data_config,
    dict_to_general_params,
    dict_to_tensorflow_params
)

from .model_result import (
    ModelStatus,
    ResultType,
    TrainingResult,
    EvaluationResult,
    PredictionResult,
    BatchPredictionResult,
    ComparisonData,
    create_training_result_from_dict
)

__all__ = [
    # Enums
    'PipelineType',
    'DataSource', 
    'ModelStatus',
    'ResultType',
    
    # Configuration entities
    'DataConfig',
    'GeneralParams',
    'TensorFlowParams',
    'TrainingConfig',
    'ValidationConfig',
    
    # Result entities
    'TrainingResult',
    'EvaluationResult',
    'PredictionResult',
    'BatchPredictionResult',
    'ComparisonData',
    
    # Utility functions
    'dict_to_data_config',
    'dict_to_general_params',
    'dict_to_tensorflow_params',
    'create_training_result_from_dict'
]
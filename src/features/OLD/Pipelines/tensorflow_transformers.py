"""
Module de compatibilité pour les transformateurs TensorFlow.

ATTENTION: Ce fichier est maintenant divisé en plusieurs modules spécialisés.
Utilisez les imports directs depuis les modules spécifiques ou depuis __init__.py

Structure modulaire:
- tensorflow_feature_extractor.py: TensorFlowFeatureExtractor
- keras_classifiers.py: KerasClassifier, TransferLearningClassifier  
- tensorflow_data_augmenter.py: TensorFlowDataAugmenter
- model_builders.py: fonctions de création de modèles
"""

# Imports pour compatibilité descendante
from ...Pipelines.Transformateurs.tensorflow_feature_extractor import TensorFlowFeatureExtractor
from ...Pipelines.Transformateurs.keras_classifiers import KerasClassifier, TransferLearningClassifier
from ...Pipelines.Transformateurs.tensorflow_data_augmenter import TensorFlowDataAugmenter
from ...Pipelines.Transformateurs.model_builders import (
    create_simple_cnn,
    create_advanced_cnn,
    create_lightweight_cnn
)

# Pour maintenir la compatibilité avec l'ancien code
__all__ = [
    'TensorFlowFeatureExtractor',
    'KerasClassifier',
    'TransferLearningClassifier', 
    'TensorFlowDataAugmenter',
    'create_simple_cnn',
    'create_advanced_cnn',
    'create_lightweight_cnn'
]

# Message informatif pour les utilisateurs
print("ATTENTION: tensorflow_transformers.py a été refactorisé en modules séparés.")
print("Utilisez les imports depuis les modules spécifiques ou depuis __init__.py")
print("Les imports actuels fonctionnent toujours pour la compatibilité.")
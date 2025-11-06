"""
Module principal d'importation des transformateurs TensorFlow.

Ce module centralise l'importation de tous les transformateurs TensorFlow
pour faciliter leur utilisation dans les pipelines.
"""

# Importation des différents transformateurs
from .tensorflow_feature_extractor import TensorFlowFeatureExtractor
from .keras_classifiers import KerasClassifier, TransferLearningClassifier
from .tensorflow_data_augmenter import TensorFlowDataAugmenter
from .model_builders import (
    create_simple_cnn,
    create_advanced_cnn,
    create_lightweight_cnn
)

# Liste des transformateurs disponibles pour l'export
__all__ = [
    'TensorFlowFeatureExtractor',
    'KerasClassifier', 
    'TransferLearningClassifier',
    'TensorFlowDataAugmenter',
    'create_simple_cnn',
    'create_advanced_cnn',
    'create_lightweight_cnn'
]

# Vérification de la disponibilité de TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_VERSION = tf.__version__
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_VERSION = None
    TENSORFLOW_AVAILABLE = False

# Information sur le module
__version__ = '1.0.0'
__author__ = 'Data Pipeline Team'
__description__ = 'Transformateurs TensorFlow/Keras pour pipelines scikit-learn'
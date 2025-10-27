"""
Extracteur de caractéristiques TensorFlow/Keras pour pipelines scikit-learn.

Ce module fournit un transformateur qui utilise des modèles pré-entraînés
TensorFlow/Keras pour extraire des caractéristiques d'images.
"""

from typing import List, Tuple
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications import (
        VGG16, VGG19, ResNet50, ResNet101, ResNet152,
        InceptionV3, InceptionResNetV2, DenseNet121,
        MobileNet, MobileNetV2, EfficientNetB0
    )
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None


class TensorFlowFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracteur de caractéristiques utilisant des modèles pré-entraînés TensorFlow/Keras.

    Utilise des modèles pré-entraînés (VGG, ResNet, etc.) pour extraire des
    caractéristiques d'images qui peuvent ensuite être utilisées par des
    classificateurs scikit-learn.
    """

    def __init__(self, model_name: str = 'VGG16', include_top: bool = False,
                 pooling: str = 'avg',
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 trainable: bool = False,
                 random_state=None):
        """
        Initialise l'extracteur de caractéristiques.

        Args:
            model_name (str): Nom du modèle pré-entraîné à utiliser
            include_top (bool): Inclure les couches de classification finales
            pooling (str): Type de pooling global ('avg', 'max', None)
            input_shape (Tuple): Forme des images d'entrée (height, width, channels)
            trainable (bool): Si les poids du modèle sont entraînables
            random_state (int, None): État aléatoire pour la reproductibilité

        Raises:
            ImportError: Si TensorFlow n'est pas installé
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow n'est pas installé. Installez-le avec: pip install tensorflow"
            )

        self.model_name = model_name
        self.include_top = include_top
        self.pooling = pooling
        self.input_shape = input_shape
        self.trainable = trainable
        self.model = None
        self.random_state = random_state

    def _get_pretrained_model(self):
        """Récupère le modèle pré-entraîné selon le nom spécifié."""
        models_dict = {
            'VGG16': VGG16,
            'VGG19': VGG19,
            'ResNet50': ResNet50,
            'ResNet101': ResNet101,
            'ResNet152': ResNet152,
            'InceptionV3': InceptionV3,
            'InceptionResNetV2': InceptionResNetV2,
            'DenseNet121': DenseNet121,
            'MobileNet': MobileNet,
            'MobileNetV2': MobileNetV2,
            'EfficientNetB0': EfficientNetB0
        }

        if self.model_name not in models_dict:
            raise ValueError(f"Modèle '{self.model_name}' non supporté. "
                           f"Modèles disponibles: {list(models_dict.keys())}")

        model_class = models_dict[self.model_name]
        return model_class(
            weights='imagenet',
            include_top=self.include_top,
            pooling=self.pooling,
            input_shape=self.input_shape
        )

    def fit(self, X, y=None):
        """
        Initialise le modèle d'extraction de caractéristiques.

        Args:
            X: Données d'entrée (non utilisées pour l'extraction de caractéristiques)
            y: Labels (optionnel)

        Returns:
            self: Instance de l'extracteur
        """
        # Créer le modèle pré-entraîné
        self.model = self._get_pretrained_model()

        # Configurer l'entraînabilité
        self.model.trainable = self.trainable

        return self

    def transform(self, X):
        """
        Extrait les caractéristiques des images.

        Args:
            X: Liste de chemins d'images, images PIL, ou array d'images

        Returns:
            np.array: Caractéristiques extraites
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été initialisé. Appelez fit() d'abord.")

        # Charger et préprocesser les images
        if isinstance(X[0], str):
            # X contient des chemins d'images
            images = self._load_images_from_paths(X)
        elif hasattr(X[0], 'resize'):  # Images PIL
            # Convertir les images PIL en arrays numpy
            images = self._convert_pil_to_array(X)
        else:
            # X contient déjà des données d'images
            images = np.array(X)

        # Préprocesser selon le modèle
        images = self._preprocess_images(images)

        # Extraire les caractéristiques
        features = self.model.predict(images, verbose=0)

        # Aplatir si nécessaire
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)

        return features

    def _load_images_from_paths(self, image_paths: List[str]) -> np.ndarray:
        """Charge les images depuis les chemins."""
        images = []
        for path in image_paths:
            # Charger l'image
            if self.input_shape[2] == 1:  # Niveaux de gris
                img = keras.preprocessing.image.load_img(
                    path, target_size=self.input_shape[:2], color_mode='grayscale'
                )
            else:  # RGB
                img = keras.preprocessing.image.load_img(
                    path, target_size=self.input_shape[:2]
                )
            # Convertir en array
            img_array = keras.preprocessing.image.img_to_array(img)
            images.append(img_array)

        return np.array(images)

    def _convert_pil_to_array(self, pil_images) -> np.ndarray:
        """Convertit une liste d'images PIL en array numpy."""
        images = []
        for pil_img in pil_images:
            # Redimensionner si nécessaire
            img_resized = pil_img.resize(self.input_shape[:2])

            # Convertir en array numpy
            img_array = np.array(img_resized)

            # Ajouter une dimension de canal si nécessaire
            if len(img_array.shape) == 2:  # Image en niveaux de gris
                img_array = np.expand_dims(img_array, axis=-1)

            # Vérifier que la forme correspond à input_shape
            if img_array.shape != self.input_shape:
                # Ajuster le nombre de canaux si nécessaire
                if self.input_shape[2] == 3 and img_array.shape[2] == 1:
                    # Convertir niveaux de gris vers RGB
                    img_array = np.repeat(img_array, 3, axis=-1)
                elif self.input_shape[2] == 1 and len(img_array.shape) == 3:
                    # Convertir RGB vers niveaux de gris
                    img_array = np.mean(img_array, axis=-1, keepdims=True)

            images.append(img_array)

        return np.array(images)

    def _preprocess_images(self, images: np.ndarray) -> np.ndarray:
        """Préprocesse les images selon le modèle utilisé."""
        # Redimensionner si nécessaire
        if images.shape[1:3] != self.input_shape[:2]:
            images = tf.image.resize(images, self.input_shape[:2])

        # Préprocessement spécifique au modèle
        if self.model_name.startswith('VGG'):
            from tensorflow.keras.applications.vgg16 import preprocess_input
        elif self.model_name.startswith('ResNet'):
            from tensorflow.keras.applications.resnet50 import preprocess_input
        elif self.model_name.startswith('Inception'):
            from tensorflow.keras.applications.inception_v3 import preprocess_input
        elif self.model_name.startswith('DenseNet'):
            from tensorflow.keras.applications.densenet import preprocess_input
        elif self.model_name.startswith('MobileNet'):
            from tensorflow.keras.applications.mobilenet import preprocess_input
        elif self.model_name.startswith('EfficientNet'):
            from tensorflow.keras.applications.efficientnet import preprocess_input
        else:
            # Préprocessement par défaut
            def preprocess_input(x):
                return x / 255.0

        return preprocess_input(images)
"""
Classificateurs Keras pour pipelines scikit-learn.

Ce module fournit des classificateurs Keras compatibles avec l'API scikit-learn
pour l'intégration dans des pipelines de machine learning.
"""

from typing import List, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import (
        VGG16, VGG19, ResNet50, ResNet101,
        InceptionV3, DenseNet121, MobileNetV2, EfficientNetB0
    )
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None


class KerasClassifier(BaseEstimator, ClassifierMixin):
    """
    Classificateur Keras compatible avec scikit-learn.

    Encapsule un modèle Keras pour être utilisé dans des pipelines scikit-learn,
    avec une API compatible (fit, predict, predict_proba).
    """

    def __init__(self, model_builder=None, epochs: int = 100, batch_size: int = 32,
                 validation_split: float = 0.2, verbose: int = 1,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: Optional[int] = None, random_state=None, **keras_params):
        """
        Initialise le classificateur Keras.

        Args:
            model_builder: Fonction qui construit le modèle Keras
            epochs (int): Nombre d'époques d'entraînement
            batch_size (int): Taille des lots
            validation_split (float): Proportion de données pour la validation
            verbose (int): Niveau de verbosité
            input_shape (Tuple): Forme des données d'entrée
            num_classes (int): Nombre de classes (détecté automatiquement si None)
            random_state: État aléatoire pour la reproductibilité
            **keras_params: Paramètres additionnels pour le modèle
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow n'est pas installé. "
                            "Installez-le avec: pip install tensorflow")

        self.model_builder = model_builder or self._default_model_builder
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.random_state = random_state
        self.keras_params = keras_params

        self.model = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def _default_model_builder(self, input_shape: Tuple[int, int, int],
                              num_classes: int) -> keras.Model:
        """Construit un modèle CNN par défaut."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def fit(self, X, y):
        """
        Entraîne le modèle Keras.

        Args:
            X: Données d'entrée (chemins d'images ou arrays)
            y: Labels

        Returns:
            self: Instance du classificateur
        """
        # Encoder les labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_

        # Déterminer le nombre de classes
        if self.num_classes is None:
            self.num_classes = len(self.classes_)

        # Charger et préprocesser les données
        if isinstance(X[0], str):
            X_processed = self._load_and_preprocess_images(X)
        else:
            X_processed = np.array(X)

        # Construire le modèle
        self.model = self.model_builder(self.input_shape, self.num_classes, **self.keras_params)

        # Entraîner le modèle
        self.model.fit(
            X_processed, y_encoded,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=self.verbose
        )

        return self

    def predict(self, X):
        """
        Prédit les classes pour les données d'entrée.

        Args:
            X: Données d'entrée

        Returns:
            np.array: Prédictions de classe
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")

        # Préprocesser les données
        if isinstance(X[0], str):
            X_processed = self._load_and_preprocess_images(X)
        else:
            X_processed = np.array(X)

        # Prédire
        predictions = self.model.predict(X_processed, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        # Décoder les labels
        return self.label_encoder.inverse_transform(predicted_classes)

    def predict_proba(self, X):
        """
        Prédit les probabilités de classe pour les données d'entrée.

        Args:
            X: Données d'entrée

        Returns:
            np.array: Probabilités de classe
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")

        # Préprocesser les données
        if isinstance(X[0], str):
            X_processed = self._load_and_preprocess_images(X)
        else:
            X_processed = np.array(X)

        # Prédire les probabilités
        return self.model.predict(X_processed, verbose=0)

    def _load_and_preprocess_images(self, image_paths: List[str]) -> np.ndarray:
        """Charge et préprocesse les images depuis les chemins."""
        images = []
        for path in image_paths:
            # Charger l'image
            img = keras.preprocessing.image.load_img(
                path, target_size=self.input_shape[:2]
            )
            # Convertir en array et normaliser
            img_array = keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img_array)

        return np.array(images)


class TransferLearningClassifier(KerasClassifier):
    """
    Classificateur utilisant le transfer learning avec des modèles pré-entraînés.

    Utilise un modèle pré-entraîné comme base et ajoute des couches de classification
    personnalisées pour la tâche spécifique.
    """

    def __init__(self, base_model_name: str = 'VGG16', freeze_base: bool = True,
                 dense_layers: List[int] = [128], dropout_rate: float = 0.5,
                 **kwargs):
        """
        Initialise le classificateur de transfer learning.

        Args:
            base_model_name (str): Nom du modèle de base pré-entraîné
            freeze_base (bool): Geler les poids du modèle de base
            dense_layers (List[int]): Tailles des couches denses ajoutées
            dropout_rate (float): Taux de dropout
            **kwargs: Arguments pour la classe parent
        """
        self.base_model_name = base_model_name
        self.freeze_base = freeze_base
        self.dense_layers = dense_layers
        self.dropout_rate = dropout_rate

        # Créer le model_builder personnalisé
        kwargs['model_builder'] = self._build_transfer_model

        super().__init__(**kwargs)

    def _build_transfer_model(self, input_shape: Tuple[int, int, int],
                             num_classes: int, **kwargs) -> keras.Model:
        """Construit un modèle de transfer learning."""
        # Modèles disponibles
        models_dict = {
            'VGG16': VGG16,
            'VGG19': VGG19,
            'ResNet50': ResNet50,
            'ResNet101': ResNet101,
            'InceptionV3': InceptionV3,
            'DenseNet121': DenseNet121,
            'MobileNetV2': MobileNetV2,
            'EfficientNetB0': EfficientNetB0
        }

        if self.base_model_name not in models_dict:
            raise ValueError(f"Modèle de base '{self.base_model_name}' non supporté.")

        # Créer le modèle de base
        base_model = models_dict[self.base_model_name](
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )

        # Geler ou non les poids
        base_model.trainable = not self.freeze_base

        # Construire le modèle complet
        model = models.Sequential([base_model])
        model.add(layers.GlobalAveragePooling2D())

        # Ajouter les couches denses
        for dense_size in self.dense_layers:
            model.add(layers.Dense(dense_size, activation='relu'))
            model.add(layers.Dropout(self.dropout_rate))

        # Couche de sortie
        model.add(layers.Dense(num_classes, activation='softmax'))

        # Compiler le modèle
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
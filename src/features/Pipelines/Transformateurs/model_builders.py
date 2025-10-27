"""
Fonctions utilitaires pour la création de modèles TensorFlow/Keras.

Ce module contient des fonctions pour créer rapidement des modèles CNN
de différentes complexités.
"""

from typing import Tuple

try:
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    keras = None


def create_simple_cnn(input_shape: Tuple[int, int, int], num_classes: int):
    """
    Crée un CNN simple pour la classification d'images.
    
    Args:
        input_shape (Tuple): Forme des images d'entrée (height, width, channels)
        num_classes (int): Nombre de classes de sortie
        
    Returns:
        keras.Model: Modèle CNN compilé
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow n'est pas installé. "
                        "Installez-le avec: pip install tensorflow")
    
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


def create_advanced_cnn(input_shape: Tuple[int, int, int], num_classes: int):
    """
    Crée un CNN plus avancé avec batch normalization et plus de couches.
    
    Args:
        input_shape (Tuple): Forme des images d'entrée (height, width, channels)
        num_classes (int): Nombre de classes de sortie
        
    Returns:
        keras.Model: Modèle CNN avancé compilé
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow n'est pas installé. "
                        "Installez-le avec: pip install tensorflow")
    
    model = models.Sequential([
        # Premier bloc conv
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Deuxième bloc conv
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Troisième bloc conv
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Couches denses
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_lightweight_cnn(input_shape: Tuple[int, int, int], num_classes: int):
    """
    Crée un CNN léger pour des ressources limitées.
    
    Args:
        input_shape (Tuple): Forme des images d'entrée (height, width, channels)
        num_classes (int): Nombre de classes de sortie
        
    Returns:
        keras.Model: Modèle CNN léger compilé
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow n'est pas installé. "
                        "Installez-le avec: pip install tensorflow")
    
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
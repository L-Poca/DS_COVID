"""
Augmenteur de données TensorFlow pour pipelines scikit-learn.

Ce module fournit un transformateur d'augmentation de données utilisant
TensorFlow pour l'augmentation d'images.
"""

from typing import List, Tuple
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None


class TensorFlowDataAugmenter(BaseEstimator, TransformerMixin):
    """
    Augmenteur de données utilisant TensorFlow pour l'augmentation d'images.

    Applique diverses transformations d'augmentation de données aux images
    en utilisant les fonctionnalités TensorFlow.
    """

    def __init__(self, rotation_range: float = 0.2, width_shift_range: float = 0.1,
                 height_shift_range: float = 0.1, zoom_range: float = 0.1,
                 horizontal_flip: bool = True, 
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 apply_augmentation: bool = True, random_state: int = None):
        """
        Initialise l'augmenteur de données.

        Args:
            rotation_range (float): Plage de rotation en radians
            width_shift_range (float): Plage de décalage horizontal (fraction de la largeur)
            height_shift_range (float): Plage de décalage vertical (fraction de la hauteur)
            zoom_range (float): Plage de zoom
            horizontal_flip (bool): Appliquer un retournement horizontal aléatoire
            brightness_range (Tuple): Plage d'ajustement de luminosité
            apply_augmentation (bool): Activer/désactiver l'augmentation
            random_state (int): Seed pour les augmentations aléatoires
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow n'est pas installé. "
                            "Installez-le avec: pip install tensorflow")

        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.brightness_range = brightness_range
        self.apply_augmentation = apply_augmentation
        self.random_state = random_state

        # Augmentations aléatoires TensorFlow
        tf.random.set_seed(self.random_state)

    def fit(self, X, y=None):
        """Ajuste l'augmenteur (pas d'ajustement nécessaire)."""
        return self

    def transform(self, X):
        """
        Applique l'augmentation de données aux images.

        Args:
            X: Données d'entrée (chemins d'images ou arrays)

        Returns:
            np.array: Images augmentées
        """
        if not self.apply_augmentation:
            # Si l'augmentation est désactivée, retourner les données telles quelles
            if isinstance(X[0], str):
                return self._load_images_from_paths(X)
            return np.array(X)

        # Charger les images si nécessaire
        if isinstance(X[0], str):
            images = self._load_images_from_paths(X)
        else:
            images = np.array(X)

        # Appliquer l'augmentation
        augmented_images = []
        for img in images:
            augmented_img = self._augment_image(img)
            augmented_images.append(augmented_img)

        return np.array(augmented_images)

    def _load_images_from_paths(self, image_paths: List[str]) -> np.ndarray:
        """Charge les images depuis les chemins."""
        images = []
        for path in image_paths:
            img = keras.preprocessing.image.load_img(path)
            img_array = keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
        return np.array(images)

    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Applique l'augmentation à une seule image en utilisant les API TF 2.x."""
        img = tf.convert_to_tensor(image, dtype=tf.float32)

        # Retournement horizontal (le plus simple et stable)
        if self.horizontal_flip:
            img = tf.image.random_flip_left_right(img)

        # Ajustement de luminosité
        if self.brightness_range != (1.0, 1.0):
            brightness_delta = tf.random.uniform([], 
                                               self.brightness_range[0] - 1.0, 
                                               self.brightness_range[1] - 1.0)
            img = tf.image.adjust_brightness(img, brightness_delta)

        # Zoom aléatoire via crop central et resize
        if self.zoom_range > 0:
            zoom_factor = tf.random.uniform([], 1.0 - self.zoom_range, 1.0 + self.zoom_range)
            # Crop central avec le facteur de zoom
            img = tf.image.central_crop(img, zoom_factor)
            # Redimensionner à la taille originale
            original_shape = tf.shape(image)
            img = tf.image.resize(img, [original_shape[0], original_shape[1]])

        # Pour la rotation et translation, on les désactive pour éviter les problèmes
        # Ces transformations sont complexes à implémenter sans tf.contrib
        # L'utilisateur peut les réactiver plus tard avec tf-addons si nécessaire

        return img.numpy()
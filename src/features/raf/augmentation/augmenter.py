# =================================
# CUSTOM IMAGE AUGMENTER
# =================================
"""
Classe d'augmentation d'images personnalisée qui remplace ImageDataGenerator
pour éviter les problèmes d'images noires et offrir plus de contrôle.
"""

import numpy as np
import cv2
import random
from typing import Tuple, Optional


class CustomImageAugmenter:
    """
    Augmentation d'images personnalisée, fiable et performante
    
    Remplace ImageDataGenerator de Keras avec des transformations
    contrôlées et prévisibles qui préservent les valeurs [0,1].
    """
    
    def __init__(self, 
                 rotation_range: float = 10,
                 width_shift_range: float = 0.1,
                 height_shift_range: float = 0.1,
                 zoom_range: float = 0.1,
                 horizontal_flip: bool = True,
                 brightness_range: Tuple[float, float] = (0.9, 1.1),
                 noise_factor: float = 0.01,
                 seed: Optional[int] = None):
        """
        Initialise l'augmenteur avec les paramètres spécifiés.
        
        Args:
            rotation_range: Angle de rotation max (degrés)
            width_shift_range: Décalage horizontal max (fraction) 
            height_shift_range: Décalage vertical max (fraction)
            zoom_range: Facteur de zoom max (fraction)
            horizontal_flip: Activer le miroir horizontal
            brightness_range: Plage de variation de luminosité (min, max)
            noise_factor: Intensité du bruit gaussien
            seed: Seed pour reproductibilité
        """
        
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.brightness_range = brightness_range
        self.noise_factor = noise_factor
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applique une augmentation aléatoire complète à une image.
        
        Args:
            image: Image d'entrée (H, W, C) en float32 [0,1]
            
        Returns:
            Image augmentée (H, W, C) en float32 [0,1]
        """
        
        # S'assurer que l'image est en float32 et [0,1]
        img = self._normalize_image(image)
        height, width = img.shape[:2]
        
        # 1. Rotation aléatoire
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            img = self._rotate_image(img, angle)
        
        # 2. Décalage horizontal/vertical
        if self.width_shift_range > 0 or self.height_shift_range > 0:
            dx = random.uniform(-self.width_shift_range, self.width_shift_range) * width
            dy = random.uniform(-self.height_shift_range, self.height_shift_range) * height
            img = self._shift_image(img, dx, dy)
        
        # 3. Zoom aléatoire
        if self.zoom_range > 0:
            zoom_factor = random.uniform(1-self.zoom_range, 1+self.zoom_range)
            img = self._zoom_image(img, zoom_factor)
        
        # 4. Miroir horizontal
        if self.horizontal_flip and random.random() > 0.5:
            img = np.fliplr(img)
        
        # 5. Variation de luminosité
        if self.brightness_range != (1.0, 1.0):
            brightness_factor = random.uniform(*self.brightness_range)
            img = img * brightness_factor
        
        # 6. Bruit gaussien léger
        if self.noise_factor > 0:
            noise = np.random.normal(0, self.noise_factor, img.shape)
            img = img + noise
        
        # Clamp final dans [0,1]
        img = np.clip(img, 0.0, 1.0)
        
        return img.astype(np.float32)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalise l'image en [0,1] si nécessaire"""
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        return img
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotation avec préservation de la taille"""
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    
    def _shift_image(self, image: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Décalage avec remplissage par les bords"""
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    
    def _zoom_image(self, image: np.ndarray, zoom_factor: float) -> np.ndarray:
        """Zoom avec redimensionnement intelligent"""
        height, width = image.shape[:2]
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
        
        # Redimensionner
        resized = cv2.resize(image, (new_width, new_height))
        
        if zoom_factor > 1:  # Zoom in - crop center
            start_x = (new_width - width) // 2
            start_y = (new_height - height) // 2
            return resized[start_y:start_y+height, start_x:start_x+width]
        else:  # Zoom out - pad avec des zéros
            result = np.zeros_like(image)
            start_x = (width - new_width) // 2  
            start_y = (height - new_height) // 2
            result[start_y:start_y+new_height, start_x:start_x+new_width] = resized
            return result
    
    def generate_batch(self, images: np.ndarray, labels: np.ndarray, 
                      batch_size: int = 32, shuffle: bool = True):
        """
        Générateur de batches augmentés pour l'entraînement.
        
        Args:
            images: Array d'images [N, H, W, C]
            labels: Array de labels [N,] ou [N, num_classes]
            batch_size: Taille des batches
            shuffle: Mélanger les données
            
        Yields:
            Tuple (batch_images, batch_labels) augmentés
        """
        indices = np.arange(len(images))
        
        while True:
            # Mélanger les indices si demandé
            if shuffle:
                np.random.shuffle(indices)
            
            for start in range(0, len(indices), batch_size):
                end = min(start + batch_size, len(indices))
                batch_indices = indices[start:end]
                
                # Créer le batch augmenté
                batch_images = []
                batch_labels = []
                
                for idx in batch_indices:
                    # Augmenter l'image
                    augmented_img = self.augment_image(images[idx])
                    batch_images.append(augmented_img)
                    batch_labels.append(labels[idx])
                
                yield np.array(batch_images), np.array(batch_labels)
    
    def preview_augmentations(self, image: np.ndarray, n_samples: int = 6) -> list:
        """
        Génère plusieurs augmentations d'une image pour prévisualisation.
        
        Args:
            image: Image d'origine
            n_samples: Nombre d'augmentations à générer
            
        Returns:
            Liste des images augmentées
        """
        augmented_images = []
        for _ in range(n_samples):
            aug_img = self.augment_image(image)
            augmented_images.append(aug_img)
        
        return augmented_images
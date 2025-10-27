"""Image augmentation transformers for the data pipeline."""

import random
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin


class ImageAugmenter(BaseEstimator, TransformerMixin):
    """Applique une augmentation simple (flip horizontal) à la moitié des images."""

    def __init__(self, rotation_range=10, zoom_range=0.1, random_state=None):
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.random_state = random_state
        
    def fit(self, X, y=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        return self

    def transform(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Transform images by applying horizontal flip to 50% of them.
        
        Args:
            data_x: Array of images to augment
            data_y: Target data (unused)
            
        Returns:
            np.ndarray: Augmented images
        """
        print(f"\nAugmentation de {len(data_x)} images (flip horizontal sur 50%) ...")
        data_aug = []
        for i, img in enumerate(data_x):
            if i % 2 == 0:
                data_aug.append(np.fliplr(img))
            else:
                data_aug.append(img)
        print("Augmentation terminée.\n")
        return np.array(data_aug)


class ImageRandomCropper(BaseEstimator, TransformerMixin):
    """Effectue un crop aléatoire sur chaque image (carré centré ou décalé)."""

    def __init__(self, crop_size=(224, 224), random_state=None):
        self.crop_size = crop_size
        self.random_state = random_state
        
    def fit(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Fit the transformer (no-op for random cropping).
        
        Args:
            data_x: Input data (unused)
            data_y: Target data (unused)
            
        Returns:
            self: Returns self for method chaining
        """
        return self

    def transform(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Transform images by applying random cropping.
        
        Args:
            data_x: Array of images to crop
            data_y: Target data (unused)
            
        Returns:
            np.ndarray: Randomly cropped images
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        cropped = []
        for img in tqdm(data_x, desc="RandomCrop"):
            height, width = img.shape[:2]
            crop_height, crop_width = self.crop_size
            if height < crop_height or width < crop_width:
                cropped.append(img)
                continue
            top = random.randint(0, height - crop_height)
            left = random.randint(0, width - crop_width)
            cropped.append(img[top:top+crop_height, left:left+crop_width])
        print(f"Random crop terminé. Shape: {cropped[0].shape if cropped else None}")
        return np.array(cropped)

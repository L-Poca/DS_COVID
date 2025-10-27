"""Utility transformers for visualization and saving features."""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin


class VisualizeTransformer(BaseEstimator, TransformerMixin):
    """Transformer de visualisation pour afficher des échantillons d'images."""

    def __init__(self, n_samples=5, prefix="step", save_dir=None):
        self.n_samples = n_samples
        self.prefix = prefix
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def fit(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Fit the transformer (no-op for visualization).
        
        Args:
            data_x: Input data (unused)
            data_y: Target data (unused)
            
        Returns:
            self: Returns self for method chaining
        """
        return self

    def transform(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Transform data by visualizing sample images.
        
        Args:
            data_x: Array of images to visualize
            data_y: Target data (unused)
            
        Returns:
            np.ndarray: Input data passed through unchanged
        """
        for i in range(min(self.n_samples, len(data_x))):
            plt.figure()
            title = f"{self.prefix}_sample_{i}"
            if data_x[i].ndim == 2:
                plt.imshow(data_x[i], cmap='gray')
            else:
                plt.imshow(data_x[i])
            plt.title(title)
            plt.axis('off')
            if self.save_dir:
                path = os.path.join(self.save_dir, f"{self.prefix}_sample_{i}.png")
                plt.savefig(path, bbox_inches='tight')
                print(f"Image sauvegardée : {path}")
            plt.show()
            plt.close()
        return data_x


class SaveTransformer(BaseEstimator, TransformerMixin):
    """Transformer pour sauvegarder les features extraites."""

    def __init__(self, save_dir="outputs", prefix="features"):
        self.save_dir = save_dir
        self.prefix = prefix
        os.makedirs(self.save_dir, exist_ok=True)

    def fit(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Fit the transformer (no-op for saving).
        
        Args:
            data_x: Input data (unused)
            data_y: Target data (unused)
            
        Returns:
            self: Returns self for method chaining
        """
        return self

    def transform(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Transform data by saving features to disk.
        
        Args:
            data_x: Array of features to save
            data_y: Target data (unused)
            
        Returns:
            np.ndarray: Input data passed through unchanged
        """
        path = os.path.join(self.save_dir, f"{self.prefix}.npy")
        np.save(path, data_x)
        print(f"Features sauvegardées dans {path}")
        return data_x

"""Image feature extraction transformers for the data pipeline."""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.features.Pipelines.Visualisations.Visu_PCA import *


class ImageHistogram(BaseEstimator, TransformerMixin):
    """Calcule l'histogramme d'intensité pour chaque image."""

    def __init__(self, bins=32):
        self.bins = bins

    def fit(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Fit the transformer (no-op for histogram computation).
        
        Args:
            data_x: Input data (unused)
            data_y: Target data (unused)
            
        Returns:
            self: Returns self for method chaining
        """
        return self

    def transform(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Transform images by computing histograms.
        
        Args:
            data_x: Array of images to transform
            data_y: Target data (unused)
            
        Returns:
            np.ndarray: Array of histogram features
        """
        print(f"Calcul des histogrammes ({self.bins} bins)")
        histos = [np.histogram(img.flatten(), bins=self.bins, range=(0, 1))[0] for img in data_x]
        return np.array(histos)


class ImagePCA(BaseEstimator, TransformerMixin):
    """Réduction de dimension par ACP (PCA) sur les images aplaties."""

    def __init__(self, n_components=50, random_state=None, visualize=True, max_viz_points=1000):
        self.n_components = n_components
        self.random_state = random_state
        self.visualize = visualize
        self.max_viz_points = max_viz_points  # Nouveau paramètre
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self._labels = None  # Stocker les labels si fournis

    def fit(self, data_x, data_y=None):
        """Fit PCA on flattened images.
        
        Args:
            data_x: Array of images to fit PCA on
            data_y: Target labels (optional, pour la visualisation)
            
        Returns:
            self: Returns self for method chaining
        """
        n_samples = data_x.shape[0]
        data_flat = data_x.reshape(n_samples, -1)
        self.pca.fit(data_flat)
        
        # Stocker les labels pour la visualisation
        if data_y is not None:
            self._labels = data_y
        
        return self

    def transform(self, data_x, data_y=None):
        """Transform images using fitted PCA.
        
        Args:
            data_x: Array of images to transform
            data_y: Target labels (optional, pour la visualisation)
            
        Returns:
            np.ndarray: PCA-transformed features
        """
        n_samples = data_x.shape[0]
        data_flat = data_x.reshape(n_samples, -1)
        data_pca = self.pca.transform(data_flat)
        
        if self.visualize:
            # Utiliser les labels fournis lors du transform, ou ceux du fit
            labels_to_use = data_y if data_y is not None else self._labels
            
            afficher_pca(self.pca, data_x, data_flat, data_pca, max_samples=self.max_viz_points)
            create_interactive_pca_plot(
                self.pca, 
                data_x, 
                data_pca, 
                labels=labels_to_use, 
                max_points=self.max_viz_points,
                sampling_method='stratified' if labels_to_use is not None else 'random'
            )
            
        return data_pca

    

class ImageStandardScaler(BaseEstimator, TransformerMixin):
    """Applique un StandardScaler pixel-wise sur les images aplaties."""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Fit StandardScaler on flattened images.
        
        Args:
            data_x: Array of images to fit scaler on
            data_y: Target data (unused)
            
        Returns:
            self: Returns self for method chaining
        """
        n_samples = data_x.shape[0]
        data_flat = data_x.reshape(n_samples, -1)
        self.scaler.fit(data_flat)
        return self

    def transform(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Transform images using fitted StandardScaler.
        
        Args:
            data_x: Array of images to transform
            data_y: Target data (unused)
            
        Returns:
            np.ndarray: Standardized images
        """
        n_samples = data_x.shape[0]
        data_flat = data_x.reshape(n_samples, -1)
        data_scaled = self.scaler.transform(data_flat)
        print(f"\nStandardisation terminée. Shape: {data_scaled.shape}")
        return data_scaled.reshape(data_x.shape)

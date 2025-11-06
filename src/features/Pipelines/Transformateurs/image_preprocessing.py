"""Image preprocessing transformers for the data pipeline."""

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin


class ImageResizer(BaseEstimator, TransformerMixin):
    """Redimensionne les images PIL ou numpy array à la taille souhaitée."""

    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size

    def fit(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Fit the transformer (no-op for resizing).
        
        Args:
            data_x: Input data (unused)
            data_y: Target data (unused)
            
        Returns:
            self: Returns self for method chaining
        """
        return self

    def transform(self, data_x):
        """Transform images by resizing them to target size.
        
        Args:
            data_x: Array of images to resize
            
        Returns:
            np.ndarray: Resized images as numpy arrays
        """
        print(f"\nRedimensionnement de {len(data_x)} images en {self.img_size} ...")
        resized = []
        for img in tqdm(data_x, desc="Redimensionnement"):
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_resized = img.resize(self.img_size)
            resized.append(np.array(img_resized))
        print("Redimensionnement terminé.\n")

        tqdm.write("\nRésolutions Uniques des images après redimensionnement :")
        for res in set(img.shape for img in resized):
            tqdm.write(f" - {res}\n")

        # Afficher quelsques exemples d'images redimensionnées
        n_img = 3
        for i in range(n_img):
            plt.subplot(1, n_img, i + 1)
            plt.imshow(resized[i], cmap='gray') 
            plt.title(f"Image {i+1}")
            plt.axis('off')
        plt.show()

        return np.array(resized)


class ImageNormalizer(BaseEstimator, TransformerMixin):
    """Normalise les images (array) pixel-wise entre 0 et 1."""

    def fit(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Fit the transformer (no-op for normalization).
        
        Args:
            data_x: Input data (unused)
            data_y: Target data (unused)
            
        Returns:
            self: Returns self for method chaining
        """
        return self

    def transform(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Transform images by normalizing pixel values to [0, 1].
        
        Args:
            data_x: Array of images to normalize
            data_y: Target data (unused)
            
        Returns:
            np.ndarray: Normalized images
        """
        print(f"\nNormalisation de {len(data_x)} images ...")
        data_norm = np.array(data_x).astype(np.float32) / 255.0
        print("Normalisation terminée.\n")

        # Afficher quelsques exemples d'images Normalisées
        n_img = 3
        for i in range(n_img):
            plt.subplot(1, n_img, i + 1)
            plt.imshow(data_norm[i], cmap='gray') 
            plt.title(f"Image {i+1}")
            plt.axis('off')
        plt.show()

        return data_norm


class ImageMasker(BaseEstimator, TransformerMixin):
    """Applique des masques sur les images."""

    def __init__(self, mask_paths):
        self.mask_paths = mask_paths

    def fit(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Fit the transformer (no-op for masking).
        
        Args:
            data_x: Input data (unused)
            data_y: Target data (unused)
            
        Returns:
            self: Returns self for method chaining
        """
        return self

    def transform(self, data_x):
        """Transform images by applying masks.
        
        Args:
            data_x: Array of images to mask
            
        Returns:
            np.ndarray: Masked images
        """
        print(f"\nApplication des masques sur {len(data_x)} images ...")
        masked = []
        for img, mask_path in tqdm(
            zip(data_x, self.mask_paths), desc="Masquage", total=len(data_x)
        ):
            mask = Image.open(mask_path).convert('L').resize(img.shape[::-1])
            mask_arr = np.array(mask) > 0  # binaire
            masked.append(img * mask_arr)
        print("Masquage terminé.\n")

        # Afficher quelsques exemples d'images Masquées
        n_img = 3
        for i in range(n_img):
            plt.subplot(1, n_img, i + 1)
            plt.imshow(masked[i], cmap='gray') 
            plt.title(f"Image {i+1}")
            plt.axis('off')
        plt.show()

        return np.array(masked)


class ImageFlattener(BaseEstimator, TransformerMixin):
    """Aplatit les images pour les modèles ML."""

    def fit(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Fit the transformer (no-op for flattening).
        
        Args:
            data_x: Input data (unused)
            data_y: Target data (unused)
            
        Returns:
            self: Returns self for method chaining
        """
        return self

    def transform(self, data_x):
        """Transform images by flattening them to 1D arrays.
        
        Args:
            data_x: Array of images to flatten
            
        Returns:
            np.ndarray: Flattened images
        """
        print(f"\nAplatissement de {data_x.shape[0]} images ...")
        data_flat = []
        for img in tqdm(data_x, desc="Aplatissement"):
            data_flat.append(img.flatten())
        data_flat = np.array(data_flat)
        print("Aplatissement terminé.\n")
        return data_flat


class ImageBinarizer(BaseEstimator, TransformerMixin):
    """Binarise les images (seuil fixe ou automatique)."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Fit the transformer (no-op for binarization).
        
        Args:
            data_x: Input data (unused)
            data_y: Target data (unused)
            
        Returns:
            self: Returns self for method chaining
        """
        return self

    def transform(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Transform images by applying binary thresholding.
        
        Args:
            data_x: Array of images to binarize
            data_y: Target data (unused)
            
        Returns:
            np.ndarray: Binarized images
        """
        print(f"Binarisation avec seuil {self.threshold}")
        return (data_x > self.threshold).astype(np.float32)

class AddChannelDim(BaseEstimator, TransformerMixin):
    """
    Transformateur qui ajoute une dimension de canal aux images 2D (N&B).
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # X peut être une liste ou un array de np.ndarray
        X_out = []
        for img in X:
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)  # (H, W) -> (H, W, 1)
            X_out.append(img)
        return np.array(X_out)
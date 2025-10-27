"""Image loading transformers for the data pipeline."""

from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin


class ImageLoader(BaseEstimator, TransformerMixin):
    """Transformer pour charger et prétraiter les images."""

    def __init__(self):
        pass

    def fit(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Fit the transformer (no-op for image loading).
        
        Args:
            data_x: Input data (unused)
            data_y: Target data (unused)
            
        Returns:
            self: Returns self for method chaining
        """
        return self

    def transform(self, data_x):
        """Transform file paths into loaded PIL images.
        
        Args:
            data_x: List of image file paths
            
        Returns:
            list: List of PIL Image objects
        """
        tqdm.write(f"Chargement de {len(data_x)} images :")
        images = []
        images_resolutions = []

        for path in tqdm(data_x, desc="Images"):
            img = Image.open(path).convert('L')  # PAS de resize ici
            images.append(img)
            images_resolutions.append(img.size)

        tqdm.write("\nRésolutions Uniques des images chargées :")
        for res in set(images_resolutions):
            tqdm.write(f" - {res}")

        tqdm.write("Chargement terminé.\n")

        # Afficher quelsques exemples d'images redimensionnées
        n_img = 3
        for i in range(n_img):
            plt.subplot(1, n_img, i + 1)
            plt.imshow(images[i], cmap='gray') 
            plt.title(f"Image {i+1}")
            plt.axis('off')
        plt.show()

        return images # Liste d'images PIL

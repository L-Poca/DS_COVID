"""
Module d'augmentation de données pour le framework RAF
Gère l'augmentation d'images médicales avec préservation des caractéristiques cliniques
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps
import random
from typing import List, Tuple, Dict, Any, Optional
import warnings

from ..utils.config import Config


class MedicalImageAugmentor:
    """
    Augmenteur spécialisé pour les images médicales
    Applique des transformations conservant les caractéristiques diagnostiques
    """
    
    def __init__(self, preserve_aspect_ratio: bool = True):
        """
        Initialise l'augmenteur
        
        Args:
            preserve_aspect_ratio: Préserver le ratio d'aspect lors des rotations
        """
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.augmentation_stats = {
            'rotation': 0,
            'flip': 0,
            'brightness': 0,
            'contrast': 0,
            'noise': 0,
            'zoom': 0,
            'translation': 0
        }
    
    def rotate_image(self, image: np.ndarray, angle_range: Tuple[float, float] = (-15, 15)) -> np.ndarray:
        """
        Rotation légère pour images médicales
        
        Args:
            image: Image source
            angle_range: Plage d'angles en degrés
            
        Returns:
            Image tournée
        """
        angle = random.uniform(angle_range[0], angle_range[1])
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Matrice de rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        if self.preserve_aspect_ratio:
            # Calcul des nouvelles dimensions pour éviter le cropping
            cos = abs(rotation_matrix[0, 0])
            sin = abs(rotation_matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # Ajustement de la translation
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]
            
            rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                   borderValue=(0, 0, 0))
            
            # Redimensionnement à la taille originale
            rotated = cv2.resize(rotated, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                                   borderValue=(0, 0, 0))
        
        self.augmentation_stats['rotation'] += 1
        return rotated
    
    def flip_image(self, image: np.ndarray, flip_type: str = 'horizontal') -> np.ndarray:
        """
        Retournement d'image (adapté aux radiographies)
        
        Args:
            image: Image source
            flip_type: Type de retournement ('horizontal', 'vertical')
            
        Returns:
            Image retournée
        """
        if flip_type == 'horizontal':
            flipped = cv2.flip(image, 1)
        elif flip_type == 'vertical':
            flipped = cv2.flip(image, 0)
        else:
            # Flip aléatoire
            flip_code = random.choice([0, 1])  # 0: vertical, 1: horizontal
            flipped = cv2.flip(image, flip_code)
        
        self.augmentation_stats['flip'] += 1
        return flipped
    
    def adjust_brightness_contrast(self, image: np.ndarray, 
                                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                                 contrast_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Ajustement de luminosité et contraste
        
        Args:
            image: Image source
            brightness_range: Plage de facteurs de luminosité
            contrast_range: Plage de facteurs de contraste
            
        Returns:
            Image ajustée
        """
        # Conversion en PIL pour utiliser ImageEnhance
        if image.dtype != np.uint8:
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
        else:
            image_pil = Image.fromarray(image)
        
        # Ajustement de luminosité
        brightness_factor = random.uniform(brightness_range[0], brightness_range[1])
        enhancer = ImageEnhance.Brightness(image_pil)
        image_pil = enhancer.enhance(brightness_factor)
        
        # Ajustement de contraste
        contrast_factor = random.uniform(contrast_range[0], contrast_range[1])
        enhancer = ImageEnhance.Contrast(image_pil)
        image_pil = enhancer.enhance(contrast_factor)
        
        # Retour en numpy
        enhanced = np.array(image_pil).astype(np.float32)
        if image.dtype != np.uint8:
            enhanced = enhanced / 255.0
        
        self.augmentation_stats['brightness'] += 1
        self.augmentation_stats['contrast'] += 1
        return enhanced
    
    def add_gaussian_noise(self, image: np.ndarray, 
                          noise_factor: float = 0.05) -> np.ndarray:
        """
        Ajout de bruit gaussien léger
        
        Args:
            image: Image source
            noise_factor: Intensité du bruit (0-1)
            
        Returns:
            Image bruitée
        """
        noise = np.random.normal(0, noise_factor, image.shape)
        noisy_image = image + noise
        
        # Clipping pour maintenir les valeurs dans la plage valide
        if image.max() <= 1.0:
            noisy_image = np.clip(noisy_image, 0, 1)
        else:
            noisy_image = np.clip(noisy_image, 0, 255)
        
        self.augmentation_stats['noise'] += 1
        return noisy_image.astype(image.dtype)
    
    def zoom_image(self, image: np.ndarray, 
                   zoom_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        Zoom léger sur l'image
        
        Args:
            image: Image source
            zoom_range: Plage de facteurs de zoom
            
        Returns:
            Image zoomée
        """
        zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
        h, w = image.shape[:2]
        
        if zoom_factor < 1.0:
            # Zoom out: réduire l'image et ajouter du padding
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Padding pour revenir à la taille originale
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            
            if len(image.shape) == 3:
                zoomed = np.pad(resized, ((pad_h, h - new_h - pad_h), 
                                        (pad_w, w - new_w - pad_w), (0, 0)), 
                              mode='constant', constant_values=0)
            else:
                zoomed = np.pad(resized, ((pad_h, h - new_h - pad_h), 
                                        (pad_w, w - new_w - pad_w)), 
                              mode='constant', constant_values=0)
        else:
            # Zoom in: agrandir l'image et cropper
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Crop central
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            zoomed = resized[start_h:start_h + h, start_w:start_w + w]
        
        self.augmentation_stats['zoom'] += 1
        return zoomed
    
    def translate_image(self, image: np.ndarray, 
                       translation_range: Tuple[float, float] = (-0.1, 0.1)) -> np.ndarray:
        """
        Translation légère de l'image
        
        Args:
            image: Image source
            translation_range: Plage de translation (fraction de la taille)
            
        Returns:
            Image translatée
        """
        h, w = image.shape[:2]
        
        # Calcul des décalages
        dx = random.uniform(translation_range[0], translation_range[1]) * w
        dy = random.uniform(translation_range[0], translation_range[1]) * h
        
        # Matrice de translation
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        
        translated = cv2.warpAffine(image, translation_matrix, (w, h),
                                  borderValue=(0, 0, 0))
        
        self.augmentation_stats['translation'] += 1
        return translated
    
    def apply_random_augmentation(self, image: np.ndarray, 
                                 augmentation_probability: float = 0.5,
                                 max_augmentations: int = 3) -> np.ndarray:
        """
        Applique des augmentations aléatoirement
        
        Args:
            image: Image source
            augmentation_probability: Probabilité d'appliquer chaque augmentation
            max_augmentations: Nombre maximum d'augmentations à appliquer
            
        Returns:
            Image augmentée
        """
        augmented = image.copy()
        applied_count = 0
        
        # Liste des augmentations possibles
        augmentations = [
            ('rotation', lambda img: self.rotate_image(img)),
            ('flip', lambda img: self.flip_image(img) if random.random() < 0.3 else img),  # Moins fréquent
            ('brightness_contrast', lambda img: self.adjust_brightness_contrast(img)),
            ('noise', lambda img: self.add_gaussian_noise(img) if random.random() < 0.2 else img),  # Rare
            ('zoom', lambda img: self.zoom_image(img)),
            ('translation', lambda img: self.translate_image(img))
        ]
        
        # Mélange des augmentations
        random.shuffle(augmentations)
        
        for aug_name, aug_func in augmentations:
            if applied_count >= max_augmentations:
                break
                
            if random.random() < augmentation_probability:
                try:
                    augmented = aug_func(augmented)
                    applied_count += 1
                except Exception as e:
                    warnings.warn(f"Erreur augmentation {aug_name}: {e}")
        
        return augmented
    
    def augment_batch(self, images: np.ndarray, labels: np.ndarray,
                     augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augmente un batch d'images
        
        Args:
            images: Batch d'images
            labels: Labels correspondants
            augmentation_factor: Facteur de multiplication des données
            
        Returns:
            Tuple (images_augmentées, labels_augmentés)
        """
        print(f"🔄 Augmentation de {len(images)} images (facteur: {augmentation_factor})")
        
        augmented_images = [images]  # Commencer avec les images originales
        augmented_labels = [labels]
        
        for i in range(augmentation_factor - 1):
            batch_augmented = []
            
            for img in images:
                aug_img = self.apply_random_augmentation(img)
                batch_augmented.append(aug_img)
            
            augmented_images.append(np.array(batch_augmented))
            augmented_labels.append(labels.copy())
        
        # Concaténation de tous les batches
        final_images = np.concatenate(augmented_images, axis=0)
        final_labels = np.concatenate(augmented_labels, axis=0)
        
        print(f"✅ Augmentation terminée: {final_images.shape[0]} images")
        print(f"📊 Statistiques d'augmentation:")
        for aug_type, count in self.augmentation_stats.items():
            if count > 0:
                print(f"   {aug_type}: {count} applications")
        
        return final_images, final_labels
    
    def reset_stats(self):
        """Remet à zéro les statistiques d'augmentation"""
        for key in self.augmentation_stats:
            self.augmentation_stats[key] = 0


def demonstrate_augmentation_fixed(sample_images: np.ndarray, n_samples: int = 3):
    """
    Démontre les effets d'augmentation sur des échantillons
    
    Args:
        sample_images: Images d'exemple
        n_samples: Nombre d'échantillons à montrer
    """
    import matplotlib.pyplot as plt
    
    augmentor = MedicalImageAugmentor()
    
    # Sélection d'échantillons aléatoirement
    indices = np.random.choice(len(sample_images), min(n_samples, len(sample_images)), replace=False)
    
    fig, axes = plt.subplots(n_samples, 6, figsize=(18, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    augmentation_types = [
        ('Original', lambda x: x),
        ('Rotation', lambda x: augmentor.rotate_image(x)),
        ('Brightness/Contrast', lambda x: augmentor.adjust_brightness_contrast(x)),
        ('Zoom', lambda x: augmentor.zoom_image(x)),
        ('Translation', lambda x: augmentor.translate_image(x)),
        ('Random Mix', lambda x: augmentor.apply_random_augmentation(x, max_augmentations=2))
    ]
    
    for row, idx in enumerate(indices):
        img = sample_images[idx]
        
        for col, (aug_name, aug_func) in enumerate(augmentation_types):
            try:
                if aug_name == 'Original':
                    processed_img = img
                else:
                    processed_img = aug_func(img.copy())
                
                # Affichage
                if len(processed_img.shape) == 3:
                    axes[row, col].imshow(processed_img)
                else:
                    axes[row, col].imshow(processed_img, cmap='gray')
                
                axes[row, col].set_title(f'{aug_name}')
                axes[row, col].axis('off')
                
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Erreur:\n{str(e)[:50]}...', 
                                   ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(f'{aug_name} (Erreur)')
    
    plt.tight_layout()
    plt.suptitle('Démonstration des augmentations d\'images médicales', y=1.02, fontsize=16)
    plt.show()
    
    # Statistiques
    print(f"\n📊 Statistiques d'augmentation:")
    for aug_type, count in augmentor.augmentation_stats.items():
        if count > 0:
            print(f"   {aug_type}: {count} applications")


def balance_dataset_with_augmentation(images: np.ndarray, labels: np.ndarray,
                                    target_samples_per_class: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Équilibre un dataset déséquilibré en utilisant l'augmentation
    
    Args:
        images: Images du dataset
        labels: Labels correspondants
        target_samples_per_class: Nombre cible d'échantillons par classe
        
    Returns:
        Tuple (images_équilibrées, labels_équilibrés)
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    print(f"📊 Distribution originale:")
    for label, count in zip(unique_labels, counts):
        print(f"   Classe {label}: {count} échantillons")
    
    # Déterminer le nombre cible
    if target_samples_per_class is None:
        target_samples_per_class = max(counts)  # Égaliser vers la classe majoritaire
    
    print(f"\n🎯 Cible: {target_samples_per_class} échantillons par classe")
    
    augmentor = MedicalImageAugmentor()
    balanced_images = []
    balanced_labels = []
    
    for label in unique_labels:
        # Images de cette classe
        class_mask = labels == label
        class_images = images[class_mask]
        class_labels = labels[class_mask]
        
        current_count = len(class_images)
        
        if current_count >= target_samples_per_class:
            # Pas besoin d'augmentation, on prend un échantillon aléatoire
            indices = np.random.choice(current_count, target_samples_per_class, replace=False)
            selected_images = class_images[indices]
            selected_labels = class_labels[indices]
        else:
            # Augmentation nécessaire
            needed_samples = target_samples_per_class - current_count
            augmentation_factor = int(np.ceil(needed_samples / current_count)) + 1
            
            print(f"   Classe {label}: augmentation x{augmentation_factor}")
            
            # Augmentation
            aug_images, aug_labels = augmentor.augment_batch(
                class_images, class_labels, augmentation_factor
            )
            
            # Sélection du nombre exact d'échantillons
            indices = np.random.choice(len(aug_images), target_samples_per_class, replace=False)
            selected_images = aug_images[indices]
            selected_labels = aug_labels[indices]
        
        balanced_images.append(selected_images)
        balanced_labels.append(selected_labels)
    
    # Concaténation et mélange
    final_images = np.concatenate(balanced_images, axis=0)
    final_labels = np.concatenate(balanced_labels, axis=0)
    
    # Mélange aléatoire
    shuffle_indices = np.random.permutation(len(final_images))
    final_images = final_images[shuffle_indices]
    final_labels = final_labels[shuffle_indices]
    
    print(f"\n✅ Dataset équilibré: {final_images.shape[0]} échantillons")
    
    # Vérification finale
    unique_final, counts_final = np.unique(final_labels, return_counts=True)
    print(f"📊 Distribution finale:")
    for label, count in zip(unique_final, counts_final):
        print(f"   Classe {label}: {count} échantillons")
    
    return final_images, final_labels


# Import existant pour compatibilité
try:
    from .augmenter import CustomImageAugmenter
    __all__ = ['MedicalImageAugmentor', 'demonstrate_augmentation_fixed', 
               'balance_dataset_with_augmentation', 'CustomImageAugmenter']
except ImportError:
    __all__ = ['MedicalImageAugmentor', 'demonstrate_augmentation_fixed', 
               'balance_dataset_with_augmentation']
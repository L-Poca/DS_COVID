"""
Module de visualisation pour le framework RAF
Gère les visualisations et analyses des données d'images médicales
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional
import warnings

from ..utils.config import Config


def print_system_info():
    """Affiche les informations système et configuration"""
    import platform
    import psutil
    import os
    
    print("=" * 60)
    print("🖥️  INFORMATIONS SYSTÈME")
    print("=" * 60)
    
    # Système
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processeur: {platform.processor()}")
    
    # Mémoire
    memory = psutil.virtual_memory()
    print(f"RAM totale: {memory.total // (1024**3)} GB")
    print(f"RAM disponible: {memory.available // (1024**3)} GB")
    
    # Python
    print(f"Python: {platform.python_version()}")
    
    # GPU (si disponible)
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU: {len(gpus)} détecté(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("GPU: Aucun détecté")
    except:
        print("GPU: Information non disponible")
    
    print("=" * 60)


def visualize_sample_images(image_paths: List[str], labels: List[str], 
                          classes: List[str], n_samples: int = 3):
    """
    Visualise un échantillon d'images pour chaque classe
    
    Args:
        image_paths: Liste des chemins d'images
        labels: Liste des labels correspondants
        classes: Liste des noms de classes
        n_samples: Nombre d'échantillons par classe
    """
    import pandas as pd
    
    # Création d'un DataFrame pour faciliter la manipulation
    df = pd.DataFrame({'path': image_paths, 'label': labels})
    
    fig, axes = plt.subplots(len(classes), n_samples, 
                            figsize=(n_samples * 4, len(classes) * 4))
    
    if len(classes) == 1:
        axes = axes.reshape(1, -1)
    if n_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i, class_name in enumerate(classes):
        class_images = df[df['label'] == class_name]['path'].tolist()
        
        if len(class_images) == 0:
            print(f"⚠️ Aucune image trouvée pour la classe {class_name}")
            continue
            
        # Sélection aléatoire d'échantillons
        sample_indices = np.random.choice(len(class_images), 
                                        min(n_samples, len(class_images)), 
                                        replace=False)
        
        for j, idx in enumerate(sample_indices):
            try:
                # Chargement de l'image
                image_path = class_images[idx]
                with Image.open(image_path) as img:
                    img_array = np.array(img.convert('RGB'))
                
                # Affichage
                axes[i, j].imshow(img_array)
                axes[i, j].set_title(f'{class_name}\n{Path(image_path).name}', 
                                    fontsize=10)
                axes[i, j].axis('off')
                
            except Exception as e:
                axes[i, j].text(0.5, 0.5, f'Erreur:\n{str(e)}', 
                               ha='center', va='center', 
                               transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
        
        # Masquer les axes supplémentaires si pas assez d'images
        for j in range(len(sample_indices), n_samples):
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Échantillons d\'images par classe', y=1.02, fontsize=16)
    plt.show()


def analyze_image_properties(image_paths: List[str], sample_size: int = 100):
    """
    Analyse les propriétés des images (dimensions, couleurs, etc.)
    
    Args:
        image_paths: Liste des chemins d'images
        sample_size: Nombre d'images à analyser
    """
    print(f"📊 Analyse des propriétés sur {min(sample_size, len(image_paths))} images...")
    
    # Échantillonnage
    if len(image_paths) > sample_size:
        indices = np.random.choice(len(image_paths), sample_size, replace=False)
        sample_paths = [image_paths[i] for i in indices]
    else:
        sample_paths = image_paths
    
    # Collecte des données
    widths, heights, channels, file_sizes = [], [], [], []
    brightness_values = []
    
    for img_path in sample_paths:
        try:
            # Propriétés du fichier
            file_size = Path(img_path).stat().st_size / 1024  # KB
            file_sizes.append(file_size)
            
            # Propriétés de l'image
            with Image.open(img_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
                
                # Conversion en RGB pour analyse
                img_rgb = img.convert('RGB')
                img_array = np.array(img_rgb)
                channels.append(img_array.shape[2] if len(img_array.shape) == 3 else 1)
                
                # Luminosité moyenne
                brightness = np.mean(img_array)
                brightness_values.append(brightness)
                
        except Exception as e:
            warnings.warn(f"Erreur analyse {img_path}: {e}")
    
    # Affichage des statistiques
    print(f"\n📐 Dimensions:")
    print(f"   Largeur: {np.mean(widths):.0f} ± {np.std(widths):.0f} px")
    print(f"   Hauteur: {np.mean(heights):.0f} ± {np.std(heights):.0f} px")
    print(f"   Canaux: {np.mean(channels):.1f}")
    
    print(f"\n💾 Taille fichiers:")
    print(f"   Moyenne: {np.mean(file_sizes):.1f} KB")
    print(f"   Min/Max: {np.min(file_sizes):.1f} / {np.max(file_sizes):.1f} KB")
    
    print(f"\n🌟 Luminosité:")
    print(f"   Moyenne: {np.mean(brightness_values):.1f}")
    print(f"   Min/Max: {np.min(brightness_values):.1f} / {np.max(brightness_values):.1f}")
    
    # Visualisations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Distribution des largeurs
    axes[0, 0].hist(widths, bins=20, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Distribution des largeurs')
    axes[0, 0].set_xlabel('Largeur (px)')
    axes[0, 0].set_ylabel('Fréquence')
    
    # Distribution des hauteurs
    axes[0, 1].hist(heights, bins=20, alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Distribution des hauteurs')
    axes[0, 1].set_xlabel('Hauteur (px)')
    axes[0, 1].set_ylabel('Fréquence')
    
    # Distribution des tailles de fichiers
    axes[1, 0].hist(file_sizes, bins=20, alpha=0.7, color='orange')
    axes[1, 0].set_title('Distribution des tailles de fichiers')
    axes[1, 0].set_xlabel('Taille (KB)')
    axes[1, 0].set_ylabel('Fréquence')
    
    # Distribution de la luminosité
    axes[1, 1].hist(brightness_values, bins=20, alpha=0.7, color='pink')
    axes[1, 1].set_title('Distribution de la luminosité')
    axes[1, 1].set_xlabel('Luminosité moyenne')
    axes[1, 1].set_ylabel('Fréquence')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'dimensions': {'width': widths, 'height': heights},
        'file_sizes': file_sizes,
        'brightness': brightness_values,
        'channels': channels
    }


def demonstrate_augmentation_fixed(X_sample: np.ndarray, y_sample: np.ndarray, 
                                 n_augmentations: int = 6):
    """
    Démontre les effets de l'augmentation de données
    
    Args:
        X_sample: Échantillon d'images
        y_sample: Labels correspondants
        n_augmentations: Nombre d'augmentations à afficher
    """
    from ..augmentation import CustomImageAugmenter
    
    augmenter = CustomImageAugmenter()
    
    # Sélection d'une image d'exemple
    sample_idx = 0
    original_image = X_sample[sample_idx]
    label = y_sample[sample_idx]
    
    fig, axes = plt.subplots(2, n_augmentations//2 + 1, figsize=(15, 8))
    axes = axes.flatten()
    
    # Image originale
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original\nClasse: {label}')
    axes[0].axis('off')
    
    # Images augmentées
    for i in range(1, n_augmentations + 1):
        try:
            # Application de l'augmentation
            augmented = augmenter.augment_image(original_image)
            
            axes[i].imshow(augmented)
            axes[i].set_title(f'Augmentation {i}')
            axes[i].axis('off')
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Erreur:\n{str(e)}', 
                        ha='center', va='center', 
                        transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Masquer les axes supplémentaires
    for i in range(n_augmentations + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Démonstration de l\'augmentation de données', y=1.02, fontsize=16)
    plt.show()
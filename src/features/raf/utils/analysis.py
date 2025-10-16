"""
Module utilitaire pour le framework RAF
Contient les fonctions d'analyse système et de support
"""

import os
import sys
import psutil
import platform
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import warnings

from ..utils.config import Config


def print_system_info():
    """
    Affiche les informations complètes du système
    """
    print("🖥️ INFORMATIONS SYSTÈME")
    print("=" * 50)
    
    # Informations de base
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processeur: {platform.processor()}")
    print(f"Python: {sys.version}")
    
    # Mémoire
    memory = psutil.virtual_memory()
    print(f"\n💾 MÉMOIRE:")
    print(f"   Total: {memory.total / (1024**3):.1f} GB")
    print(f"   Disponible: {memory.available / (1024**3):.1f} GB")
    print(f"   Utilisée: {memory.percent:.1f}%")
    
    # CPU
    print(f"\n🔥 PROCESSEUR:")
    print(f"   Cœurs logiques: {psutil.cpu_count()}")
    print(f"   Cœurs physiques: {psutil.cpu_count(logical=False)}")
    print(f"   Utilisation: {psutil.cpu_percent(interval=1):.1f}%")
    
    # Disque
    disk = psutil.disk_usage('/')
    print(f"\n💽 STOCKAGE:")
    print(f"   Total: {disk.total / (1024**3):.1f} GB")
    print(f"   Libre: {disk.free / (1024**3):.1f} GB")
    print(f"   Utilisé: {disk.percent:.1f}%")
    
    # GPU (si disponible)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n🎮 GPU (CUDA):")
            print(f"   Disponible: Oui")
            print(f"   Périphériques: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"   Mémoire: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB")
        else:
            print(f"\n🎮 GPU: Non disponible")
    except ImportError:
        print(f"\n🎮 GPU: PyTorch non installé")
    
    # Répertoire de travail
    print(f"\n📁 ENVIRONNEMENT:")
    print(f"   Répertoire: {os.getcwd()}")
    
    # Variables d'environnement importantes
    important_vars = ['PATH', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES']
    for var in important_vars:
        value = os.environ.get(var, 'Non définie')
        if len(str(value)) > 100:
            value = str(value)[:100] + "..."
        print(f"   {var}: {value}")


def analyze_image_properties(images: np.ndarray, labels: Optional[np.ndarray] = None,
                           sample_size: int = 1000) -> Dict[str, Any]:
    """
    Analyse les propriétés statistiques d'un ensemble d'images
    
    Args:
        images: Array d'images
        labels: Labels optionnels
        sample_size: Taille de l'échantillon pour l'analyse
        
    Returns:
        Dictionnaire avec les statistiques d'analyse
    """
    print(f"🔍 ANALYSE DES PROPRIÉTÉS D'IMAGES")
    print("=" * 50)
    
    # Échantillonnage si nécessaire
    n_images = len(images)
    if n_images > sample_size:
        indices = np.random.choice(n_images, sample_size, replace=False)
        sample_images = images[indices]
        sample_labels = labels[indices] if labels is not None else None
        print(f"Analyse sur un échantillon de {sample_size} images (sur {n_images})")
    else:
        sample_images = images
        sample_labels = labels
        print(f"Analyse sur {n_images} images")
    
    stats = {}
    
    # Dimensions
    print(f"\n📏 DIMENSIONS:")
    shapes = [img.shape for img in sample_images]
    unique_shapes = list(set(shapes))
    stats['shapes'] = unique_shapes
    print(f"   Formes uniques: {len(unique_shapes)}")
    for shape in unique_shapes:
        count = shapes.count(shape)
        print(f"     {shape}: {count} images ({count/len(shapes)*100:.1f}%)")
    
    # Canaux
    if len(sample_images[0].shape) == 3:
        n_channels = sample_images[0].shape[2]
        print(f"   Canaux: {n_channels}")
        stats['channels'] = n_channels
    else:
        print(f"   Images en niveaux de gris")
        stats['channels'] = 1
    
    # Statistiques de pixels
    print(f"\n🎨 STATISTIQUES DE PIXELS:")
    
    # Conversion pour analyse
    flattened_pixels = []
    for img in sample_images:
        if len(img.shape) == 3:
            # Image couleur -> niveaux de gris
            gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img
        flattened_pixels.extend(gray.flatten())
    
    flattened_pixels = np.array(flattened_pixels)
    
    # Statistiques de base
    stats['pixel_stats'] = {
        'mean': float(np.mean(flattened_pixels)),
        'std': float(np.std(flattened_pixels)),
        'min': float(np.min(flattened_pixels)),
        'max': float(np.max(flattened_pixels)),
        'median': float(np.median(flattened_pixels))
    }
    
    print(f"   Moyenne: {stats['pixel_stats']['mean']:.3f}")
    print(f"   Écart-type: {stats['pixel_stats']['std']:.3f}")
    print(f"   Min/Max: {stats['pixel_stats']['min']:.3f} / {stats['pixel_stats']['max']:.3f}")
    print(f"   Médiane: {stats['pixel_stats']['median']:.3f}")
    
    # Histogramme global
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(flattened_pixels, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution des pixels')
    plt.xlabel('Intensité')
    plt.ylabel('Fréquence')
    
    # Distribution par classe si labels disponibles
    if sample_labels is not None:
        plt.subplot(1, 3, 2)
        unique_labels = np.unique(sample_labels)
        
        for label in unique_labels:
            mask = sample_labels == label
            class_images = sample_images[mask]
            
            class_pixels = []
            for img in class_images:
                if len(img.shape) == 3:
                    gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
                else:
                    gray = img
                class_pixels.extend(gray.flatten())
            
            plt.hist(class_pixels, bins=30, alpha=0.6, label=f'Classe {label}')
        
        plt.title('Distribution par classe')
        plt.xlabel('Intensité')
        plt.ylabel('Fréquence')
        plt.legend()
        
        # Statistiques par classe
        stats['class_stats'] = {}
        print(f"\n📊 STATISTIQUES PAR CLASSE:")
        
        for label in unique_labels:
            mask = sample_labels == label
            class_images = sample_images[mask]
            
            class_pixels = []
            for img in class_images:
                if len(img.shape) == 3:
                    gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
                else:
                    gray = img
                class_pixels.extend(gray.flatten())
            
            class_pixels = np.array(class_pixels)
            
            stats['class_stats'][str(label)] = {
                'mean': float(np.mean(class_pixels)),
                'std': float(np.std(class_pixels)),
                'count': int(np.sum(mask))
            }
            
            print(f"   Classe {label}:")
            print(f"     Échantillons: {stats['class_stats'][str(label)]['count']}")
            print(f"     Moyenne: {stats['class_stats'][str(label)]['mean']:.3f}")
            print(f"     Écart-type: {stats['class_stats'][str(label)]['std']:.3f}")
    
    # Boxplot des intensités
    plt.subplot(1, 3, 3)
    
    if sample_labels is not None:
        intensity_by_class = []
        class_names = []
        
        for label in unique_labels:
            mask = sample_labels == label
            class_images = sample_images[mask]
            
            class_means = []
            for img in class_images:
                if len(img.shape) == 3:
                    gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
                else:
                    gray = img
                class_means.append(np.mean(gray))
            
            intensity_by_class.append(class_means)
            class_names.append(f'Classe {label}')
        
        plt.boxplot(intensity_by_class, labels=class_names)
        plt.title('Intensité moyenne par classe')
        plt.ylabel('Intensité moyenne')
        plt.xticks(rotation=45)
    else:
        # Moyennes d'intensité par image
        image_means = []
        for img in sample_images:
            if len(img.shape) == 3:
                gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = img
            image_means.append(np.mean(gray))
        
        plt.hist(image_means, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Distribution des intensités moyennes')
        plt.xlabel('Intensité moyenne')
        plt.ylabel('Nombre d\'images')
    
    plt.tight_layout()
    plt.show()
    
    # Détection d'anomalies
    print(f"\n⚠️ DÉTECTION D'ANOMALIES:")
    
    # Images très sombres ou très claires
    image_means = []
    for img in sample_images:
        if len(img.shape) == 3:
            gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img
        image_means.append(np.mean(gray))
    
    image_means = np.array(image_means)
    
    # Seuils basés sur les percentiles
    dark_threshold = np.percentile(image_means, 5)
    bright_threshold = np.percentile(image_means, 95)
    
    dark_images = np.sum(image_means < dark_threshold)
    bright_images = np.sum(image_means > bright_threshold)
    
    stats['anomalies'] = {
        'dark_images': int(dark_images),
        'bright_images': int(bright_images),
        'dark_threshold': float(dark_threshold),
        'bright_threshold': float(bright_threshold)
    }
    
    print(f"   Images très sombres (< {dark_threshold:.3f}): {dark_images}")
    print(f"   Images très claires (> {bright_threshold:.3f}): {bright_images}")
    
    # Images de taille atypique
    if len(unique_shapes) > 1:
        shape_counts = [shapes.count(shape) for shape in unique_shapes]
        main_shape_idx = np.argmax(shape_counts)
        main_shape = unique_shapes[main_shape_idx]
        
        atypical_shapes = 0
        for i, shape in enumerate(unique_shapes):
            if i != main_shape_idx:
                atypical_shapes += shape_counts[i]
        
        stats['anomalies']['atypical_shapes'] = int(atypical_shapes)
        print(f"   Images de taille atypique: {atypical_shapes}")
    
    return stats


def visualize_sample_images(images: np.ndarray, labels: Optional[np.ndarray] = None,
                          n_samples: int = 12, figsize: Tuple[int, int] = (15, 10),
                          title: str = "Échantillon d'images") -> None:
    """
    Visualise un échantillon d'images du dataset
    
    Args:
        images: Array d'images
        labels: Labels optionnels
        n_samples: Nombre d'images à afficher
        figsize: Taille de la figure
        title: Titre de la figure
    """
    # Sélection aléatoire d'images
    n_available = min(n_samples, len(images))
    indices = np.random.choice(len(images), n_available, replace=False)
    
    # Calcul de la grille
    n_cols = min(4, n_available)
    n_rows = (n_available + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # S'assurer que axes est un array 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, idx in enumerate(indices):
        row = i // n_cols
        col = i % n_cols
        
        img = images[idx]
        
        # Affichage de l'image
        if len(img.shape) == 3:
            axes[row, col].imshow(img)
        else:
            axes[row, col].imshow(img, cmap='gray')
        
        # Titre avec label si disponible
        if labels is not None:
            axes[row, col].set_title(f'Classe: {labels[idx]}')
        else:
            axes[row, col].set_title(f'Image {idx}')
        
        axes[row, col].axis('off')
    
    # Masquer les axes inutilisés
    for i in range(n_available, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def get_memory_usage() -> Dict[str, float]:
    """
    Récupère l'utilisation mémoire actuelle
    
    Returns:
        Dictionnaire avec les informations mémoire
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
        'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
        'percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / (1024 * 1024)
    }


def check_data_quality(images: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Vérifie la qualité des données
    
    Args:
        images: Array d'images
        labels: Array de labels
        
    Returns:
        Rapport de qualité des données
    """
    print("🔍 VÉRIFICATION DE LA QUALITÉ DES DONNÉES")
    print("=" * 50)
    
    quality_report = {
        'issues': [],
        'warnings': [],
        'info': []
    }
    
    # Vérification des dimensions
    if len(images) != len(labels):
        quality_report['issues'].append(f"Désalignement: {len(images)} images vs {len(labels)} labels")
    
    # Vérification des NaN
    nan_images = 0
    for i, img in enumerate(images):
        if np.any(np.isnan(img)):
            nan_images += 1
    
    if nan_images > 0:
        quality_report['issues'].append(f"{nan_images} images contiennent des NaN")
    
    # Vérification des valeurs infinies
    inf_images = 0
    for i, img in enumerate(images):
        if np.any(np.isinf(img)):
            inf_images += 1
    
    if inf_images > 0:
        quality_report['issues'].append(f"{inf_images} images contiennent des valeurs infinies")
    
    # Vérification de la distribution des classes
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_class_size = min(counts)
    max_class_size = max(counts)
    
    if max_class_size / min_class_size > 10:
        quality_report['warnings'].append(f"Dataset très déséquilibré (ratio {max_class_size/min_class_size:.1f}:1)")
    elif max_class_size / min_class_size > 3:
        quality_report['warnings'].append(f"Dataset déséquilibré (ratio {max_class_size/min_class_size:.1f}:1)")
    
    # Informations générales
    quality_report['info'].append(f"Total: {len(images)} images, {len(unique_labels)} classes")
    quality_report['info'].append(f"Distribution: {dict(zip(unique_labels, counts))}")
    
    # Affichage du rapport
    if quality_report['issues']:
        print("❌ PROBLÈMES DÉTECTÉS:")
        for issue in quality_report['issues']:
            print(f"   - {issue}")
    
    if quality_report['warnings']:
        print("\n⚠️ AVERTISSEMENTS:")
        for warning in quality_report['warnings']:
            print(f"   - {warning}")
    
    if quality_report['info']:
        print("\n✅ INFORMATIONS:")
        for info in quality_report['info']:
            print(f"   - {info}")
    
    if not quality_report['issues'] and not quality_report['warnings']:
        print("\n✅ Aucun problème de qualité détecté")
    
    return quality_report
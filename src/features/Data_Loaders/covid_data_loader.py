"""Module utilitaire pour charger les données COVID-19.

Ce module fournit des fonctions pour charger et préprocesser les images
du dataset COVID-19 Radiography Database de manière standardisée.

Auteur: L-Poca
Date: 2025
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

# Configuration globale
COVID_CLASSES = {
    'COVID': 0,
    'Normal': 1, 
    'Viral Pneumonia': 2,
    'Lung_Opacity': 3
}

CLASS_NAMES = ['COVID', 'Normal', 'Viral Pneumonia', 'Lung_Opacity']


def get_data_paths(project_root: Union[str, Path] = None) -> Dict[str, Dict[str, str]]:
    """
    Retourne les chemins vers les données COVID-19.
    
    Args:
        project_root: Chemin racine du projet (auto-détecté si None)
        
    Returns:
        Dict contenant les chemins vers images et masques par classe
    """
    if project_root is None:
        # Auto-détection du projet root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
    
    project_root = Path(project_root)
    data_root = project_root / "data" / "raw" / "COVID-19_Radiography_Dataset" / "COVID-19_Radiography_Dataset"
    
    paths = {
        "COVID": {
            "images": str(data_root / "COVID" / "images"),
            "masks": str(data_root / "COVID" / "masks")
        },
        "Normal": {
            "images": str(data_root / "Normal" / "images"),
            "masks": str(data_root / "Normal" / "masks")
        },
        "Viral Pneumonia": {
            "images": str(data_root / "Viral Pneumonia" / "images"),
            "masks": str(data_root / "Viral Pneumonia" / "masks")
        },
        "Lung_Opacity": {
            "images": str(data_root / "Lung_Opacity" / "images"),
            "masks": str(data_root / "Lung_Opacity" / "masks")
        }
    }
    
    return paths


def check_data_availability(project_root: Union[str, Path] = None) -> Dict[str, Dict[str, any]]:
    """
    Vérifie la disponibilité des données COVID-19.
    
    Args:
        project_root: Chemin racine du projet
        
    Returns:
        Dict avec les statistiques de chaque classe
    """
    paths = get_data_paths(project_root)
    stats = {}
    
    for class_name, class_paths in paths.items():
        image_dir = Path(class_paths["images"])
        mask_dir = Path(class_paths["masks"])
        
        # Compter les images
        if image_dir.exists():
            image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
            n_images = len(image_files)
        else:
            n_images = 0
            image_files = []
        
        # Compter les masques
        if mask_dir.exists():
            mask_files = list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.jpg"))
            n_masks = len(mask_files)
        else:
            n_masks = 0
            mask_files = []
        
        stats[class_name] = {
            "n_images": n_images,
            "n_masks": n_masks,
            "images_available": image_dir.exists() and n_images > 0,
            "masks_available": mask_dir.exists() and n_masks > 0,
            "image_dir": str(image_dir),
            "mask_dir": str(mask_dir),
            "sample_files": image_files[:5]  # Premiers 5 fichiers
        }
    
    return stats


def load_image(image_path: str, target_size: Tuple[int, int] = (224, 224), 
               color_mode: str = 'rgb') -> np.ndarray:
    """
    Charge et préprocesse une image.
    
    Args:
        image_path: Chemin vers l'image
        target_size: Taille cible (width, height)
        color_mode: 'rgb', 'grayscale' ou 'original'
        
    Returns:
        Array numpy de l'image preprocessée
    """
    try:
        # Charger l'image
        image = Image.open(image_path)
        
        # Conversion de mode si nécessaire
        if color_mode == 'rgb' and image.mode != 'RGB':
            image = image.convert('RGB')
        elif color_mode == 'grayscale':
            image = image.convert('L')
        
        # Redimensionnement
        if target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Conversion en array numpy
        img_array = np.array(image)
        
        # Normalisation
        if img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
        
    except Exception as e:
        print(f"Erreur lors du chargement de {image_path}: {e}")
        return None


def load_images_from_class(class_name: str, 
                          max_images: Optional[int] = None,
                          target_size: Tuple[int, int] = (224, 224),
                          color_mode: str = 'rgb',
                          project_root: Union[str, Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge les images d'une classe spécifique.
    
    Args:
        class_name: Nom de la classe ('COVID', 'Normal', etc.)
        max_images: Nombre maximum d'images à charger
        target_size: Taille cible des images
        color_mode: Mode de couleur
        project_root: Racine du projet
        
    Returns:
        Tuple (images, labels) sous forme d'arrays numpy
    """
    paths = get_data_paths(project_root)
    
    if class_name not in paths:
        raise ValueError(f"Classe inconnue: {class_name}. Classes disponibles: {list(paths.keys())}")
    
    image_dir = Path(paths[class_name]["images"])
    
    if not image_dir.exists():
        print(f"⚠️ Dossier non trouvé: {image_dir}")
        return np.array([]), np.array([])
    
    # Lister les fichiers images
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    
    if max_images and len(image_files) > max_images:
        image_files = image_files[:max_images]
    
    print(f"📊 Chargement de {len(image_files)} images de la classe {class_name}")
    
    images = []
    labels = []
    class_label = COVID_CLASSES[class_name]
    
    for i, img_path in enumerate(image_files):
        if (i + 1) % 100 == 0:
            print(f"  Progression: {i+1}/{len(image_files)}")
        
        img_array = load_image(str(img_path), target_size, color_mode)
        
        if img_array is not None:
            images.append(img_array)
            labels.append(class_label)
    
    return np.array(images), np.array(labels)


def load_covid_dataset(classes: List[str] = None,
                      max_images_per_class: Optional[int] = None,
                      target_size: Tuple[int, int] = (224, 224),
                      color_mode: str = 'rgb',
                      test_size: float = 0.2,
                      validation_size: float = 0.1,
                      random_state: int = 42,
                      project_root: Union[str, Path] = None) -> Dict[str, any]:
    """
    Charge le dataset COVID-19 complet.
    
    Args:
        classes: Liste des classes à charger (toutes si None)
        max_images_per_class: Nombre maximum d'images par classe
        target_size: Taille cible des images
        color_mode: Mode de couleur
        test_size: Proportion du jeu de test
        validation_size: Proportion du jeu de validation
        random_state: Seed pour la reproductibilité
        project_root: Racine du projet
        
    Returns:
        Dict contenant les données train/val/test et métadonnées
    """
    if classes is None:
        classes = list(COVID_CLASSES.keys())
    
    print(f"🔄 Chargement du dataset COVID-19")
    print(f"Classes: {classes}")
    print(f"Taille cible: {target_size}")
    print(f"Mode couleur: {color_mode}")
    print(f"Max images/classe: {max_images_per_class or 'Toutes'}")
    
    all_images = []
    all_labels = []
    class_stats = {}
    
    # Charger chaque classe
    for class_name in classes:
        images, labels = load_images_from_class(
            class_name=class_name,
            max_images=max_images_per_class,
            target_size=target_size,
            color_mode=color_mode,
            project_root=project_root
        )
        
        if len(images) > 0:
            all_images.append(images)
            all_labels.append(labels)
            class_stats[class_name] = len(images)
            print(f"✅ {class_name}: {len(images)} images chargées")
        else:
            print(f"⚠️ {class_name}: Aucune image chargée")
    
    if not all_images:
        raise ValueError("Aucune image n'a pu être chargée!")
    
    # Concaténer toutes les données
    X = np.concatenate(all_images, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    print(f"📊 Dataset total: {len(X)} images, {len(np.unique(y))} classes")
    print(f"📐 Forme des images: {X.shape}")
    
    # Mélanger les données
    X, y = shuffle(X, y, random_state=random_state)
    
    # Division train/validation/test
    # D'abord train+val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Puis train / validation
    val_size_adjusted = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    # Distribution des classes
    def get_class_distribution(labels):
        unique, counts = np.unique(labels, return_counts=True)
        return {CLASS_NAMES[label]: count for label, count in zip(unique, counts)}
    
    train_dist = get_class_distribution(y_train)
    val_dist = get_class_distribution(y_val)
    test_dist = get_class_distribution(y_test)
    
    print(f"\n📊 DISTRIBUTION DES DONNÉES:")
    print(f"Train: {train_dist}")
    print(f"Validation: {val_dist}")
    print(f"Test: {test_dist}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'class_names': [CLASS_NAMES[i] for i in sorted(np.unique(y))],
        'class_distribution': {
            'train': train_dist,
            'validation': val_dist,
            'test': test_dist
        },
        'metadata': {
            'total_samples': len(X),
            'image_shape': X.shape[1:],
            'n_classes': len(np.unique(y)),
            'classes_loaded': classes,
            'max_per_class': max_images_per_class,
            'color_mode': color_mode,
            'target_size': target_size
        }
    }


def load_sample_images(n_samples: int = 5,
                      classes: List[str] = None,
                      target_size: Tuple[int, int] = (224, 224),
                      project_root: Union[str, Path] = None) -> Dict[str, List[np.ndarray]]:
    """
    Charge quelques images d'exemple pour chaque classe.
    
    Args:
        n_samples: Nombre d'échantillons par classe
        classes: Classes à charger
        target_size: Taille des images
        project_root: Racine du projet
        
    Returns:
        Dict avec les images échantillons par classe
    """
    if classes is None:
        classes = list(COVID_CLASSES.keys())
    
    samples = {}
    
    for class_name in classes:
        images, _ = load_images_from_class(
            class_name=class_name,
            max_images=n_samples,
            target_size=target_size,
            project_root=project_root
        )
        
        samples[class_name] = images
    
    return samples


def prepare_data_for_sklearn(X: np.ndarray) -> np.ndarray:
    """
    Prépare les données pour sklearn (aplatissement).
    
    Args:
        X: Array d'images
        
    Returns:
        Array aplati pour sklearn
    """
    if len(X.shape) > 2:
        return X.reshape(X.shape[0], -1)
    return X


def prepare_data_for_tensorflow(dataset: Dict) -> Dict:
    """
    Prépare les données pour TensorFlow/Keras.
    
    Args:
        dataset: Dataset de load_covid_dataset
        
    Returns:
        Dataset formaté pour TensorFlow
    """
    from tensorflow.keras.utils import to_categorical
    
    # One-hot encoding des labels
    n_classes = len(dataset['class_names'])
    
    y_train_cat = to_categorical(dataset['y_train'], n_classes)
    y_val_cat = to_categorical(dataset['y_val'], n_classes)
    y_test_cat = to_categorical(dataset['y_test'], n_classes)
    
    return {
        **dataset,
        'y_train_categorical': y_train_cat,
        'y_val_categorical': y_val_cat,
        'y_test_categorical': y_test_cat
    }


def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calcule les poids des classes pour gérer le déséquilibre.
    
    Args:
        y: Labels
        
    Returns:
        Dict des poids par classe
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


# Fonction de test
if __name__ == "__main__":
    print("🧪 Test du module de chargement COVID-19")
    
    # Vérifier la disponibilité des données
    print("\n📊 Vérification des données...")
    stats = check_data_availability()
    
    for class_name, class_stats in stats.items():
        status = "✅" if class_stats["images_available"] else "❌"
        print(f"{status} {class_name}: {class_stats['n_images']} images")
    
    # Test de chargement d'un échantillon
    print("\n🔄 Test de chargement d'échantillons...")
    try:
        samples = load_sample_images(n_samples=3)
        for class_name, images in samples.items():
            print(f"✅ {class_name}: {len(images)} échantillons chargés, forme: {images.shape if len(images) > 0 else 'N/A'}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
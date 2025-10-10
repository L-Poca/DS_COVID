"""
Feature extraction and data loading utilities for COVID-19 analysis
"""

import os
import random
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import cv2


def load_images_flat(
    folder_path: str, 
    label: int, 
    img_size: Tuple[int, int] = (256, 256), 
    max_images: Optional[int] = None,
    seed: int = 42
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load images in grayscale and flatten them
    
    Args:
        folder_path: Path to folder containing images
        label: Numeric label for the images
        img_size: Target image size (width, height)
        max_images: Maximum number of images to load
        seed: Random seed for shuffling
        
    Returns:
        Tuple of (flattened_images, labels)
    """
    data, labels = [], []
    
    if not os.path.exists(folder_path):
        print(f"[WARNING] Path not found: {folder_path}")
        return data, labels
        
    files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.seed(seed)
    random.shuffle(files)

    if max_images:
        files = files[:max_images]

    print(f"[INFO] Loading {len(files)} images from {folder_path}")

    for file in files:
        try:
            img = Image.open(os.path.join(folder_path, file)).convert('L')
            img = img.resize(img_size)
            img = np.array(img).astype('float32')
            img_norm = (img / 127.5) - 1  # Normalize to [-1, 1]
            data.append(img_norm.flatten())
            labels.append(label)
        except Exception as e:
            print(f"[WARNING] Error with {file}: {e}")

    return data, labels


def prepare_covid_data(
    dataset_path: str, 
    categories: List[str], 
    target_size: Tuple[int, int] = (128, 128),
    max_samples_per_class: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare COVID-19 dataset for ML classification
    
    Args:
        dataset_path: Path to main dataset directory
        categories: List of category names
        target_size: Target image size
        max_samples_per_class: Maximum samples per category
        
    Returns:
        Tuple of (features, labels)
    """
    X = []  # Image vectors
    y = []  # Labels
    
    for i, category in enumerate(categories):
        images_path = Path(dataset_path) / category / "images"
        
        if not images_path.exists():
            print(f"[WARNING] Category path not found: {images_path}")
            continue
            
        image_files = list(images_path.glob("*.png"))
        
        if max_samples_per_class:
            image_files = image_files[:max_samples_per_class]
        
        print(f"[INFO] Processing {len(image_files)} images for {category}")
        
        for img_file in image_files:
            try:
                # Load and preprocess
                image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                    
                image = cv2.resize(image, target_size)
                
                # Flatten and normalize
                vector = image.flatten() / 255.0
                
                X.append(vector)
                y.append(i)  # Numeric label
                
            except Exception as e:
                print(f"[WARNING] Error processing {img_file}: {e}")
                continue
    
    return np.array(X), np.array(y)


def get_image_mask_pairs(dataset_path: str, category: str) -> List[Tuple[Path, Path]]:
    """
    Get image/mask pairs for a category
    
    Args:
        dataset_path: Path to main dataset
        category: Category name
        
    Returns:
        List of (image_path, mask_path) tuples
    """
    category_path = Path(dataset_path) / category
    images_path = category_path / "images"
    masks_path = category_path / "masks"
    
    if not images_path.exists() or not masks_path.exists():
        return []
    
    pairs = []
    for img_file in images_path.glob("*.png"):
        mask_file = masks_path / img_file.name
        if mask_file.exists():
            pairs.append((img_file, mask_file))
    
    return pairs


# Default configurations
DEFAULT_CLASS_PATHS = {
    0: "COVID/images",
    1: "Normal/images", 
    2: "Viral Pneumonia/images",
    3: "Lung_Opacity/images",
}

DEFAULT_CLASS_LABELS = {
    0: 'COVID',
    1: 'NORMAL', 
    2: 'VIRAL',
    3: 'LUNG'
}

DEFAULT_CATEGORIES = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]
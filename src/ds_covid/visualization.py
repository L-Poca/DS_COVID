"""
Visualization utilities for COVID-19 analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from pathlib import Path
import cv2


def visualize_samples(
    dataset_path: str, 
    categories: List[str],
    mask_applicator,
    n_samples: int = 1,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    Visualize samples from each category with masks applied
    
    Args:
        dataset_path: Path to dataset
        categories: List of category names
        mask_applicator: MaskApplicator instance
        n_samples: Number of samples per category
        figsize: Figure size
    """
    fig, axes = plt.subplots(len(categories), 4, figsize=figsize)
    fig.suptitle('Sample Images by Category: Original - Mask - Overlay - Extract', 
                 fontsize=16, fontweight='bold')
    
    for i, category in enumerate(categories):
        category_path = Path(dataset_path) / category
        images_path = category_path / "images"
        masks_path = category_path / "masks"
        
        if images_path.exists() and masks_path.exists():
            # Get first image
            image_files = list(images_path.glob("*.png"))
            if image_files:
                img_path = image_files[0]
                mask_path = masks_path / img_path.name
                
                if mask_path.exists():
                    try:
                        # Apply masks
                        overlay_result, original, mask = mask_applicator.apply_mask(
                            img_path, mask_path, method="overlay", alpha=0.5)
                        extract_result, _, _ = mask_applicator.apply_mask(
                            img_path, mask_path, method="extract")
                        
                        # Display
                        axes[i, 0].imshow(original, cmap='gray')
                        axes[i, 0].set_title(f'{category}\nOriginal')
                        axes[i, 0].axis('off')
                        
                        axes[i, 1].imshow(mask, cmap='gray')
                        axes[i, 1].set_title('Mask')
                        axes[i, 1].axis('off')
                        
                        axes[i, 2].imshow(overlay_result, cmap='gray')
                        axes[i, 2].set_title('Overlay (α=0.5)')
                        axes[i, 2].axis('off')
                        
                        axes[i, 3].imshow(extract_result, cmap='gray')
                        axes[i, 3].set_title('Extract')
                        axes[i, 3].axis('off')
                        
                    except Exception as e:
                        for j in range(4):
                            axes[i, j].text(0.5, 0.5, f'Error: {str(e)}', 
                                           ha='center', va='center', transform=axes[i, j].transAxes)
                            axes[i, j].axis('off')
                else:
                    for j in range(4):
                        axes[i, j].text(0.5, 0.5, 'No mask found', 
                                       ha='center', va='center', transform=axes[i, j].transAxes)
                        axes[i, j].axis('off')
            else:
                for j in range(4):
                    axes[i, j].text(0.5, 0.5, 'No images found', 
                                   ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].axis('off')
        else:
            for j in range(4):
                axes[i, j].text(0.5, 0.5, 'Path not found', 
                               ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()


def compare_methods(
    dataset_path: str,
    category: str = "COVID",
    sample_idx: int = 0,
    mask_applicator = None,
    figsize: Tuple[int, int] = (16, 8)
) -> None:
    """
    Compare different mask application methods
    
    Args:
        dataset_path: Path to dataset
        category: Category to analyze
        sample_idx: Index of sample to use
        mask_applicator: MaskApplicator instance
        figsize: Figure size
    """
    if mask_applicator is None:
        from .models import MaskApplicator
        mask_applicator = MaskApplicator()
    
    category_path = Path(dataset_path) / category
    images_path = category_path / "images"
    masks_path = category_path / "masks"
    
    if not images_path.exists() or not masks_path.exists():
        print(f"❌ Paths not found for {category}")
        return
    
    image_files = list(images_path.glob("*.png"))
    if sample_idx >= len(image_files):
        print(f"❌ Sample index {sample_idx} out of range for {category}")
        return
    
    img_path = image_files[sample_idx]
    mask_path = masks_path / img_path.name
    
    if not mask_path.exists():
        print(f"❌ Mask not found for {img_path.name}")
        return
    
    # Test different alpha values for overlay
    alphas = [0.3, 0.5, 0.7]
    
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.suptitle(f'Method Comparison - {category} (Sample {sample_idx})', 
                 fontsize=16, fontweight='bold')
    
    try:
        # Original images
        original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # First row: Original and overlays
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Overlays with different alphas
        for i, alpha in enumerate(alphas):
            overlay_result, _, _ = mask_applicator.apply_mask(
                img_path, mask_path, method="overlay", alpha=alpha)
            axes[0, i+1].imshow(overlay_result, cmap='gray')
            axes[0, i+1].set_title(f'Overlay α={alpha}')
            axes[0, i+1].axis('off')
        
        # Second row: Mask and other methods
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('Mask')
        axes[1, 0].axis('off')
        
        multiply_result, _, _ = mask_applicator.apply_mask(
            img_path, mask_path, method="multiply")
        axes[1, 1].imshow(multiply_result, cmap='gray')
        axes[1, 1].set_title('Multiply')
        axes[1, 1].axis('off')
        
        extract_result, _, _ = mask_applicator.apply_mask(
            img_path, mask_path, method="extract")
        axes[1, 2].imshow(extract_result, cmap='gray')
        axes[1, 2].set_title('Extract')
        axes[1, 2].axis('off')
        
        # Histogram of mask
        axes[1, 3].hist(mask.flatten(), bins=50, alpha=0.7, color='blue')
        axes[1, 3].set_title('Mask Histogram')
        axes[1, 3].set_xlabel('Intensity')
        axes[1, 3].set_ylabel('Frequency')
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
    
    plt.tight_layout()
    plt.show()


def plot_training_history(history, figsize: Tuple[int, int] = (12, 4)) -> None:
    """
    Plot training history (accuracy and loss)
    
    Args:
        history: Keras training history
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str],
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        figsize: Figure size
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
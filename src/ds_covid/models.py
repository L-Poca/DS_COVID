"""
Deep Learning Models for COVID-19 Radiography Analysis
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from typing import Tuple, Optional


def build_baseline_cnn(
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    num_classes: int = 4,
    dropout_rate: float = 0.5
) -> tf.keras.Model:
    """
    Build a baseline CNN model for COVID-19 classification
    
    Args:
        input_shape: Input shape for images (height, width, channels)
        num_classes: Number of classification classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Second convolutional block  
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        layers.Flatten(),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
         
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


class MaskApplicator:
    """
    Class for applying masks to radiographic images
    """
    
    def __init__(self, max_image_size: Tuple[int, int] = (512, 512)):
        """
        Initialize MaskApplicator
        
        Args:
            max_image_size: Maximum size for images to prevent memory issues
        """
        self.methods = ["overlay", "multiply", "extract"]
        self.max_size = max_image_size
    
    def apply_mask(
        self, 
        image_path: str, 
        mask_path: str, 
        method: str = "overlay", 
        alpha: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply mask to image using specified method
        
        Args:
            image_path: Path to input image
            mask_path: Path to mask image
            method: Application method ("overlay", "multiply", "extract")
            alpha: Alpha value for overlay method
            
        Returns:
            Tuple of (result_image, original_image, mask)
        """
        import cv2
        
        # Load images
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise ValueError("Could not load image or mask")
        
        # Resize if necessary
        if image.shape[0] > self.max_size[0] or image.shape[1] > self.max_size[1]:
            image = cv2.resize(image, self.max_size)
            mask = cv2.resize(mask, self.max_size)
        elif image.shape != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Apply mask based on method
        if method == "overlay":
            result = cv2.addWeighted(image, 1-alpha, mask, alpha, 0)
        elif method == "multiply":
            image_norm = image.astype(np.float32) / 255.0
            mask_norm = mask.astype(np.float32) / 255.0
            result = (image_norm * mask_norm * 255).astype(np.uint8)
        elif method == "extract":
            mask_norm = mask.astype(np.float32) / 255.0
            result = np.where(mask_norm > 0.5, image, 0).astype(np.uint8)
        else:
            raise ValueError(f"Method '{method}' not supported. Use: {self.methods}")
        
        return result, image, mask
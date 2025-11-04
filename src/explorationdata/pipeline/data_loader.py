"""
Data loader for COVID-19 radiography dataset
Handles loading images, masks, and metadata files
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


class DatasetLoader:
    """Loads and validates the COVID-19 radiography dataset"""

    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        classes: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the dataset loader

        Args:
            base_path: Path to COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset
            metadata_path: Path to metadata directory
            classes: List of classes to load (default: all)
            logger: Logger instance
        """
        self.base_path = Path(base_path)
        self.metadata_path = Path(metadata_path)
        self.logger = logger or logging.getLogger(__name__)

        # Default classes based on problem statement
        self.classes = classes or [
            "COVID",
            "Lung_Opacity",
            "Normal",
            "Viral Pneumonia"
        ]

        self.image_data = []
        self.corrupted_images = []

    def load_metadata(self, class_name: str) -> pd.DataFrame:
        """Load metadata Excel file for a given class"""
        metadata_file = self.metadata_path / f"{class_name}.metadata.xlsx"

        if not metadata_file.exists():
            self.logger.warning(
                f"Metadata file not found: {metadata_file}"
            )
            return pd.DataFrame()

        try:
            df = pd.read_excel(metadata_file)
            df.columns = df.columns.str.strip()
            self.logger.info(
                f"Loaded metadata for {class_name}: {len(df)} entries"
            )
            return df
        except Exception as e:
            self.logger.error(
                f"Error loading metadata for {class_name}: {e}"
            )
            return pd.DataFrame()

    def validate_image(
        self,
        image_path: Path
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Validate and get basic stats for an image

        Returns:
            (is_valid, stats_dict)
        """
        try:
            img = Image.open(image_path)
            img.verify()  # Verify it's a valid image

            # Re-open after verify (verify closes the file)
            img = Image.open(image_path)
            img_array = np.array(img)

            stats = {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format,
                "size_bytes": image_path.stat().st_size,
                "mean": float(np.mean(img_array)),
                "std": float(np.std(img_array)),
                "min": float(np.min(img_array)),
                "max": float(np.max(img_array))
            }
            return True, stats

        except Exception as e:
            self.logger.error(f"Corrupted image {image_path}: {e}")
            return False, None

    def load_images_from_class(
        self,
        class_name: str,
        max_images: Optional[int] = None
    ) -> List[Dict]:
        """
        Load images from a specific class directory

        Args:
            class_name: Name of the class
            max_images: Maximum number of images to load (None = all)

        Returns:
            List of image info dictionaries
        """
        class_dir = self.base_path / class_name
        images_dir = class_dir / "images"
        masks_dir = class_dir / "masks"

        if not images_dir.exists():
            self.logger.error(
                f"Images directory not found: {images_dir}"
            )
            return []

        # Get all image files
        image_files = sorted(
            list(images_dir.glob("*.png")) +
            list(images_dir.glob("*.jpg")) +
            list(images_dir.glob("*.jpeg"))
        )

        if max_images:
            image_files = image_files[:max_images]

        class_images = []
        self.logger.info(
            f"Loading {len(image_files)} images from {class_name}"
        )

        for img_path in tqdm(
            image_files,
            desc=f"Loading {class_name}"
        ):
            # Check for corresponding mask
            mask_path = masks_dir / img_path.name
            has_mask = mask_path.exists()

            # Validate image
            is_valid, stats = self.validate_image(img_path)

            if not is_valid:
                self.corrupted_images.append({
                    "path": str(img_path),
                    "class": class_name
                })
                continue

            # Build image info
            img_info = {
                "filename": img_path.name,
                "class": class_name,
                "image_path": str(img_path),
                "mask_path": str(mask_path) if has_mask else None,
                "has_mask": has_mask,
                **stats
            }

            # If mask exists, get mask stats
            if has_mask:
                try:
                    mask = Image.open(mask_path)
                    mask_array = np.array(mask)
                    img_array = np.array(Image.open(img_path))

                    # Calculate masked region stats
                    # Handle both 2D (grayscale) and 3D (RGB) masks
                    if mask_array.ndim == 3:
                        # Convert RGB mask to grayscale if needed
                        mask_array = mask_array[:, :, 0]
                    
                    # Ensure mask and image have the same spatial dimensions
                    if mask_array.shape[:2] != img_array.shape[:2]:
                        # Resize mask to match image dimensions
                        mask_resized = Image.fromarray(mask_array).resize(
                            (img_array.shape[1], img_array.shape[0]),
                            Image.NEAREST
                        )
                        mask_array = np.array(mask_resized)
                    
                    mask_binary = mask_array > 0
                    if mask_binary.any():
                        # Handle both grayscale and RGB images
                        if img_array.ndim == 3:
                            # For RGB images, apply mask to each channel
                            masked_pixels = img_array[mask_binary].mean(axis=1)
                        else:
                            # For grayscale images
                            masked_pixels = img_array[mask_binary]
                        
                        img_info["mask_area_fraction"] = float(
                            mask_binary.sum() / mask_binary.size
                        )
                        img_info["masked_mean"] = float(
                            np.mean(masked_pixels)
                        )
                        img_info["masked_std"] = float(
                            np.std(masked_pixels)
                        )
                    else:
                        img_info["mask_area_fraction"] = 0.0
                        img_info["masked_mean"] = None
                        img_info["masked_std"] = None

                except Exception as e:
                    self.logger.warning(
                        f"Error processing mask for {img_path.name}: {e}"
                    )
                    img_info["mask_area_fraction"] = None
                    img_info["masked_mean"] = None
                    img_info["masked_std"] = None

            class_images.append(img_info)

        return class_images

    def load_all_images(
        self,
        max_images_per_class: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load all images from all classes

        Args:
            max_images_per_class: Max images per class (None = all)

        Returns:
            DataFrame with all image information
        """
        self.logger.info("Starting to load dataset...")

        for class_name in self.classes:
            class_images = self.load_images_from_class(
                class_name,
                max_images_per_class
            )
            self.image_data.extend(class_images)

        df = pd.DataFrame(self.image_data)

        self.logger.info(
            f"Loaded {len(df)} images total"
        )
        self.logger.info(
            f"Found {len(self.corrupted_images)} corrupted images"
        )

        # Print class distribution
        if not df.empty:
            class_counts = df['class'].value_counts()
            self.logger.info("Class distribution:")
            for cls, count in class_counts.items():
                self.logger.info(f"  {cls}: {count}")

        return df

    def get_corrupted_images(self) -> List[Dict]:
        """Return list of corrupted images found"""
        return self.corrupted_images

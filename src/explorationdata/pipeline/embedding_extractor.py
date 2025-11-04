"""
Embedding extraction using pre-trained deep learning models
Supports ResNet50 and other architectures with GPU/CPU adaptivity
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available, embeddings will be limited")


class EmbeddingExtractor:
    """Extract embeddings from images using pre-trained models"""

    def __init__(
        self,
        model_name: str = "resnet50",
        device: Optional[str] = None,
        batch_size: int = 32,
        image_size: int = 224,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize embedding extractor

        Args:
            model_name: Name of pre-trained model
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for processing
            image_size: Size to resize images to
            logger: Logger instance
        """
        self.model_name = model_name
        self.image_size = image_size
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)

        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for embedding extraction. "
                "Install with: pip install torch torchvision"
            )

        # Determine device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.logger.info(f"Using device: {self.device}")

        # Adapt batch size based on device
        if self.device.type == "cpu" and batch_size > 8:
            self.batch_size = 8
            self.logger.info(
                f"Reduced batch size to {self.batch_size} for CPU"
            )

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self) -> nn.Module:
        """Load and prepare the pre-trained model"""
        self.logger.info(f"Loading {self.model_name} model...")

        if self.model_name == "resnet50":
            model = models.resnet50(weights='IMAGENET1K_V2')
            # Remove the final classification layer
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == "resnet18":
            model = models.resnet18(weights='IMAGENET1K_V1')
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        model = model.to(self.device)
        return model

    def preprocess_image(
        self,
        image_path: str,
        mask_path: Optional[str] = None,
        apply_mask: bool = False
    ) -> torch.Tensor:
        """
        Preprocess single image for embedding extraction

        Args:
            image_path: Path to image
            mask_path: Path to mask (optional)
            apply_mask: Whether to apply mask before extraction

        Returns:
            Preprocessed image tensor
        """
        img = Image.open(image_path).convert('RGB')

        # Apply mask if requested and available
        if apply_mask and mask_path:
            try:
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize(img.size)
                mask_array = np.array(mask)
                img_array = np.array(img)

                # Apply mask
                mask_binary = mask_array > 0
                masked_img = np.zeros_like(img_array)
                for c in range(3):
                    masked_img[:, :, c] = img_array[:, :, c] * mask_binary

                img = Image.fromarray(masked_img)
            except Exception as e:
                self.logger.warning(
                    f"Failed to apply mask for {image_path}: {e}"
                )

        return self.transform(img)

    def extract_embeddings(
        self,
        image_df: pd.DataFrame,
        apply_mask: bool = False,
        checkpoint_interval: int = 500
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract embeddings from all images in dataframe

        Args:
            image_df: DataFrame with image information
            apply_mask: Whether to apply masks before extraction
            checkpoint_interval: Save checkpoint every N images

        Returns:
            (embeddings_array, filenames_list)
        """
        self.logger.info(
            f"Extracting embeddings for {len(image_df)} images "
            f"(apply_mask={apply_mask})"
        )

        embeddings_list = []
        filenames = []

        with torch.no_grad():
            for i in tqdm(
                range(0, len(image_df), self.batch_size),
                desc="Extracting embeddings"
            ):
                batch_df = image_df.iloc[i:i + self.batch_size]
                batch_tensors = []

                for _, row in batch_df.iterrows():
                    try:
                        tensor = self.preprocess_image(
                            row['image_path'],
                            row.get('mask_path'),
                            apply_mask=apply_mask
                        )
                        batch_tensors.append(tensor)
                        filenames.append(row['filename'])
                    except Exception as e:
                        self.logger.error(
                            f"Error processing {row['filename']}: {e}"
                        )
                        continue

                if batch_tensors:
                    batch = torch.stack(batch_tensors).to(self.device)
                    embeddings = self.model(batch)
                    embeddings = embeddings.squeeze().cpu().numpy()

                    # Handle single image case
                    if len(embeddings.shape) == 1:
                        embeddings = embeddings.reshape(1, -1)

                    embeddings_list.append(embeddings)

        # Concatenate all embeddings
        if embeddings_list:
            embeddings_array = np.vstack(embeddings_list)
            self.logger.info(
                f"Extracted embeddings shape: {embeddings_array.shape}"
            )
            return embeddings_array, filenames
        else:
            self.logger.error("No embeddings extracted")
            return np.array([]), []

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        filenames: List[str],
        output_dir: Path,
        prefix: str = ""
    ):
        """
        Save embeddings and metadata

        Args:
            embeddings: Embeddings array
            filenames: List of filenames
            output_dir: Output directory
            prefix: Prefix for filenames (e.g., 'masked_')
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save embeddings as numpy array
        embeddings_path = output_dir / f"{prefix}embeddings.npy"
        np.save(embeddings_path, embeddings)
        self.logger.info(f"Saved embeddings to {embeddings_path}")

        # Save filenames
        files_df = pd.DataFrame({
            'filename': filenames,
            'embedding_index': range(len(filenames))
        })
        files_path = output_dir / f"{prefix}embeddings_files.csv"
        files_df.to_csv(files_path, index=False)
        self.logger.info(f"Saved filenames to {files_path}")

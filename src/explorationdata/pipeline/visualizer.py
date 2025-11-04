"""
Visualization generation for the EDA pipeline
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random


class Visualizer:
    """Generate visualizations for the dataset"""

    def __init__(
        self,
        random_state: int = 42,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize visualizer

        Args:
            random_state: Random seed for reproducibility
            logger: Logger instance
        """
        self.random_state = random_state
        self.logger = logger or logging.getLogger(__name__)
        random.seed(random_state)
        np.random.seed(random_state)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100

    def plot_class_distribution(
        self,
        df: pd.DataFrame,
        output_path: Path
    ):
        """Plot class distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Count plot
        class_counts = df['class'].value_counts()
        class_counts.plot(kind='bar', ax=ax1, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_title('Image Count by Class')
        ax1.tick_params(axis='x', rotation=45)

        # Pie chart
        ax2.pie(
            class_counts.values,
            labels=class_counts.index,
            autopct='%1.1f%%',
            startangle=90
        )
        ax2.set_title('Class Distribution')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved class distribution to {output_path}")

    def plot_image_dimensions(
        self,
        df: pd.DataFrame,
        output_path: Path
    ):
        """Plot image dimension distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Width distribution
        df['width'].hist(bins=30, ax=axes[0, 0], color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Image Width Distribution')
        axes[0, 0].axvline(df['width'].median(), color='red', linestyle='--', label='Median')
        axes[0, 0].legend()

        # Height distribution
        df['height'].hist(bins=30, ax=axes[0, 1], color='lightcoral', edgecolor='black')
        axes[0, 1].set_xlabel('Height (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Image Height Distribution')
        axes[0, 1].axvline(df['height'].median(), color='red', linestyle='--', label='Median')
        axes[0, 1].legend()

        # Aspect ratio
        df['aspect_ratio'] = df['width'] / df['height']
        df['aspect_ratio'].hist(bins=30, ax=axes[1, 0], color='lightgreen', edgecolor='black')
        axes[1, 0].set_xlabel('Aspect Ratio (W/H)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Aspect Ratio Distribution')

        # Size by class
        df.boxplot(column='size_bytes', by='class', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('File Size (bytes)')
        axes[1, 1].set_title('File Size by Class')
        plt.suptitle('')  # Remove automatic title

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved dimension distributions to {output_path}")

    def plot_intensity_statistics(
        self,
        df: pd.DataFrame,
        output_path: Path
    ):
        """Plot intensity statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Mean intensity by class
        df.boxplot(column='mean', by='class', ax=axes[0, 0])
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Mean Intensity')
        axes[0, 0].set_title('Mean Intensity by Class')
        plt.suptitle('')

        # Std intensity by class
        df.boxplot(column='std', by='class', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Std Intensity')
        axes[0, 1].set_title('Std Intensity by Class')

        # Scatter: mean vs std
        for cls in df['class'].unique():
            cls_df = df[df['class'] == cls]
            axes[1, 0].scatter(
                cls_df['mean'],
                cls_df['std'],
                alpha=0.5,
                label=cls,
                s=10
            )
        axes[1, 0].set_xlabel('Mean Intensity')
        axes[1, 0].set_ylabel('Std Intensity')
        axes[1, 0].set_title('Mean vs Std Intensity')
        axes[1, 0].legend()

        # Masked mean (if available)
        if 'masked_mean' in df.columns:
            df_masked = df.dropna(subset=['masked_mean'])
            if not df_masked.empty:
                df_masked.boxplot(column='masked_mean', by='class', ax=axes[1, 1])
                axes[1, 1].set_xlabel('Class')
                axes[1, 1].set_ylabel('Masked Mean Intensity')
                axes[1, 1].set_title('Masked Region Mean Intensity by Class')
            else:
                axes[1, 1].text(0.5, 0.5, 'No masked data', ha='center', va='center')
        else:
            axes[1, 1].text(0.5, 0.5, 'No masked data', ha='center', va='center')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved intensity statistics to {output_path}")

    def create_image_grid(
        self,
        df: pd.DataFrame,
        output_path: Path,
        n_per_class: int = 5,
        sample_mode: str = "random",
        title: str = "Sample Images"
    ):
        """
        Create grid of sample images

        Args:
            df: DataFrame with image info
            output_path: Path to save figure
            n_per_class: Number of images per class
            sample_mode: 'random', 'top', or 'outlier'
            title: Figure title
        """
        classes = sorted(df['class'].unique())
        n_classes = len(classes)

        fig, axes = plt.subplots(
            n_classes,
            n_per_class,
            figsize=(n_per_class * 2, n_classes * 2)
        )

        if n_classes == 1:
            axes = axes.reshape(1, -1)

        for i, cls in enumerate(classes):
            cls_df = df[df['class'] == cls]

            # Select images based on mode
            if sample_mode == "random":
                sampled = cls_df.sample(n=min(n_per_class, len(cls_df)), random_state=self.random_state)
            elif sample_mode == "top":
                sampled = cls_df.head(n_per_class)
            elif sample_mode == "outlier":
                # Select images with extreme mean intensity
                if len(cls_df) >= n_per_class:
                    sorted_df = cls_df.sort_values('mean')
                    n_extreme = n_per_class // 2
                    sampled = pd.concat([
                        sorted_df.head(n_extreme),
                        sorted_df.tail(n_per_class - n_extreme)
                    ])
                else:
                    sampled = cls_df
            else:
                sampled = cls_df.sample(n=min(n_per_class, len(cls_df)), random_state=self.random_state)

            for j, (_, row) in enumerate(sampled.iterrows()):
                if j >= n_per_class:
                    break

                try:
                    img = Image.open(row['image_path'])
                    axes[i, j].imshow(img, cmap='gray')
                    axes[i, j].axis('off')

                    if j == 0:
                        axes[i, j].set_ylabel(cls, fontsize=10, rotation=0, ha='right')
                except Exception as e:
                    self.logger.warning(f"Error loading image {row['filename']}: {e}")
                    axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center')
                    axes[i, j].axis('off')

        plt.suptitle(title, fontsize=14, y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved image grid to {output_path}")

    def create_image_mask_overlay(
        self,
        df: pd.DataFrame,
        output_path: Path,
        n_samples: int = 8
    ):
        """Create side-by-side image and mask overlays"""
        # Filter images with masks
        df_masked = df[df['has_mask'] == True].copy()

        if df_masked.empty:
            self.logger.warning("No masked images available")
            return

        # Sample images
        sampled = df_masked.sample(n=min(n_samples, len(df_masked)), random_state=self.random_state)

        fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 3))

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i, (_, row) in enumerate(sampled.iterrows()):
            try:
                # Load image and mask
                img = Image.open(row['image_path']).convert('RGB')
                mask = Image.open(row['mask_path']).convert('L')

                img_array = np.array(img)
                mask_array = np.array(mask)

                # Create overlay
                overlay = img_array.copy()
                mask_colored = np.zeros_like(img_array)
                mask_colored[:, :, 1] = mask_array  # Green channel
                overlay = np.where(
                    mask_array[:, :, np.newaxis] > 0,
                    (0.7 * img_array + 0.3 * mask_colored).astype(np.uint8),
                    img_array
                )

                # Plot
                axes[i, 0].imshow(img)
                axes[i, 0].set_title(f'{row["class"]} - Original')
                axes[i, 0].axis('off')

                axes[i, 1].imshow(mask, cmap='gray')
                axes[i, 1].set_title('Mask')
                axes[i, 1].axis('off')

                axes[i, 2].imshow(overlay)
                axes[i, 2].set_title('Overlay')
                axes[i, 2].axis('off')

            except Exception as e:
                self.logger.warning(f"Error creating overlay for {row['filename']}: {e}")
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center')
                    axes[i, j].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved image-mask overlays to {output_path}")

    def plot_2d_embeddings(
        self,
        projections: np.ndarray,
        labels: list,
        output_path: Path,
        title: str = "2D Projection",
        color_by: str = "label"
    ):
        """
        Plot 2D embeddings scatter plot

        Args:
            projections: 2D array of projections
            labels: Labels for coloring
            output_path: Output path
            title: Plot title
            color_by: What to color by ('label' or 'continuous')
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        if color_by == "label":
            unique_labels = sorted(set(labels))
            for label in unique_labels:
                mask = np.array(labels) == label
                ax.scatter(
                    projections[mask, 0],
                    projections[mask, 1],
                    label=label,
                    alpha=0.6,
                    s=10
                )
            ax.legend()
        else:
            scatter = ax.scatter(
                projections[:, 0],
                projections[:, 1],
                c=labels,
                cmap='viridis',
                alpha=0.6,
                s=10
            )
            plt.colorbar(scatter, ax=ax)

        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved 2D projection plot to {output_path}")

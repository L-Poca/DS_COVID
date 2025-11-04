"""
Advanced analysis: Grad-CAM, PCA reconstructions, and other visualizations
"""

import logging
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AdvancedAnalyzer:
    """Perform advanced analysis and visualizations"""

    def __init__(
        self,
        random_state: int = 42,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize advanced analyzer

        Args:
            random_state: Random seed
            logger: Logger instance
        """
        self.random_state = random_state
        self.logger = logger or logging.getLogger(__name__)

    def create_pca_reconstructions(
        self,
        pca_model,
        embeddings: np.ndarray,
        original_images_df: pd.DataFrame,
        output_path: Path,
        n_components_list: List[int] = [5, 10, 20, 50],
        n_samples: int = 4
    ):
        """
        Visualize PCA reconstructions with different component counts

        Args:
            pca_model: Fitted PCA model
            embeddings: Original embeddings
            original_images_df: DataFrame with image information
            output_path: Path to save figure
            n_components_list: List of component counts to try
            n_samples: Number of samples to show
        """
        self.logger.info("Creating PCA reconstructions...")

        # Sample images
        sampled_df = original_images_df.sample(
            n=min(n_samples, len(original_images_df)),
            random_state=self.random_state
        )

        n_rows = n_samples
        n_cols = len(n_components_list) + 1  # +1 for original

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 2, n_rows * 2)
        )

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, (idx, row) in enumerate(sampled_df.iterrows()):
            # Original image
            try:
                img = Image.open(row['image_path']).convert('L')
                img_resized = img.resize((64, 64))
                axes[i, 0].imshow(img_resized, cmap='gray')
                axes[i, 0].set_title('Original')
                axes[i, 0].axis('off')

                # Get embedding for this image
                embedding = embeddings[idx:idx+1]

                # Transform to PCA space
                pca_embedding = pca_model.transform(embedding)

                # Reconstruct with different components
                for j, n_comp in enumerate(n_components_list):
                    # Keep only n_comp components
                    pca_truncated = pca_embedding.copy()
                    pca_truncated[:, n_comp:] = 0

                    # Inverse transform
                    reconstructed = pca_model.inverse_transform(pca_truncated)

                    # Reshape to image (approximate visualization)
                    # Note: This is just for visualization purposes
                    # The embedding space doesn't directly map to pixel space
                    recon_vis = reconstructed[0][:64*64].reshape(64, 64)
                    recon_vis = (recon_vis - recon_vis.min()) / (recon_vis.max() - recon_vis.min() + 1e-8)

                    axes[i, j+1].imshow(recon_vis, cmap='gray')
                    axes[i, j+1].set_title(f'{n_comp} PC')
                    axes[i, j+1].axis('off')

            except Exception as e:
                self.logger.warning(f"Error creating reconstruction: {e}")
                for j in range(n_cols):
                    axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center')
                    axes[i, j].axis('off')

        plt.suptitle('PCA Reconstruction Quality', fontsize=14, y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved PCA reconstructions to {output_path}")

    def visualize_pca_loadings(
        self,
        pca_model,
        output_path: Path,
        n_components: int = 10
    ):
        """
        Visualize PCA component loadings

        Args:
            pca_model: Fitted PCA model
            output_path: Path to save figure
            n_components: Number of components to visualize
        """
        self.logger.info("Visualizing PCA loadings...")

        components = pca_model.components_[:n_components]
        var_explained = pca_model.explained_variance_ratio_[:n_components]

        n_cols = 5
        n_rows = (n_components + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        axes = axes.flatten()

        for i in range(n_components):
            component = components[i]

            # Reshape for visualization (approximate)
            size = int(np.sqrt(len(component)))
            if size * size == len(component):
                component_2d = component.reshape(size, size)
            else:
                # Take first size*size elements
                size = int(np.sqrt(len(component)))
                component_2d = component[:size*size].reshape(size, size)

            # Normalize
            component_2d = (component_2d - component_2d.min()) / (component_2d.max() - component_2d.min() + 1e-8)

            axes[i].imshow(component_2d, cmap='seismic', aspect='auto')
            axes[i].set_title(f'PC{i+1} ({var_explained[i]:.2%})', fontsize=8)
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(n_components, len(axes)):
            axes[i].axis('off')

        plt.suptitle('PCA Principal Components (Loadings)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved PCA loadings to {output_path}")

    def create_cluster_representatives(
        self,
        df: pd.DataFrame,
        cluster_labels: np.ndarray,
        embeddings: np.ndarray,
        output_path: Path,
        n_per_cluster: int = 5
    ):
        """
        Show representative images for each cluster

        Args:
            df: DataFrame with image info
            cluster_labels: Cluster assignments
            embeddings: Image embeddings
            output_path: Path to save figure
            n_per_cluster: Number of images per cluster
        """
        self.logger.info("Creating cluster representative montage...")

        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['cluster'] = cluster_labels

        # Get unique clusters (excluding noise if DBSCAN)
        unique_clusters = sorted([c for c in set(cluster_labels) if c != -1])

        if not unique_clusters:
            self.logger.warning("No clusters found")
            return

        n_clusters = len(unique_clusters)
        fig, axes = plt.subplots(
            n_clusters,
            n_per_cluster,
            figsize=(n_per_cluster * 2, n_clusters * 2)
        )

        if n_clusters == 1:
            axes = axes.reshape(1, -1)

        for i, cluster_id in enumerate(unique_clusters):
            cluster_df = df_clustered[df_clustered['cluster'] == cluster_id]
            cluster_indices = cluster_df.index.tolist()

            if len(cluster_indices) == 0:
                continue

            # Get cluster center
            cluster_embeddings = embeddings[cluster_indices]
            cluster_center = np.mean(cluster_embeddings, axis=0)

            # Find closest images to center
            distances = np.linalg.norm(
                cluster_embeddings - cluster_center,
                axis=1
            )
            closest_indices = np.argsort(distances)[:n_per_cluster]

            for j, idx in enumerate(closest_indices):
                if j >= n_per_cluster:
                    break

                row = cluster_df.iloc[idx]

                try:
                    img = Image.open(row['image_path'])
                    axes[i, j].imshow(img, cmap='gray')
                    axes[i, j].axis('off')

                    if j == 0:
                        axes[i, j].set_ylabel(
                            f'Cluster {cluster_id}\n({len(cluster_df)} imgs)',
                            fontsize=9,
                            rotation=0,
                            ha='right'
                        )
                except Exception as e:
                    self.logger.warning(f"Error loading image: {e}")
                    axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center')
                    axes[i, j].axis('off')

        plt.suptitle('Cluster Representatives', fontsize=14, y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved cluster representatives to {output_path}")

    def create_extreme_samples_montage(
        self,
        df: pd.DataFrame,
        metric: str,
        output_path: Path,
        n_samples: int = 10,
        mode: str = "both"
    ):
        """
        Create montage of extreme samples based on a metric

        Args:
            df: DataFrame with image info
            metric: Metric to use ('mean', 'std', 'masked_mean', etc.)
            output_path: Path to save figure
            n_samples: Number of samples to show
            mode: 'high', 'low', or 'both'
        """
        self.logger.info(
            f"Creating extreme samples montage for {metric}..."
        )

        if metric not in df.columns:
            self.logger.warning(f"Metric {metric} not found in dataframe")
            return

        # Sort by metric
        df_sorted = df.dropna(subset=[metric]).sort_values(metric)

        if mode == "both":
            n_each = n_samples // 2
            sampled = pd.concat([
                df_sorted.head(n_each),
                df_sorted.tail(n_samples - n_each)
            ])
            title_suffix = "Lowest and Highest"
        elif mode == "low":
            sampled = df_sorted.head(n_samples)
            title_suffix = "Lowest"
        else:  # high
            sampled = df_sorted.tail(n_samples)
            title_suffix = "Highest"

        n_cols = 5
        n_rows = (len(sampled) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        axes = axes.flatten()

        for i, (_, row) in enumerate(sampled.iterrows()):
            try:
                img = Image.open(row['image_path'])
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(
                    f'{row["class"]}\n{metric}={row[metric]:.1f}',
                    fontsize=8
                )
                axes[i].axis('off')
            except Exception as e:
                self.logger.warning(f"Error loading image: {e}")
                axes[i].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[i].axis('off')

        # Hide unused subplots
        for i in range(len(sampled), len(axes)):
            axes[i].axis('off')

        plt.suptitle(
            f'Extreme Samples - {title_suffix} {metric}',
            fontsize=14
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved extreme samples montage to {output_path}")

"""
Dimensionality reduction using PCA, UMAP, and t-SNE
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available")


class DimensionalityReducer:
    """Perform dimensionality reduction on embeddings"""

    def __init__(
        self,
        random_state: int = 42,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize dimensionality reducer

        Args:
            random_state: Random seed for reproducibility
            logger: Logger instance
        """
        self.random_state = random_state
        self.logger = logger or logging.getLogger(__name__)

        self.pca_model = None
        self.umap_model = None
        self.tsne_model = None

    def fit_pca(
        self,
        embeddings: np.ndarray,
        n_components: int = 50
    ) -> Tuple[np.ndarray, PCA]:
        """
        Fit PCA and transform embeddings

        Args:
            embeddings: Input embeddings
            n_components: Number of PCA components

        Returns:
            (transformed_embeddings, pca_model)
        """
        self.logger.info(
            f"Fitting PCA with {n_components} components..."
        )

        self.pca_model = PCA(
            n_components=n_components,
            random_state=self.random_state
        )
        pca_embeddings = self.pca_model.fit_transform(embeddings)

        # Log variance explained
        var_explained = self.pca_model.explained_variance_ratio_
        cumsum_var = np.cumsum(var_explained)

        self.logger.info(
            f"PCA variance explained by top 10 components: "
            f"{cumsum_var[9]:.3f}"
        )
        self.logger.info(
            f"PCA variance explained by all {n_components} components: "
            f"{cumsum_var[-1]:.3f}"
        )

        return pca_embeddings, self.pca_model

    def plot_pca_scree(
        self,
        output_path: Path,
        max_components: int = 50
    ):
        """Create scree plot for PCA"""
        if self.pca_model is None:
            self.logger.warning("PCA not fitted yet")
            return

        var_explained = self.pca_model.explained_variance_ratio_
        cumsum_var = np.cumsum(var_explained)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Individual variance explained
        ax1.bar(
            range(1, len(var_explained) + 1),
            var_explained,
            alpha=0.7
        )
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Variance Explained')
        ax1.set_title('PCA Scree Plot')
        ax1.set_xlim(0, min(max_components, len(var_explained)))

        # Cumulative variance explained
        ax2.plot(
            range(1, len(cumsum_var) + 1),
            cumsum_var,
            marker='o',
            linestyle='-'
        )
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95%')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Variance Explained')
        ax2.set_title('Cumulative Variance Explained')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, min(max_components, len(cumsum_var)))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved PCA scree plot to {output_path}")

    def save_pca_components(
        self,
        output_path: Path,
        top_n: int = 10
    ):
        """Save top PCA components to CSV"""
        if self.pca_model is None:
            self.logger.warning("PCA not fitted yet")
            return

        var_explained = self.pca_model.explained_variance_ratio_
        cumsum_var = np.cumsum(var_explained)

        pca_df = pd.DataFrame({
            'component': range(1, min(top_n, len(var_explained)) + 1),
            'variance_explained': var_explained[:top_n],
            'cumulative_variance': cumsum_var[:top_n]
        })

        pca_df.to_csv(output_path, index=False)
        self.logger.info(
            f"Saved top {top_n} PCA components to {output_path}"
        )

    def fit_umap(
        self,
        embeddings: np.ndarray,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 2,
        metric: str = 'cosine'
    ) -> np.ndarray:
        """
        Fit UMAP and transform embeddings

        Args:
            embeddings: Input embeddings
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            n_components: Number of output dimensions
            metric: Distance metric

        Returns:
            UMAP-transformed embeddings
        """
        if not UMAP_AVAILABLE:
            self.logger.error("UMAP not available")
            return np.array([])

        self.logger.info(
            f"Fitting UMAP (n_neighbors={n_neighbors}, "
            f"min_dist={min_dist})..."
        )

        self.umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=self.random_state,
            n_jobs=-1
        )

        umap_embeddings = self.umap_model.fit_transform(embeddings)

        self.logger.info(
            f"UMAP transformation complete: {umap_embeddings.shape}"
        )

        return umap_embeddings

    def fit_tsne(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
        perplexity: int = 30,
        max_samples: Optional[int] = 5000
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit t-SNE and transform embeddings

        Args:
            embeddings: Input embeddings
            n_components: Number of output dimensions
            perplexity: t-SNE perplexity parameter
            max_samples: Maximum samples to use (t-SNE is slow)

        Returns:
            (tsne_embeddings, sample_indices)
        """
        self.logger.info(
            f"Fitting t-SNE (perplexity={perplexity})..."
        )

        # Sample if dataset is too large
        sample_indices = None
        if max_samples and len(embeddings) > max_samples:
            self.logger.info(
                f"Sampling {max_samples} points for t-SNE"
            )
            rng = np.random.RandomState(self.random_state)
            sample_indices = rng.choice(
                len(embeddings),
                size=max_samples,
                replace=False
            )
            sample_indices.sort()
            embeddings_sample = embeddings[sample_indices]
        else:
            embeddings_sample = embeddings

        self.tsne_model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=self.random_state,
            n_jobs=-1
        )

        tsne_embeddings = self.tsne_model.fit_transform(embeddings_sample)

        self.logger.info(
            f"t-SNE transformation complete: {tsne_embeddings.shape}"
        )

        return tsne_embeddings, sample_indices

    def save_projections(
        self,
        projections: np.ndarray,
        filenames: list,
        labels: list,
        output_path: Path,
        method: str = "umap"
    ):
        """
        Save dimensionality reduction projections

        Args:
            projections: 2D projections
            filenames: List of filenames
            labels: List of class labels
            output_path: Output CSV path
            method: Name of the method (for column naming)
        """
        df = pd.DataFrame({
            'filename': filenames,
            'label': labels,
            f'{method}_1': projections[:, 0],
            f'{method}_2': projections[:, 1]
        })

        df.to_csv(output_path, index=False)
        self.logger.info(
            f"Saved {method.upper()} projections to {output_path}"
        )

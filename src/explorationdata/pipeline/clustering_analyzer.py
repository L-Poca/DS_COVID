"""
Clustering analysis with KMeans and DBSCAN
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


class ClusteringAnalyzer:
    """Perform clustering analysis on embeddings"""

    def __init__(
        self,
        random_state: int = 42,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize clustering analyzer

        Args:
            random_state: Random seed for reproducibility
            logger: Logger instance
        """
        self.random_state = random_state
        self.logger = logger or logging.getLogger(__name__)

        self.kmeans_model = None
        self.dbscan_model = None

    def fit_kmeans(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 4
    ) -> np.ndarray:
        """
        Fit KMeans clustering

        Args:
            embeddings: Input embeddings
            n_clusters: Number of clusters

        Returns:
            Cluster labels
        """
        self.logger.info(
            f"Fitting KMeans with {n_clusters} clusters..."
        )

        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )

        labels = self.kmeans_model.fit_predict(embeddings)

        # Calculate silhouette score
        try:
            sil_score = silhouette_score(embeddings, labels)
            self.logger.info(f"KMeans silhouette score: {sil_score:.3f}")
        except Exception as e:
            self.logger.warning(f"Could not compute silhouette score: {e}")

        return labels

    def fit_dbscan(
        self,
        embeddings: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> np.ndarray:
        """
        Fit DBSCAN clustering

        Args:
            embeddings: Input embeddings
            eps: DBSCAN eps parameter
            min_samples: DBSCAN min_samples parameter

        Returns:
            Cluster labels
        """
        self.logger.info(
            f"Fitting DBSCAN (eps={eps}, min_samples={min_samples})..."
        )

        self.dbscan_model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            n_jobs=-1
        )

        labels = self.dbscan_model.fit_predict(embeddings)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        self.logger.info(
            f"DBSCAN found {n_clusters} clusters and {n_noise} noise points"
        )

        return labels

    def evaluate_clustering(
        self,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        method_name: str = "clustering"
    ) -> Dict[str, float]:
        """
        Evaluate clustering against ground truth labels

        Args:
            true_labels: Ground truth labels
            pred_labels: Predicted cluster labels
            method_name: Name of clustering method

        Returns:
            Dictionary of metrics
        """
        # Convert labels to numeric if needed
        if isinstance(true_labels[0], str):
            unique_labels = sorted(set(true_labels))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            true_labels_numeric = np.array(
                [label_map[label] for label in true_labels]
            )
        else:
            true_labels_numeric = true_labels

        # Calculate metrics
        ari = adjusted_rand_score(true_labels_numeric, pred_labels)
        nmi = normalized_mutual_info_score(true_labels_numeric, pred_labels)

        self.logger.info(f"{method_name} - ARI: {ari:.3f}, NMI: {nmi:.3f}")

        return {
            'method': method_name,
            'ari': ari,
            'nmi': nmi
        }

    def compute_similarity_matrix(
        self,
        embeddings: np.ndarray,
        sample_size: Optional[int] = 1000
    ) -> np.ndarray:
        """
        Compute cosine similarity matrix

        Args:
            embeddings: Input embeddings
            sample_size: Sample size for large datasets

        Returns:
            Similarity matrix
        """
        if sample_size and len(embeddings) > sample_size:
            self.logger.info(
                f"Sampling {sample_size} points for similarity matrix"
            )
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(
                len(embeddings),
                size=sample_size,
                replace=False
            )
            embeddings_sample = embeddings[indices]
        else:
            embeddings_sample = embeddings

        self.logger.info("Computing cosine similarity matrix...")
        sim_matrix = cosine_similarity(embeddings_sample)

        return sim_matrix

    def plot_similarity_heatmap(
        self,
        similarity_matrix: np.ndarray,
        labels: Optional[list] = None,
        output_path: Path = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot similarity matrix as heatmap

        Args:
            similarity_matrix: Similarity matrix
            labels: Optional class labels for ordering
            output_path: Path to save plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)

        # If labels provided, sort by label
        if labels is not None:
            sorted_indices = np.argsort(labels)
            similarity_matrix = similarity_matrix[sorted_indices][:, sorted_indices]

        sns.heatmap(
            similarity_matrix,
            cmap='coolwarm',
            center=0,
            square=True,
            cbar_kws={'label': 'Cosine Similarity'},
            xticklabels=False,
            yticklabels=False
        )

        plt.title('Image Similarity Matrix (Cosine)')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved similarity heatmap to {output_path}")
        plt.close()

    def plot_inter_class_similarity(
        self,
        embeddings: np.ndarray,
        labels: list,
        output_path: Path,
        sample_per_class: int = 100
    ):
        """
        Plot average inter-class similarity

        Args:
            embeddings: Input embeddings
            labels: Class labels
            output_path: Path to save plot
            sample_per_class: Number of samples per class
        """
        unique_labels = sorted(set(labels))
        n_classes = len(unique_labels)

        # Sample from each class
        sampled_embeddings = []
        sampled_labels = []

        rng = np.random.RandomState(self.random_state)

        for label in unique_labels:
            label_indices = [i for i, l in enumerate(labels) if l == label]
            if len(label_indices) > sample_per_class:
                label_indices = rng.choice(
                    label_indices,
                    size=sample_per_class,
                    replace=False
                )
            sampled_embeddings.extend([embeddings[i] for i in label_indices])
            sampled_labels.extend([label] * len(label_indices))

        sampled_embeddings = np.array(sampled_embeddings)

        # Compute similarity
        sim_matrix = cosine_similarity(sampled_embeddings)

        # Compute inter-class similarity
        inter_class_sim = np.zeros((n_classes, n_classes))

        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels):
                idx1 = [k for k, l in enumerate(sampled_labels) if l == label1]
                idx2 = [k for k, l in enumerate(sampled_labels) if l == label2]

                block = sim_matrix[np.ix_(idx1, idx2)]
                inter_class_sim[i, j] = np.mean(block)

        # Plot heatmap
        plt.figure(figsize=(8, 7))
        sns.heatmap(
            inter_class_sim,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            xticklabels=unique_labels,
            yticklabels=unique_labels,
            cbar_kws={'label': 'Average Cosine Similarity'}
        )
        plt.title('Inter-Class Similarity Matrix')
        plt.xlabel('Class')
        plt.ylabel('Class')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(
            f"Saved inter-class similarity heatmap to {output_path}"
        )

    def save_cluster_results(
        self,
        filenames: list,
        labels: list,
        kmeans_labels: np.ndarray,
        dbscan_labels: np.ndarray,
        output_path: Path
    ):
        """
        Save clustering results to CSV

        Args:
            filenames: List of filenames
            labels: Ground truth labels
            kmeans_labels: KMeans cluster labels
            dbscan_labels: DBSCAN cluster labels
            output_path: Output CSV path
        """
        df = pd.DataFrame({
            'filename': filenames,
            'true_label': labels,
            'kmeans_cluster': kmeans_labels,
            'dbscan_cluster': dbscan_labels
        })

        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved cluster results to {output_path}")

"""
Main pipeline runner for comprehensive EDA
Coordinates all analysis modules with checkpointing and logging
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from .data_loader import DatasetLoader
from .embedding_extractor import EmbeddingExtractor
from .dimensionality_reducer import DimensionalityReducer
from .clustering_analyzer import ClusteringAnalyzer
from .visualizer import Visualizer
from .advanced_analysis import AdvancedAnalyzer


class EDAPipeline:
    """Main pipeline for comprehensive exploratory data analysis"""

    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        output_dir: str,
        seed: int = 42,
        device: Optional[str] = None,
        max_images_per_class: Optional[int] = None
    ):
        """
        Initialize EDA pipeline

        Args:
            base_path: Path to dataset
            metadata_path: Path to metadata directory
            output_dir: Output directory for results
            seed: Random seed for reproducibility
            device: Device for deep learning ('cuda', 'cpu', or None)
            max_images_per_class: Max images per class (None = all)
        """
        self.base_path = Path(base_path)
        self.metadata_path = Path(metadata_path)
        self.seed = seed
        self.device = device
        self.max_images_per_class = max_images_per_class

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize components
        self.data_loader = DatasetLoader(
            base_path,
            metadata_path,
            logger=self.logger
        )
        self.visualizer = Visualizer(
            random_state=seed,
            logger=self.logger
        )
        self.dim_reducer = DimensionalityReducer(
            random_state=seed,
            logger=self.logger
        )
        self.clustering = ClusteringAnalyzer(
            random_state=seed,
            logger=self.logger
        )
        self.advanced = AdvancedAnalyzer(
            random_state=seed,
            logger=self.logger
        )

        # Track timing
        self.timings = {}
        self.start_time = time.time()

        # Summary data
        self.summary = {
            'seed': seed,
            'device': str(device),
            'max_images_per_class': max_images_per_class,
            'timestamp': timestamp,
            'output_dir': str(self.output_dir)
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('EDA_Pipeline')
        logger.setLevel(logging.INFO)

        # File handler
        log_file = self.output_dir / 'log.txt'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def _time_step(self, step_name: str):
        """Record timing for a step"""
        if hasattr(self, '_last_step_time'):
            duration = time.time() - self._last_step_time
            self.timings[self._last_step_name] = duration
            self.logger.info(
                f"Step '{self._last_step_name}' took {duration:.2f} seconds"
            )

        self._last_step_name = step_name
        self._last_step_time = time.time()

    def run_step_1_data_loading(self):
        """Step 1: Load and validate dataset"""
        self.logger.info("="*60)
        self.logger.info("STEP 1: Data Loading and Validation")
        self.logger.info("="*60)

        self._time_step("data_loading")

        # Load all images
        self.image_df = self.data_loader.load_all_images(
            max_images_per_class=self.max_images_per_class
        )

        if self.image_df.empty:
            self.logger.error("No images loaded! Aborting.")
            raise ValueError("No images found in dataset")

        # Save image statistics
        stats_path = self.tables_dir / "image_stats.csv"
        self.image_df.to_csv(stats_path, index=False)
        self.logger.info(f"Saved image statistics to {stats_path}")

        # Save corrupted images list
        corrupted = self.data_loader.get_corrupted_images()
        if corrupted:
            corrupted_df = pd.DataFrame(corrupted)
            corrupted_path = self.tables_dir / "corrupted_images.csv"
            corrupted_df.to_csv(corrupted_path, index=False)
            self.logger.info(
                f"Found {len(corrupted)} corrupted images"
            )

        # Update summary
        self.summary['total_images'] = len(self.image_df)
        self.summary['corrupted_images'] = len(corrupted)
        self.summary['classes'] = self.image_df['class'].unique().tolist()

    def run_step_2_basic_visualizations(self):
        """Step 2: Create basic visualizations"""
        self.logger.info("="*60)
        self.logger.info("STEP 2: Basic Visualizations")
        self.logger.info("="*60)

        self._time_step("basic_visualizations")

        # Class distribution
        self.visualizer.plot_class_distribution(
            self.image_df,
            self.figures_dir / "class_distribution.png"
        )

        # Image dimensions
        self.visualizer.plot_image_dimensions(
            self.image_df,
            self.figures_dir / "image_dimensions.png"
        )

        # Intensity statistics
        self.visualizer.plot_intensity_statistics(
            self.image_df,
            self.figures_dir / "intensity_statistics.png"
        )

        # Sample grids
        self.visualizer.create_image_grid(
            self.image_df,
            self.figures_dir / "sample_grid_random.png",
            n_per_class=5,
            sample_mode="random",
            title="Random Sample Images"
        )

        self.visualizer.create_image_grid(
            self.image_df,
            self.figures_dir / "sample_grid_top.png",
            n_per_class=5,
            sample_mode="top",
            title="First Images per Class"
        )

        # Image-mask overlays
        self.visualizer.create_image_mask_overlay(
            self.image_df,
            self.figures_dir / "image_mask_overlays.png",
            n_samples=8
        )

    def run_step_3_embeddings(self):
        """Step 3: Extract embeddings"""
        self.logger.info("="*60)
        self.logger.info("STEP 3: Embedding Extraction")
        self.logger.info("="*60)

        self._time_step("embeddings")

        # Initialize extractor
        self.embedding_extractor = EmbeddingExtractor(
            model_name="resnet50",
            device=self.device,
            batch_size=64,
            image_size=224,
            logger=self.logger
        )

        # Extract embeddings - full images
        self.embeddings, self.embedding_filenames = \
            self.embedding_extractor.extract_embeddings(
                self.image_df,
                apply_mask=False
            )

        # Save embeddings
        self.embedding_extractor.save_embeddings(
            self.embeddings,
            self.embedding_filenames,
            self.output_dir,
            prefix=""
        )

        # Extract embeddings - masked regions (if masks available)
        if self.image_df['has_mask'].any():
            self.logger.info("Extracting embeddings for masked regions...")
            self.masked_embeddings, self.masked_filenames = \
                self.embedding_extractor.extract_embeddings(
                    self.image_df[self.image_df['has_mask']],
                    apply_mask=True
                )

            self.embedding_extractor.save_embeddings(
                self.masked_embeddings,
                self.masked_filenames,
                self.output_dir,
                prefix="masked_"
            )

        self.summary['embedding_shape'] = list(self.embeddings.shape)

    def run_step_4_dimensionality_reduction(self):
        """Step 4: Dimensionality reduction"""
        self.logger.info("="*60)
        self.logger.info("STEP 4: Dimensionality Reduction")
        self.logger.info("="*60)

        self._time_step("dimensionality_reduction")

        # PCA
        self.pca_embeddings, self.pca_model = self.dim_reducer.fit_pca(
            self.embeddings,
            n_components=50
        )

        # Save PCA results
        self.dim_reducer.plot_pca_scree(
            self.figures_dir / "pca_scree.png"
        )
        self.dim_reducer.save_pca_components(
            self.tables_dir / "pca_top10.csv",
            top_n=10
        )

        # UMAP
        self.umap_embeddings = self.dim_reducer.fit_umap(
            self.embeddings,
            n_neighbors=15,
            min_dist=0.1
        )

        # Get labels for visualization
        labels = [self.image_df.iloc[i]['class']
                  for i in range(len(self.embedding_filenames))
                  if self.image_df.iloc[i]['filename'] in self.embedding_filenames]

        # Save UMAP projections
        self.dim_reducer.save_projections(
            self.umap_embeddings,
            self.embedding_filenames,
            labels,
            self.tables_dir / "umap_proj.csv",
            method="umap"
        )

        # Visualize UMAP
        self.visualizer.plot_2d_embeddings(
            self.umap_embeddings,
            labels,
            self.figures_dir / "umap_scatter.png",
            title="UMAP Projection - Colored by Class"
        )

        # t-SNE (on sample)
        self.tsne_embeddings, self.tsne_indices = self.dim_reducer.fit_tsne(
            self.embeddings,
            max_samples=5000
        )

        if self.tsne_indices is not None:
            tsne_labels = [labels[i] for i in self.tsne_indices]
            tsne_filenames = [self.embedding_filenames[i]
                              for i in self.tsne_indices]
        else:
            tsne_labels = labels
            tsne_filenames = self.embedding_filenames

        # Save t-SNE projections
        self.dim_reducer.save_projections(
            self.tsne_embeddings,
            tsne_filenames,
            tsne_labels,
            self.tables_dir / "tsne_proj.csv",
            method="tsne"
        )

        # Visualize t-SNE
        self.visualizer.plot_2d_embeddings(
            self.tsne_embeddings,
            tsne_labels,
            self.figures_dir / "tsne_scatter.png",
            title="t-SNE Projection - Colored by Class"
        )

    def run_step_5_clustering(self):
        """Step 5: Clustering analysis"""
        self.logger.info("="*60)
        self.logger.info("STEP 5: Clustering Analysis")
        self.logger.info("="*60)

        self._time_step("clustering")

        # Get labels
        labels = [self.image_df.iloc[i]['class']
                  for i in range(len(self.embedding_filenames))
                  if self.image_df.iloc[i]['filename'] in self.embedding_filenames]

        # KMeans
        n_classes = len(set(labels))
        self.kmeans_labels = self.clustering.fit_kmeans(
            self.embeddings,
            n_clusters=n_classes
        )

        # Evaluate
        kmeans_metrics = self.clustering.evaluate_clustering(
            labels,
            self.kmeans_labels,
            "KMeans"
        )

        # DBSCAN
        self.dbscan_labels = self.clustering.fit_dbscan(
            self.embeddings,
            eps=0.5,
            min_samples=5
        )

        # Evaluate
        dbscan_metrics = self.clustering.evaluate_clustering(
            labels,
            self.dbscan_labels,
            "DBSCAN"
        )

        # Save cluster results
        self.clustering.save_cluster_results(
            self.embedding_filenames,
            labels,
            self.kmeans_labels,
            self.dbscan_labels,
            self.tables_dir / "clusters.csv"
        )

        # Similarity analysis
        sim_matrix = self.clustering.compute_similarity_matrix(
            self.embeddings,
            sample_size=1000
        )

        self.clustering.plot_similarity_heatmap(
            sim_matrix,
            output_path=self.figures_dir / "similarity_heatmap.png"
        )

        self.clustering.plot_inter_class_similarity(
            self.embeddings,
            labels,
            self.figures_dir / "inter_class_similarity.png"
        )

        # Visualize clusters on UMAP
        self.visualizer.plot_2d_embeddings(
            self.umap_embeddings,
            self.kmeans_labels,
            self.figures_dir / "umap_kmeans_clusters.png",
            title="UMAP - Colored by KMeans Clusters",
            color_by="continuous"
        )

        self.summary['kmeans_metrics'] = kmeans_metrics
        self.summary['dbscan_metrics'] = dbscan_metrics

    def run_step_6_advanced_analysis(self):
        """Step 6: Advanced analysis and visualizations"""
        self.logger.info("="*60)
        self.logger.info("STEP 6: Advanced Analysis")
        self.logger.info("="*60)

        self._time_step("advanced_analysis")

        # PCA loadings visualization
        self.advanced.visualize_pca_loadings(
            self.pca_model,
            self.figures_dir / "pca_loadings.png"
        )

        # Cluster representatives
        self.advanced.create_cluster_representatives(
            self.image_df,
            self.kmeans_labels,
            self.embeddings,
            self.figures_dir / "cluster_representatives.png",
            n_per_cluster=5
        )

        # Extreme samples
        self.advanced.create_extreme_samples_montage(
            self.image_df,
            'mean',
            self.figures_dir / "extreme_mean_intensity.png",
            n_samples=10,
            mode="both"
        )

        if 'masked_mean' in self.image_df.columns:
            self.advanced.create_extreme_samples_montage(
                self.image_df,
                'masked_mean',
                self.figures_dir / "extreme_masked_mean.png",
                n_samples=10,
                mode="both"
            )

    def finalize(self):
        """Finalize pipeline and save summary"""
        self.logger.info("="*60)
        self.logger.info("Finalizing Pipeline")
        self.logger.info("="*60)

        self._time_step("finalization")

        # Calculate total time
        total_time = time.time() - self.start_time
        self.summary['total_time_seconds'] = total_time
        self.summary['timings'] = self.timings

        # Save summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.summary, f, indent=2)

        self.logger.info(f"Pipeline complete in {total_time:.2f} seconds")
        self.logger.info(f"Results saved to: {self.output_dir}")

    def run_full_pipeline(self):
        """Run the complete pipeline"""
        try:
            self.run_step_1_data_loading()
            self.run_step_2_basic_visualizations()
            self.run_step_3_embeddings()
            self.run_step_4_dimensionality_reduction()
            self.run_step_5_clustering()
            self.run_step_6_advanced_analysis()
            self.finalize()

            return True

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.summary['error'] = str(e)
            self.finalize()
            return False

# Complete EDA Pipeline - COVID-19 Radiography Dataset

## Overview

This pipeline performs comprehensive exploratory data analysis on the COVID-19 radiography dataset, including:

- **Data Loading & Validation**: Load images, masks, and metadata with quality checks
- **Basic Visualizations**: Distribution plots, sample grids, image-mask overlays
- **Deep Learning Embeddings**: Extract features using ResNet50 (GPU/CPU adaptive)
- **Dimensionality Reduction**: PCA, UMAP, t-SNE for visualization
- **Clustering Analysis**: KMeans and DBSCAN with evaluation metrics
- **Advanced Visualizations**: Similarity matrices, cluster representatives, extreme samples

## Quick Start

### Option 1: Run on Google Colab (Recommended)

1. Open the notebook: `notebooks/Complete_EDA_COVID_Dataset.ipynb`
2. Upload to Google Colab
3. Follow the instructions in the notebook
4. Results will be saved to your Google Drive

### Option 2: Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python src/explorationdata/run_eda_pipeline.py \
    --base-path /path/to/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset \
    --metadata-path /path/to/metadata \
    --output-dir outputs \
    --seed 42
```

### Option 3: Use as Python Module

```python
from src.explorationdata.pipeline.pipeline_runner import EDAPipeline

pipeline = EDAPipeline(
    base_path="data/raw/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset",
    metadata_path="metadata",
    output_dir="outputs",
    seed=42,
    device=None,  # Auto-detect CUDA
    max_images_per_class=None  # None = all images
)

success = pipeline.run_full_pipeline()
```

## Dataset Structure

Expected directory structure:

```
data/raw/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset/
├── COVID/
│   ├── images/
│   │   ├── COVID-1.png
│   │   └── ...
│   └── masks/
│       ├── COVID-1.png
│       └── ...
├── Lung_Opacity/
│   ├── images/
│   └── masks/
├── Normal/
│   ├── images/
│   └── masks/
└── Viral Pneumonia/
    ├── images/
    └── masks/

metadata/
├── COVID.metadata.xlsx
├── Lung_Opacity.metadata.xlsx
├── Normal.metadata.xlsx
└── Viral Pneumonia.metadata.xlsx
```

## Pipeline Steps

### Step 1: Data Loading and Validation
- Load all images from all classes
- Validate image integrity (detect corrupted files)
- Extract basic statistics (dimensions, intensity)
- Process masks and calculate masked region statistics
- Save `image_stats.csv`

### Step 2: Basic Visualizations
- Class distribution (bar plot, pie chart)
- Image dimension distributions
- Intensity statistics by class
- Sample image grids (random, top, outliers)
- Image-mask overlay visualizations

### Step 3: Embedding Extraction
- Extract deep learning embeddings using ResNet50
- Process both full images and masked regions
- GPU-accelerated when available, falls back to CPU
- Save embeddings as `.npy` files with metadata

### Step 4: Dimensionality Reduction
- **PCA**: Fit with 50 components, create scree plot, save top components
- **UMAP**: 2D projection with configurable parameters (n_neighbors=15, min_dist=0.1)
- **t-SNE**: 2D projection on sample (max 5000 points for speed)
- Visualize projections colored by class labels

### Step 5: Clustering Analysis
- **KMeans**: Fit with number of clusters = number of classes
- **DBSCAN**: Density-based clustering with configurable parameters
- Evaluate clustering: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI)
- Compute similarity matrices (cosine similarity)
- Create inter-class similarity heatmaps

### Step 6: Advanced Analysis
- PCA loadings visualization
- Cluster representative images
- Extreme sample montages (highest/lowest intensity, masked intensity)
- (Optional) Grad-CAM visualizations

## Output Structure

```
outputs/YYYYMMDD_HHMMSS/
├── figures/
│   ├── class_distribution.png
│   ├── image_dimensions.png
│   ├── intensity_statistics.png
│   ├── sample_grid_random.png
│   ├── sample_grid_top.png
│   ├── image_mask_overlays.png
│   ├── pca_scree.png
│   ├── pca_loadings.png
│   ├── umap_scatter.png
│   ├── tsne_scatter.png
│   ├── umap_kmeans_clusters.png
│   ├── similarity_heatmap.png
│   ├── inter_class_similarity.png
│   ├── cluster_representatives.png
│   └── extreme_*.png
├── tables/
│   ├── image_stats.csv
│   ├── corrupted_images.csv
│   ├── pca_top10.csv
│   ├── umap_proj.csv
│   ├── tsne_proj.csv
│   └── clusters.csv
├── embeddings.npy
├── embeddings_files.csv
├── masked_embeddings.npy (if masks available)
├── masked_embeddings_files.csv
├── summary.json
└── log.txt
```

## Configuration

### Command Line Arguments

```bash
--base-path          Path to dataset (required)
--metadata-path      Path to metadata directory (required)
--output-dir         Output directory (default: outputs)
--seed               Random seed (default: 42)
--device             Device: cuda, cpu, or None for auto (default: None)
--max-images-per-class  Limit images per class (default: None = all)
```

### Key Parameters (in code)

```python
# Embedding extraction
batch_size = 64 (GPU) / 8 (CPU)
image_size = 224x224
model = "resnet50"

# Dimensionality reduction
pca_components = 50
umap_n_neighbors = 15
umap_min_dist = 0.1
tsne_perplexity = 30
tsne_max_samples = 5000

# Clustering
kmeans_n_clusters = number of classes
dbscan_eps = 0.5
dbscan_min_samples = 5
```

## Requirements

### Core Dependencies
- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- opencv-python >= 4.5.0
- pillow >= 8.0.0
- tqdm >= 4.60.0
- openpyxl >= 3.0.0

### Deep Learning Dependencies
- torch >= 1.13.0
- torchvision >= 0.14.0

### Dimensionality Reduction
- umap-learn >= 0.5.0

## Performance

### Expected Runtime
- **Small dataset** (1000 images): 5-10 minutes
- **Medium dataset** (10000 images): 20-40 minutes
- **Full dataset** (20000+ images): 40-90 minutes

Runtime varies based on:
- Number of images
- Image resolution
- Hardware (GPU vs CPU)
- Available RAM

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU
- **Recommended**: 16GB RAM, NVIDIA GPU with 6GB+ VRAM
- **Optimal**: 32GB RAM, NVIDIA GPU with 12GB+ VRAM

## Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
batch_size = 4  # or even 1

# Limit images per class
max_images_per_class = 500

# Use CPU instead of GPU
device = "cpu"
```

### Slow Performance
```python
# Limit t-SNE samples
tsne_max_samples = 1000

# Reduce similarity matrix samples
similarity_sample_size = 500

# Use smaller embedding model
model_name = "resnet18"  # instead of resnet50
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# On Colab, use Colab requirements
pip install -r requirements-colab.txt
```

## Extending the Pipeline

### Add Custom Visualization

```python
# In visualizer.py
def plot_custom_analysis(self, df, output_path):
    # Your custom visualization code
    pass
```

### Add Custom Analysis Step

```python
# In pipeline_runner.py
def run_step_7_custom_analysis(self):
    self.logger.info("Running custom analysis...")
    # Your custom analysis code
    pass

# Add to run_full_pipeline()
def run_full_pipeline(self):
    # ... existing steps ...
    self.run_step_7_custom_analysis()
    self.finalize()
```

### Use Different Embedding Model

```python
# In embedding_extractor.py
# Modify _load_model() to support your model
```

## Citation

If you use this pipeline in your research, please cite:

```
@software{covid_eda_pipeline_2025,
  title={Complete EDA Pipeline for COVID-19 Radiography Dataset},
  author={DS_COVID Project},
  year={2025},
  url={https://github.com/L-Poca/DS_COVID}
}
```

## License

This pipeline is part of the DS_COVID project. See LICENSE file for details.

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review the example notebook

## Changelog

### v1.0.0 (2025-01)
- Initial release
- Complete EDA pipeline with 6 analysis steps
- Support for Colab and local execution
- GPU/CPU adaptive processing
- Comprehensive visualizations and metrics

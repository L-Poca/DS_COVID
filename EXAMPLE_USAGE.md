# EDA Pipeline - Example Usage

## Quick Start Examples

### 1. Run on Google Colab (Recommended for First Time)

1. Open `notebooks/Complete_EDA_COVID_Dataset.ipynb` in Google Colab
2. Mount your Google Drive (where your dataset is stored)
3. Update the paths in the notebook:
   ```python
   BASE_PATH = '/content/drive/MyDrive/DS_COVID/data/raw/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset'
   ```
4. Run all cells
5. Results will be saved to your Drive in the `outputs/` directory

### 2. Run Locally with Command Line

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python src/explorationdata/run_eda_pipeline.py \
    --base-path "data/raw/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset" \
    --metadata-path "metadata" \
    --output-dir "outputs" \
    --seed 42

# Generate report
python src/explorationdata/generate_report.py \
    --output-dir "outputs/20250122_103045" \
    --report-file "outputs/20250122_103045/report.md"
```

### 3. Run with Python API

```python
from src.explorationdata.pipeline.pipeline_runner import EDAPipeline

# Initialize pipeline
pipeline = EDAPipeline(
    base_path="data/raw/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset",
    metadata_path="metadata",
    output_dir="outputs",
    seed=42,
    device=None,  # Auto-detect CUDA
    max_images_per_class=None  # None = all images
)

# Run complete pipeline
success = pipeline.run_full_pipeline()

if success:
    print(f"Results saved to: {pipeline.output_dir}")
```

### 4. Run with Limited Dataset (for Testing)

```bash
# Test with only 100 images per class
python src/explorationdata/run_eda_pipeline.py \
    --base-path "data/raw/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset" \
    --metadata-path "metadata" \
    --output-dir "outputs_test" \
    --max-images-per-class 100 \
    --seed 42
```

## Configuration Options

### Basic Configuration

```python
EDAPipeline(
    base_path="path/to/dataset",      # Required: dataset location
    metadata_path="path/to/metadata", # Required: metadata Excel files
    output_dir="outputs",             # Output directory
    seed=42,                          # Random seed for reproducibility
    device=None,                      # 'cuda', 'cpu', or None (auto)
    max_images_per_class=None         # Limit per class (None = all)
)
```

### Advanced Configuration

Modify parameters in the pipeline modules:

```python
# In embedding_extractor.py
batch_size = 32  # Reduce if OOM
image_size = 224  # Input size for ResNet50

# In dimensionality_reducer.py
n_components = 50  # PCA components
n_neighbors = 15   # UMAP neighbors
min_dist = 0.1     # UMAP min_dist

# In clustering_analyzer.py
eps = 0.5          # DBSCAN eps
min_samples = 5    # DBSCAN min_samples
```

## Output Structure

After running the pipeline, you'll get:

```
outputs/20250122_103045/
├── figures/
│   ├── class_distribution.png
│   ├── sample_grid_random.png
│   ├── umap_scatter.png
│   └── ... (20+ visualizations)
├── tables/
│   ├── image_stats.csv
│   ├── clusters.csv
│   └── ... (6+ data tables)
├── embeddings.npy
├── embeddings_files.csv
├── summary.json
├── log.txt
└── report.md (after running generate_report.py)
```

## Troubleshooting

### Out of Memory (OOM)

```python
# Option 1: Reduce batch size
pipeline.embedding_extractor.batch_size = 8

# Option 2: Use CPU instead of GPU
device = "cpu"

# Option 3: Limit dataset size
max_images_per_class = 500
```

### Slow Performance

```python
# Option 1: Use smaller model
model_name = "resnet18"  # instead of resnet50

# Option 2: Reduce t-SNE samples
tsne_max_samples = 1000  # instead of 5000

# Option 3: Skip some visualizations
# Comment out specific visualization calls in pipeline_runner.py
```

### Missing Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install specific packages
pip install torch torchvision umap-learn
```

## Expected Runtime

- **Small dataset** (1000 images): 5-10 minutes
- **Medium dataset** (10000 images): 20-40 minutes
- **Full dataset** (20000+ images): 40-90 minutes

Runtime depends on:
- Hardware (GPU vs CPU)
- Dataset size
- Image resolution
- Available RAM

## Next Steps After EDA

1. **Review visualizations** in `figures/` directory
2. **Analyze clusters** using `clusters.csv`
3. **Use embeddings** for ML tasks:
   ```python
   import numpy as np
   embeddings = np.load('outputs/TIMESTAMP/embeddings.npy')
   # Use for classification, clustering, etc.
   ```
4. **Read the report** at `report.md` for insights
5. **Iterate on parameters** for deeper analysis

## Support

- Full documentation: `src/explorationdata/README_EDA_PIPELINE.md`
- Example notebook: `notebooks/Complete_EDA_COVID_Dataset.ipynb`
- Report any issues on GitHub

## Tips

1. **Always use a fixed seed** for reproducibility
2. **Start with a small subset** to test (max_images_per_class=100)
3. **Check the log file** (`log.txt`) for detailed progress
4. **Review summary.json** for quick metrics
5. **Generate the report** after pipeline completes for insights

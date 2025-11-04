# 🦠 Détection COVID-19
## Application de détection COVID-19 à partir d'images radiographiques

## 🚀 NEW: Complete EDA Pipeline

A comprehensive exploratory data analysis pipeline for the COVID-19 radiography dataset is now available!

**Features:**
- 📊 Complete dataset processing with validation
- 🧠 Deep learning embeddings (ResNet50)
- 📉 Dimensionality reduction (PCA, UMAP, t-SNE)
- 🔍 Clustering analysis (KMeans, DBSCAN)
- 📈 20+ advanced visualizations
- 📝 Automated report generation
- ☁️ Google Colab ready

**Quick Start:**
```bash
python src/explorationdata/run_eda_pipeline.py \
    --base-path "data/raw/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset" \
    --metadata-path "metadata"
```

**Documentation:**
- [Complete Guide](src/explorationdata/README_EDA_PIPELINE.md)
- [Examples](EXAMPLE_USAGE.md)
- [Summary](src/explorationdata/PIPELINE_SUMMARY.md)
- [Colab Notebook](notebooks/Complete_EDA_COVID_Dataset.ipynb)

---

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
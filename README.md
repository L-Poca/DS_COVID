# 🦠 DS-COVID: COVID-19 Radiography Analysis Package

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-red)](https://opencv.org)

Un package Python complet pour l'analyse d'images radiographiques COVID-19 utilisant l'apprentissage automatique et l'apprentissage profond.

## 🚀 Installation

### Installation depuis PyPI (recommandée)
```bash
pip install ds-covid
```

### Installation depuis les sources
```bash
# Cloner le repository
git clone https://github.com/L-Poca/DS_COVID.git
cd DS_COVID

# Installation en mode développement
pip install -e .

# Ou installation avec dépendances de développement
pip install -e ".[dev]"
```

### Installation pour Google Colab
```bash
pip install "ds-covid[colab]"
```

## 📦 Fonctionnalités

- **🧠 Modèles Deep Learning** : CNN baseline pour classification COVID-19
- **🎭 Application de Masques** : Techniques d'overlay, multiply et extract
- **📊 Visualisations** : Graphiques interactifs et comparaisons de méthodes  
- **⚡ Interface CLI** : Commandes en ligne de commande pour entraînement et prédiction
- **🌐 Support Colab** : Configuration automatique pour Google Colab
- **📱 Interface Streamlit** : Application web interactive

## 🔧 Usage Rapide

### Utilisation en tant que package
```python
from ds_covid import build_baseline_cnn, MaskApplicator, prepare_covid_data

# Créer un modèle CNN
model = build_baseline_cnn(input_shape=(256, 256, 1), num_classes=4)

# Appliquer des masques
mask_app = MaskApplicator()
result, original, mask = mask_app.apply_mask("image.png", "mask.png", method="overlay")

# Préparer les données
X, y = prepare_covid_data("dataset_path", ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"])
```

### Interface en ligne de commande
```bash
# Entraîner un modèle
ds-covid-train --dataset-path ./data --epochs 50 --batch-size 32

# Faire des prédictions
ds-covid-predict --model-path model.h5 --image-path image.png

# Appliquer des masques
ds-covid-apply-masks --dataset-path ./data --method overlay --output ./results

# Lancer l'interface Streamlit
ds-covid-streamlit
```

## 📊 Dataset

Le package est conçu pour fonctionner avec le **COVID-19 Radiography Database** :
- **COVID** : 7,232 images
- **Normal** : 20,384 images  
- **Lung_Opacity** : 12,024 images
- **Viral Pneumonia** : 2,690 images

**Structure attendue :**
```
dataset/
├── COVID/
│   ├── images/
│   └── masks/
├── Normal/
│   ├── images/
│   └── masks/
├── Lung_Opacity/
│   ├── images/
│   └── masks/
└── Viral Pneumonia/
    ├── images/
    └── masks/
```

## 🌐 Google Colab

Utilisation directe dans Colab :

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/L-Poca/DS_COVID/blob/Rafael_new/notebooks/04_apply_masks_colab.ipynb)

```python
# Configuration automatique pour Colab
!pip install ds-covid[colab]
!python -c "from ds_covid.cli import setup_colab; setup_colab()"
```

## 🏗️ Structure du Projet
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
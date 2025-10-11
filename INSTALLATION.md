# Guide d'installation et d'utilisation - DS COVID

## 🎯 Installation

### Installation en mode développement (recommandée)
```bash
# Cloner le projet
git clone https://github.com/L-Poca/DS_COVID.git
cd DS_COVID

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Sur Linux/Mac
# ou .venv\Scripts\activate sur Windows

# Installer le package en mode développement
pip install -e .
```

### Installation depuis PyPI (quand publié)
```bash
pip install ds-covid
```

## 🚀 Utilisation

### Import des modules
```python
# Importer les modules principaux
import features
import models
import explorationdata
import streamlit

# Ou depuis src directement
from src import features, models, explorationdata
```

### Utilisation des fonctionnalités
```python
# Exemple pour les features
from features import some_function

# Exemple pour les modèles
from models import train_model

# Exemple pour l'exploration de données
from explorationdata import analyze_data
```

## 🧪 Test de l'installation

```python
# Vérifier que tout fonctionne
python -c "import features; import models; import explorationdata; print('Installation réussie!')"
```

## 📦 Structure du package

```
ds-covid/
├── src/
│   ├── features/          # Fonctionnalités d'ingénierie des données
│   ├── models/            # Modèles de machine learning
│   ├── explorationdata/   # Outils d'exploration de données
│   └── streamlit/         # Application web Streamlit
├── notebooks/             # Notebooks Jupyter
├── tests/                 # Tests unitaires
└── data/                  # Données (non incluses dans le package)
```

## 🛠 Développement

### Installer les dépendances de développement
```bash
pip install -e ".[dev]"
```

### Lancer les tests
```bash
pytest
```

### Linter le code
```bash
ruff check src/
ruff format src/
```

## 📝 Publication sur PyPI

### Construire le package
```bash
pip install build twine
python -m build
```

### Publier sur PyPI
```bash
twine upload dist/*
```

## 🔧 Commandes disponibles

Une fois installé, vous aurez accès à :
- `ds-covid-streamlit` : Lancer l'application Streamlit

## 📋 Dépendances principales

- TensorFlow >= 2.10.0
- OpenCV >= 4.5.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Matplotlib >= 3.5.0
- Seaborn >= 0.11.0
- Scikit-learn >= 1.0.0
- Streamlit >= 1.10.0

## ❓ Problèmes courants

### Permission denied lors de l'installation
```bash
# Utiliser un environnement virtuel
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Module non trouvé après installation
```bash
# Vérifier l'installation
pip show ds-covid
pip list | grep ds-covid
```
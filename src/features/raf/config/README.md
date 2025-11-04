# Configuration JSON du projet COVID-19

## 📁 Structure

```
config/
├── default_config.json    # Configuration par défaut (tous les paramètres)
├── colab_config.json      # Surcharges spécifiques à Google Colab
└── README.md             # Ce fichier
```

## 🎯 Principe

La configuration est maintenant **entièrement basée sur JSON** (plus de fichiers `.env`).

### Avantages
- ✅ **Plus clair** : structure hiérarchique lisible
- ✅ **Plus simple** : pas de parsing de variables d'environnement
- ✅ **Plus flexible** : surcharges par environnement faciles
- ✅ **Pas de secrets** : ces paramètres ne sont pas sensibles

## 📋 Utilisation

### Chargement automatique

```python
from src.features.raf.utils.config import get_config

# Détecte automatiquement l'environnement et charge la bonne config
config = get_config()

print(config.batch_size)  # 32
print(config.img_size)    # (256, 256)
print(config.classes)     # ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
```

### Environnements supportés

1. **Local / WSL** : Utilise `default_config.json`
2. **Google Colab** : Fusionne `default_config.json` + `colab_config.json`

## 🔧 Modification de la configuration

### Pour changer un paramètre global

Éditer `default_config.json` :

```json
{
  "training": {
    "batch_size": 64,  // Changé de 32 à 64
    "epochs": 100      // Changé de 50 à 100
  }
}
```

### Pour un paramètre spécifique à Colab

Éditer `colab_config.json` :

```json
{
  "training": {
    "batch_size": 16   // Colab a moins de RAM, on réduit
  }
}
```

Le système fusionne intelligemment les deux fichiers.

## 📊 Structure de la configuration

### Sections principales

- **paths** : Chemins du projet, données, modèles, résultats
- **images** : Dimensions et canaux des images
- **training** : Batch size, epochs, learning rate, splits
- **dataset** : Classes du dataset
- **models** :
  - `random_forest` : Paramètres RF
  - `xgboost` : Paramètres XGBoost
  - `transfer_learning` : Poids pré-entraînés, fine-tuning
- **callbacks** : Early stopping, reduce LR
- **cross_validation** : Nombre de folds, n_jobs
- **system** : Verbosité, niveau de log
- **memory** : Limites mémoire, tailles d'échantillons
- **visualization** : Style matplotlib, couleurs, tailles
- **export** : Formats de sauvegarde

## 🔄 Fonctionnement technique

1. **Chargement** : `default_config.json` est chargé
2. **Surcharge** : Si Colab détecté, `colab_config.json` est fusionné (deep merge)
3. **Aplatissement** : Structure hiérarchique → clés plates (`images_width`, `training_batch_size`, etc.)
4. **Conversion** : Dictionnaire JSON → objet `Config` Python (dataclass)
5. **Post-init** : Calculs dérivés (`num_classes`, `img_size`, création de dossiers)

## 💡 Exemples

### Ajouter un nouveau paramètre

1. Dans `default_config.json` :
```json
{
  "training": {
    "optimizer": "adam",
    "momentum": 0.9
  }
}
```

2. Dans `config.py`, ajouter au dataclass :
```python
@dataclass
class Config:
    # ... autres champs ...
    optimizer: str = "adam"
    momentum: float = 0.9
```

3. Dans `get_project_config()`, ajouter au mapping :
```python
field_mapping = {
    # ... autres mappings ...
    'training_optimizer': ('optimizer', str),
    'training_momentum': ('momentum', float),
}
```

### Créer un environnement custom

Créer `local_gpu_config.json` :
```json
{
  "training": {
    "batch_size": 128,
    "epochs": 200
  },
  "system": {
    "verbose": 2
  }
}
```

Modifier `get_project_config()` pour le charger conditionnellement.

## 🚫 Migration depuis .env

**Ancien système** (`.env`) :
```bash
BATCH_SIZE=32
EPOCHS=50
IMG_WIDTH=256
```

**Nouveau système** (JSON) :
```json
{
  "training": {
    "batch_size": 32,
    "epochs": 50
  },
  "images": {
    "width": 256
  }
}
```

Les fichiers `.env` existants sont **ignorés** maintenant.

# Configurations Pipelines JSON

Ce dossier contient les fichiers de configuration JSON pour les différents pipelines de traitement d'images utilisés dans le projet de détection COVID-19.

## Fichiers de configuration

### `pipeline_config.json`
Fichier de registre principal qui référence tous les pipelines disponibles. Il contient:
- **default**: Pipeline simple par défaut
- **simple**: Pipeline simple et rapide pour classification de base
- **augmented**: Pipeline avec augmentation de données et masquage
- **feature_engineering**: Pipeline avec feature engineering avancé
- **composite**: Pipeline composite avec visualisation
- **composite_no_viz**: Pipeline composite sans visualisation (plus rapide)

### Pipelines disponibles

#### `pipeline_simple.json`
Pipeline de base pour classification rapide.
- **Catégorie**: data_processing
- **Étapes**:
  - ImageLoader
  - ImageResizer (256x256)
  - ImageNormalizer
  - ImageFlattener
  - LogisticRegression

#### `pipeline_augmented.json`
Pipeline avec augmentation de données et masquage pour améliorer la robustesse.
- **Catégorie**: data_processing
- **Étapes**:
  - ImageLoader
  - ImageResizer (256x256)
  - ImageNormalizer
  - ImageAugmenter
  - ImageMasker
  - ImageFlattener
  - ImageStandardScaler
  - LogisticRegression

#### `pipeline_feature_engineering.json`
Pipeline avec feature engineering avancé pour extraction de caractéristiques complexes (histogrammes, PCA).
- **Catégorie**: feature_engineering
- **Étapes**:
  - ImageLoader
  - ImageResizer (256x256)
  - ImageNormalizer
  - ImageMasker
  - ImageFlattener
  - ImageHistogram (64 bins)
  - ImagePCA (50 composantes)
  - LogisticRegression

#### `pipeline_composite_example.json`
Exemple de pipeline composite avec visualisation des résultats. Combine d'autres pipelines.
- **Catégorie**: composite
- **Étapes**:
  - Preprocessing (depuis pipeline_simple, sans clf et flatten)
  - VisualizeTransformer
  - ImageFlattener
  - Feature Engineering (depuis pipeline_feature_engineering: histogram + PCA)
  - LogisticRegression

#### `pipeline_composite_no_viz.json`
Pipeline composite sans visualisation, optimisé pour la vitesse.
- **Catégorie**: composite
- **Étapes**:
  - ImageLoader
  - ImageResizer (256x256)
  - ImageNormalizer
  - ImageFlattener
  - LogisticRegression

## Utilisation

Pour charger un pipeline depuis les notebooks, utilisez le chemin relatif vers ce dossier:

```python
import json

def load_pipeline_config(pipeline_file):
    """Charge la configuration d'un pipeline depuis un fichier JSON."""
    config_path = "../features/Pipelines/Configs_Pipelines/" + pipeline_file
    with open(config_path) as f:
        return json.load(f)

# Charger le registre
registry_config = load_pipeline_config("pipeline_config.json")

# Charger un pipeline spécifique
pipeline_config = load_pipeline_config("pipeline_simple.json")
```

## Structure d'un fichier de pipeline

Chaque fichier de pipeline suit cette structure:

```json
{
  "name": "nom_du_pipeline",
  "description": "Description du pipeline",
  "steps": [
    {
      "name": "nom_etape",
      "class": "NomClasse",
      "params": { "param1": "valeur1", "param2": "valeur2" }
    }
  ]
}
```
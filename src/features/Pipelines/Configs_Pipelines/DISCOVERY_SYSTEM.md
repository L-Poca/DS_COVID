# Système de découverte automatique des pipelines

## 📋 Aperçu

Le nouveau système de découverte automatique remplace l'ancien fichier `pipeline_config.json` statique par une approche dynamique qui scanne automatiquement le dossier `Configs_Pipelines/`.

## 🔍 Fonctionnalités

### Découverte automatique
- **Scan automatique** : Détecte tous les fichiers `*.json` dans le dossier de configurations  
- **Catégorisation intelligente** : Classe automatiquement les pipelines par catégorie
- **Métadonnées extraites** : Récupère nom, description et catégorie depuis chaque fichier

### Fonctions principales

#### `discover_available_pipelines(configs_dir=None)`
Découvre tous les pipelines disponibles
```python
pipelines = discover_available_pipelines()
print(f"Trouvé {len(pipelines)} pipelines")
```

#### `print_available_pipelines(configs_dir=None)` 
Affiche tous les pipelines organisés par catégorie
```python
print_available_pipelines()
```

#### `load_pipeline_by_name(nom_pipeline, masques=None, configs_dir=None)`
Charge un pipeline spécifique par son nom
```python
pipeline = load_pipeline_by_name('tensorflow_pipeline', masks)
```

#### `get_default_pipeline(masques=None, configs_dir=None)`
Charge automatiquement le pipeline par défaut (simple, basic, ou premier disponible)
```python
pipeline = get_default_pipeline(masks)
```

#### `list_pipelines_by_category(category=None, configs_dir=None)`
Liste les pipelines par catégorie
```python
deep_learning_pipelines = list_pipelines_by_category('deep_learning')
```

## 📁 Catégories automatiques

Le système détecte automatiquement les catégories basées sur :

| Catégorie | Critères de détection |
|-----------|----------------------|
| `basic` | Noms contenant 'simple', 'basic' |
| `deep_learning` | Noms avec 'tensorflow', 'keras', 'transfer' ou classes TF |
| `data_augmentation` | Noms avec 'augment' ou classes d'augmentation |
| `feature_engineering` | Noms avec 'feature' ou classes PCA/histogram |
| `composite` | Type 'composite' ou nom contenant 'composite' |
| `data_processing` | Catégorie par défaut |

## 🚀 Utilisation dans les notebooks

```python
# Import des nouvelles fonctions
from src.features.Pipelines.loading_pipelines import (
    print_available_pipelines,
    get_default_pipeline,
    load_pipeline_by_name,
    list_pipelines_by_category
)

# Découvrir tous les pipelines
print_available_pipelines()

# Charger le pipeline par défaut
pipeline = get_default_pipeline(masks)

# Charger un pipeline spécifique
tf_pipeline = load_pipeline_by_name('tensorflow_pipeline', masks)

# Lister par catégorie
deep_learning_names = list_pipelines_by_category('deep_learning')
```

## ✅ Avantages

- **✨ Aucune maintenance manuelle** : Plus besoin de mettre à jour `pipeline_config.json`
- **🔄 Ajout automatique** : Nouveaux pipelines détectés automatiquement
- **📊 Catégorisation intelligente** : Classification automatique
- **🛡️ Robuste** : Gestion d'erreurs et fichiers malformés
- **🔍 Découverte flexible** : Support de dossiers personnalisés

## 🔄 Migration

### Ancien système (deprecated)
```python
config = load_pipeline_config("pipeline_config.json")
pipeline = create_pipeline_from_config(config["simple"])
```

### Nouveau système (recommandé)  
```python
pipeline = load_pipeline_by_name('simple', masks)
# ou plus simple :
pipeline = get_default_pipeline(masks)
```

## 📂 Structure des pipelines supportée

Chaque fichier JSON doit contenir au minimum :
```json
{
  "name": "Nom du pipeline",
  "description": "Description du pipeline",
  "type": "simple|composite", 
  "steps": [...]
}
```

Le champ `category` est optionnel et sera détecté automatiquement si absent.
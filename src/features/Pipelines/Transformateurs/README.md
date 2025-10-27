# Transformateurs TensorFlow pour pipelines scikit-learn

Ce dossier contient les transformateurs TensorFlow/Keras compatibles avec les pipelines scikit-learn.

## Structure modulaire

Le code a été refactorisé en plusieurs modules spécialisés pour une meilleure maintenabilité :

### 📁 Modules disponibles

| Module | Description | Classes/Fonctions |
|--------|-------------|-------------------|
| `tensorflow_feature_extractor.py` | Extraction de caractéristiques avec modèles pré-entraînés | `TensorFlowFeatureExtractor` |
| `keras_classifiers.py` | Classificateurs Keras pour pipelines | `KerasClassifier`, `TransferLearningClassifier` |
| `tensorflow_data_augmenter.py` | Augmentation de données TensorFlow | `TensorFlowDataAugmenter` |
| `model_builders.py` | Fonctions utilitaires de création de modèles | `create_simple_cnn`, `create_advanced_cnn`, `create_lightweight_cnn` |
| `__init__.py` | Module principal d'importation | Tous les éléments ci-dessus |

### 🔧 Utilisation

#### Import recommandé (nouveau style)
```python
# Import depuis le module principal
from src.features.Pipelines.Transformateurs import (
    TensorFlowFeatureExtractor,
    KerasClassifier,
    TransferLearningClassifier,
    TensorFlowDataAugmenter
)

# Ou import depuis modules spécifiques
from src.features.Pipelines.Transformateurs.tensorflow_feature_extractor import TensorFlowFeatureExtractor
from src.features.Pipelines.Transformateurs.keras_classifiers import KerasClassifier
```

#### Import compatible (ancien style)
```python
# Fonctionne toujours pour la compatibilité descendante
from src.features.Pipelines.Transformateurs.tensorflow_transformers import (
    TensorFlowFeatureExtractor,
    KerasClassifier
)
```

### 🎯 Avantages de la refactorisation

- **Maintenabilité** : Code organisé en modules cohérents
- **Réutilisabilité** : Import sélectif des composants nécessaires
- **Lisibilité** : Fichiers plus courts et focalisés
- **Développement** : Facilite les modifications et extensions
- **Tests** : Permet de tester chaque module indépendamment

### 📋 Classes disponibles

#### TensorFlowFeatureExtractor
Extracteur de caractéristiques utilisant des modèles pré-entraînés (VGG, ResNet, etc.)

#### KerasClassifier
Classificateur Keras générique compatible scikit-learn

#### TransferLearningClassifier
Classificateur utilisant le transfer learning avec des modèles pré-entraînés

#### TensorFlowDataAugmenter
Augmenteur de données utilisant les API TensorFlow 2.x

### 🔧 Fonctions utilitaires

- `create_simple_cnn()` : CNN simple pour débuter
- `create_advanced_cnn()` : CNN avancé avec batch normalization
- `create_lightweight_cnn()` : CNN léger pour ressources limitées

### ⚠️ Compatibilité

Le fichier `tensorflow_transformers.py` reste disponible pour la compatibilité mais redirige vers les nouveaux modules. Il est recommandé de migrer vers les nouveaux imports.

### 🚀 Exemple d'utilisation

```python
from sklearn.pipeline import Pipeline
from src.features.Pipelines.Transformateurs import (
    TensorFlowFeatureExtractor,
    KerasClassifier
)

# Pipeline d'extraction de caractéristiques + classification
pipeline = Pipeline([
    ('features', TensorFlowFeatureExtractor(model_name='VGG16')),
    ('classifier', KerasClassifier(epochs=50))
])

# Entraînement
pipeline.fit(X_train, y_train)

# Prédiction
predictions = pipeline.predict(X_test)
```
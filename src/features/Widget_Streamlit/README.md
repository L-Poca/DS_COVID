# Widgets Streamlit Modulaires

Ce dossier contient les widgets modulaires pour l'application Streamlit COVID-19. La refactorisation vise à améliorer la lisibilité, la maintenabilité et la réutilisabilité du code.

## Structure des Widgets

### 📁 Organisation des fichiers

```
Widget_Streamlit/
├── __init__.py                    # Exports centralisés
├── W_Configuration_Commune.py    # Widgets de configuration partagés
├── W_Training.py                  # Widgets pour l'entraînement
├── W_Evaluation.py               # Widgets pour l'évaluation
├── W_Prediction.py               # Widgets pour la prédiction
├── W_Vérifications_Front.py      # Widgets de vérification
└── README.md                     # Documentation
```

## 🔧 Widgets de Configuration Commune

**Fichier**: `W_Configuration_Commune.py`

Contient les composants partagés entre plusieurs pages :

- `create_data_source_config()` - Configuration des sources de données
- `create_general_parameters()` - Paramètres généraux (test size, random state...)
- `create_pipeline_type_selector()` - Sélecteur de type de pipeline
- `create_model_selection_widget()` - Widget de sélection de modèle
- `display_data_statistics()` - Affichage des statistiques des données
- `create_training_options()` - Options d'entraînement communes
- `create_tensorflow_specific_params()` - Paramètres spécifiques TensorFlow
- `show_configuration_summary()` - Résumé de configuration

## 🏋️ Widgets d'Entraînement

**Fichier**: `W_Training.py`

Widgets spécifiques à la page d'entraînement :

- `create_pipeline_configuration_tab()` - Configuration des pipelines
- `create_training_tab()` - Interface d'entraînement
- `create_results_comparison_tab()` - Comparaison des résultats
- `create_data_verification_widget()` - Vérification des données
- `save_training_results_to_session()` - Sauvegarde des résultats

## 📊 Widgets d'Évaluation

**Fichier**: `W_Evaluation.py`

Widgets pour l'évaluation des modèles :

- `create_model_selection_tab()` - Sélection des modèles à évaluer
- `create_detailed_metrics_tab()` - Métriques détaillées
- `create_visualizations_tab()` - Visualisations avancées
- `create_evaluation_report_tab()` - Génération de rapport
- `create_radar_comparison_chart()` - Graphique radar de comparaison
- `generate_evaluation_report()` - Génération automatique de rapport

## 🔮 Widgets de Prédiction

**Fichier**: `W_Prediction.py`

Widgets pour les prédictions :

- `create_prediction_config_sidebar()` - Configuration de prédiction
- `create_simple_prediction_tab()` - Prédiction simple
- `create_batch_prediction_tab()` - Prédiction en lot
- `create_prediction_analysis_tab()` - Analyse des prédictions
- `create_simulated_data_prediction()` - Prédiction avec données simulées
- `create_file_upload_prediction()` - Prédiction avec upload de fichier

## ✅ Widgets de Vérification

**Fichier**: `W_Vérifications_Front.py`

Widgets pour les vérifications système :

- `show_global_status()` - Affichage du statut global

## 🎯 Utilisation

### Import des widgets

```python
from src.features.Widget_Streamlit import (
    create_data_source_config,
    create_pipeline_configuration_tab,
    create_model_selection_tab,
    # ... autres widgets
)
```

### Exemple d'utilisation dans une page

```python
import streamlit as st
from src.features.Widget_Streamlit import create_data_source_config, create_training_tab

# Dans la sidebar
with st.sidebar:
    data_config = create_data_source_config()

# Dans le contenu principal
training_results = create_training_tab(
    selected_configs=['config1', 'config2'],
    pipeline_type='Sklearn',
    data_config=data_config,
    general_params={'test_size': 0.2}
)
```

## 📝 Pages Refactorisées

### Nouvelles pages modulaires créées :

1. **`1_Training_Refactored.py`** - Version modulaire de la page d'entraînement
2. **`2_Evaluation_Refactored.py`** - Version modulaire de la page d'évaluation  
3. **`3_Prediction_Refactored.py`** - Version modulaire de la page de prédiction

### Structure type d'une page refactorisée :

```python
# 1. Imports et configuration
import streamlit as st
from src.features.Widget_Streamlit import widget1, widget2

# 2. Interface principale
st.title("Page Title")

# 3. Sidebar avec widgets de configuration
with st.sidebar:
    config = widget1()

# 4. Onglets principaux avec widgets spécialisés
tab1, tab2 = st.tabs(["Tab1", "Tab2"])

with tab1:
    results = widget2(config)

# 5. Debug info optionnel
if st.sidebar.checkbox("Debug"):
    st.write(debug_info)
```

## 🚀 Avantages de la Refactorisation

### ✅ **Modularité**
- Composants réutilisables entre pages
- Code organisé par fonctionnalité
- Maintenance facilitée

### ✅ **Lisibilité**
- Pages principales très concises
- Logic métier séparée de l'interface
- Structure claire et cohérente

### ✅ **Maintenabilité**
- Modifications centralisées dans les widgets
- Tests unitaires possibles sur chaque widget
- Réduction de la duplication de code

### ✅ **Extensibilité**
- Ajout facile de nouveaux widgets
- Composition flexible des interfaces
- Paramétrage avancé des composants

## 🛠️ Migration depuis les pages originales

Pour migrer du code existant :

1. **Identifier** les blocs de code réutilisables
2. **Extraire** vers un widget avec paramètres
3. **Tester** le widget isolément  
4. **Remplacer** dans la page par l'appel au widget
5. **Valider** le fonctionnement identique

## 📚 Bonnes Pratiques

### Widget Design

- **Un widget = une responsabilité** claire
- **Paramètres explicites** pour la configuration
- **Valeurs de retour** structurées (dict, tuple...)
- **Gestion d'erreurs** robuste avec fallbacks

### Naming Convention

- Préfixe `create_` pour les widgets d'interface
- Préfixe `display_` pour les widgets d'affichage pur
- Préfixe `generate_` pour les widgets de génération de contenu
- Noms descriptifs et auto-explicatifs

### Documentation

- **Docstrings** complètes avec Args et Returns
- **Exemples d'usage** dans les docstrings
- **Comments** pour la logique complexe
- **Type hints** quand possible

## 🔄 Prochaines Étapes

1. **Tests unitaires** pour chaque widget
2. **Documentation** interactive avec des exemples
3. **Optimisation** des performances
4. **Standardisation** des styles et thèmes
5. **Integration** avec d'autres modules du projet
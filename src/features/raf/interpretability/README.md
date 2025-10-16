# 🔍 Module d'Interprétabilité - Framework RAF

Ce module fournit des outils d'interprétabilité pour les modèles de classification COVID-19, intégrant **SHAP** et **GradCAM** dans le framework RAF.

## 📋 Fonctionnalités

### 🎯 GradCAM (Gradient-weighted Class Activation Mapping)
- Visualisation des zones importantes pour les prédictions CNN
- Support de GradCAM et GradCAM++
- Analyse quantitative des cartes d'activation
- Comparaison entre différentes couches

### 📊 SHAP (SHapley Additive exPlanations)
- Explications pour tous types de modèles (tree, linear, deep, kernel)
- Graphiques waterfall, summary et force plots
- Calcul d'importance des features
- Comparaison entre modèles

### ⚖️ Analyse Comparative
- Comparaison SHAP vs GradCAM
- Tableau de bord intégré
- Rapports d'interprétabilité
- Analyse de cohérence

## 🚀 Installation

```bash
# Installation des dépendances
pip install -r requirements-interpretability.txt

# Ou installation manuelle des modules principaux
pip install shap tensorflow opencv-python scikit-learn matplotlib
```

## 📁 Structure du Module

```
src/features/raf/interpretability/
├── __init__.py                    # Exports principaux
├── shap_explainer.py             # Module SHAP
├── gradcam_explainer.py          # Module GradCAM
└── utils.py                      # Utilitaires et analyseur intégré
```

## 💻 Utilisation Rapide

### Import du Module

```python
from src.features.raf.interpretability import (
    SHAPExplainer,
    GradCAMExplainer,
    InterpretabilityAnalyzer
)
```

### GradCAM pour CNN

```python
# Initialisation
gradcam_explainer = GradCAMExplainer(cnn_model)

# Génération de la carte d'activation
results = gradcam_explainer.generate_gradcam(image)

# Visualisation
fig = gradcam_explainer.plot_gradcam(results, class_names=class_names)
```

### SHAP pour Modèles ML

```python
# Initialisation
shap_explainer = SHAPExplainer(ml_model, model_type='tree')

# Ajustement sur données d'arrière-plan
shap_explainer.fit_explainer(X_background)

# Calcul des valeurs SHAP
shap_values = shap_explainer.explain(X_test)

# Visualisations
shap_explainer.plot_waterfall(0, feature_names=feature_names)
shap_explainer.plot_summary(feature_names=feature_names)
```

### Analyse Comparative

```python
# Analyseur intégré
analyzer = InterpretabilityAnalyzer(
    cnn_model=cnn_model,
    ml_model=ml_model
)

# Analyse comparative
results = analyzer.compare_predictions(
    img=image,
    X_flat=image_flat,
    X_background=X_background,
    class_names=class_names
)

# Tableau de bord
dashboard = create_interpretability_dashboard(
    analyzer, image, image_flat, X_background, class_names
)
```

## 🖥️ Interface Streamlit

Une interface web complète est disponible :

```bash
streamlit run src/streamlit/pages/03_Interpretability.py
```

Fonctionnalités de l'interface :
- 🖼️ **Analyse Simple** : Upload et analyse d'une image
- 📊 **Comparaison** : Comparaison entre modèles
- 📈 **Rapport Batch** : Analyse de plusieurs images
- 🔧 **Utilitaires** : Configuration et tests

## 📓 Notebook de Démonstration

Un notebook complet est disponible pour apprendre et tester :

```
notebooks/Interpretability_SHAP_GradCAM_Demo.ipynb
```

Le notebook couvre :
- Configuration et imports
- Démonstration GradCAM
- Démonstration SHAP
- Analyse comparative
- Cas d'usage médical
- Génération de rapports

## 🏥 Cas d'Usage Médical

### Diagnostic Assisté par IA

```python
# Analyse d'une radiographie
gradcam_result = gradcam_explainer.generate_gradcam(xray_image)

predicted_class = gradcam_result['predicted_class']
confidence = gradcam_result['prediction_confidence']

# Interprétation médicale
if predicted_class == COVID_CLASS:
    print("⚠️ ALERTE: Suspicion COVID-19 détectée")
    print(f"Confiance: {confidence:.1%}")
    
# Zones d'attention pour le radiologue
features = extract_gradcam_features(gradcam_result['heatmap'])
regions = analyze_gradcam_regions(gradcam_result['heatmap'])
```

### Rapport Médical Automatisé

```python
# Génération de rapport pour plusieurs patients
report_df = generate_interpretability_report(
    analyzer=analyzer,
    images=patient_images,
    X_background=reference_images,
    class_names=disease_classes,
    sample_names=patient_ids
)

# Export pour dossier médical
report_df.to_csv('rapport_interpretabilite_patients.csv')
```

## 📊 API Principale

### Classes Principales

- **`SHAPExplainer`** : Explications SHAP pour tous modèles
- **`GradCAMExplainer`** : Cartes d'activation pour CNN
- **`InterpretabilityAnalyzer`** : Analyseur intégré SHAP+GradCAM

### Fonctions Utilitaires

- **`explain_prediction()`** : Explication rapide d'une prédiction
- **`generate_gradcam()`** : Génération rapide de GradCAM
- **`create_interpretability_dashboard()`** : Tableau de bord complet
- **`generate_interpretability_report()`** : Rapport batch

### Fonctions de Visualisation

- **`visualize_shap_values()`** : Visualisations SHAP personnalisées
- **`visualize_gradcam_comparison()`** : Comparaison de méthodes GradCAM
- **`plot_interpretability_summary()`** : Résumé visuel

## ⚙️ Configuration

### Types de Modèles Supportés

**SHAP :**
- `tree` : Random Forest, XGBoost, LightGBM
- `linear` : Régression linéaire, SVM linéaire
- `deep` : Réseaux de neurones TensorFlow/Keras
- `kernel` : Tout modèle avec fonction predict

**GradCAM :**
- Modèles CNN TensorFlow/Keras
- Couches convolutionnelles 2D
- Classification multi-classe

### Paramètres Recommandés

```python
# GradCAM
gradcam_params = {
    'alpha': 0.4,          # Transparence superposition
    'colormap': 'jet',     # Colormap heatmap
}

# SHAP
shap_params = {
    'max_evals': 100,      # Évaluations pour kernel
    'model_type': 'auto',  # Détection automatique
}
```

## 🔧 Intégration dans le Framework RAF

Le module s'intègre naturellement dans RAF :

```python
# Import du framework complet
from src.features.raf import *

# Les modules d'interprétabilité sont automatiquement disponibles
analyzer = InterpretabilityAnalyzer(cnn_model, ml_model)
```

## 📚 Ressources et Références

### Documentation
- [SHAP Documentation](https://shap.readthedocs.io/)
- [GradCAM Paper](https://arxiv.org/abs/1610.02391)
- [GradCAM++ Paper](https://arxiv.org/abs/1710.11063)

### Exemples d'Utilisation
- `notebooks/Interpretability_SHAP_GradCAM_Demo.ipynb`
- `src/streamlit/pages/03_Interpretability.py`

### Tests et Validation
- Tests unitaires avec données COVID-19
- Validation sur modèles réels
- Comparaison avec experts médicaux

## 🚨 Bonnes Pratiques Médicales

### ⚠️ Avertissements
- **L'IA est un outil d'aide**, pas de remplacement du diagnostic médical
- **Toujours valider** avec un expert médical
- **Définir des seuils** de confiance appropriés
- **Documenter les décisions** pour traçabilité

### ✅ Recommandations
- Utiliser **SHAP ET GradCAM** pour plus de robustesse
- **Tester régulièrement** sur de nouveaux cas
- **Former les utilisateurs** à l'interprétation
- **Maintenir une supervision** médicale

## 🤝 Contribution

Pour contribuer au module d'interprétabilité :

1. **Fork** le repository
2. **Créer une branche** pour votre fonctionnalité
3. **Tester** avec le notebook de démonstration
4. **Documenter** vos ajouts
5. **Soumettre une pull request**

## 📄 Licence

Ce module fait partie du projet DS_COVID et suit la même licence.

---

**🔍 L'interprétabilité est essentielle pour l'IA médicale de confiance !**
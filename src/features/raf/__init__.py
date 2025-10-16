# =================================
# RAW AUGMENTATION FRAMEWORK (RAF)
# =================================
"""
Framework complet d'apprentissage automatique pour le projet DS_COVID
Architecture modulaire pour data science professionnelle.

Modules:
- data: Chargement, preprocessing et division des données
- augmentation: Augmentation d'images personnalisée
- models: Modèles ML/DL (baseline, boosting, deep learning, ensemble)
- evaluation: Métriques, visualisation et comparaison
- interpretability: Interprétabilité avec SHAP et GradCAM
- utils: Configuration et utilitaires
"""

__version__ = "2.0.0"
__author__ = "DS_COVID Team"
__description__ = "Framework ML/DL complet pour classification d'images COVID-19"

# Imports principaux
try:
    # Utils (prioritaire - configuration universelle)
    from .utils import Config, get_config, setup_universal_environment
    
    # Data
    from .data import DataLoader, MaskProcessor
    
    # Augmentation
    from .augmentation import CustomImageAugmenter
    
    # Interpretability
    from .interpretability import SHAPExplainer, GradCAMExplainer
    
    print(f"🎨 RAF (Raw Augmentation Framework) v{__version__} chargé avec succès")
    print(f"✨ NOUVELLE FONCTIONNALITÉ: setup_universal_environment() remplace la cellule 1!")
    print(f"� NOUVEAU MODULE: interpretability (SHAP + GradCAM)")
    print(f"��������� Modules disponibles: utils, data, augmentation, interpretability")

except ImportError as e:
    print(f"⚠️ Erreur import RAF: {e}")
    print("💡 Certains modules peuvent ne pas être disponibles")

# Configuration par défaut
DEFAULT_CONFIG = {
    'img_size': (256, 256),
    'batch_size': 32,
    'random_seed': 42,
    'classes': ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
}
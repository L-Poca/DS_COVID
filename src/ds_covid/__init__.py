"""
DS-COVID: Package d'analyse de radiographies COVID-19
=====================================================

Ce package fournit des outils pour l'analyse de radiographies pulmonaires
et la classification COVID-19 en utilisant des techniques de machine learning
et deep learning avancées.

Modules principaux:
- config: Configuration centralisée
- models: Modèles ML et DL
- data: Utilitaires de données
- utils: Fonctions utilitaires

Usage basique:
    >>> from ds_covid.config import get_settings, configure_package
    >>> settings = configure_package(data_dir="./data")
    >>> print(f"Classes: {settings.data.class_names}")
"""

__version__ = "0.1.0"
__author__ = "Rafael Cepa, Cirine, Steven Moire"
__email__ = "rafael.cepa@example.fr"

# Imports principaux
from .config import Settings, get_settings, configure_package

__all__ = [
    'Settings',
    'get_settings', 
    'configure_package',
    '__version__'
]
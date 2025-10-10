"""
DS-COVID: COVID-19 Radiography Analysis Package

A comprehensive Python package for analyzing COVID-19 radiography images
using machine learning and deep learning techniques.

Authors:
    - Rafael Cepa <rafael.cepa@cnrs-orleans.fr>
    - Cirine Moire
    - Steven Moire

License: MIT
"""

__version__ = "0.1.0"
__author__ = "Rafael Cepa, Cirine Moire, Steven Moire"
__email__ = "rafael.cepa@cnrs-orleans.fr"
__license__ = "MIT"

# Main package imports - will be populated after creating modules
try:
    from .models import build_baseline_cnn, MaskApplicator
    from .features import load_images_flat, prepare_covid_data
    from .visualization import visualize_samples, compare_methods
    
    __all__ = [
        "build_baseline_cnn",
        "MaskApplicator", 
        "load_images_flat",
        "prepare_covid_data",
        "visualize_samples",
        "compare_methods",
    ]
except ImportError:
    # During development, modules might not exist yet
    __all__ = []
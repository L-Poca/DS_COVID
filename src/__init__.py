"""DS COVID - COVID-19 Radiography Analysis Package.

A comprehensive package for analyzing COVID-19 radiographic images using deep learning.
"""

__version__ = "0.1.0"
__author__ = "Rafael Cepa, Cirine, Steven Moire"
__email__ = "rafael.cepa@example.fr"

# Import main modules for easy access
try:
    from . import features
    from . import models
    from . import explorationdata
    from . import streamlit
except ImportError:
    # Handle cases where some modules might not be available
    pass

__all__ = [
    "features",
    "models", 
    "explorationdata",
    "streamlit",
    "__version__",
]
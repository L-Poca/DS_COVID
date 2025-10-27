"""
Module des adaptateurs pour l'architecture Clean.
Adaptateurs wrappant le code existant pour respecter les interfaces définies.
"""

from .sklearn_pipeline_adapter import SklearnPipelineAdapter
from .tensorflow_pipeline_adapter import TensorFlowPipelineAdapter
from .covid_data_adapter import CovidDataLoaderAdapter

# Export des adaptateurs principaux
__all__ = [
    'SklearnPipelineAdapter',
    'TensorFlowPipelineAdapter',
    'CovidDataLoaderAdapter'
]


# Documentation des responsabilités des adaptateurs
ADAPTER_DOCUMENTATION = {
    'SklearnPipelineAdapter': {
        'description': 'Adaptateur pour le PipelineManager Sklearn existant',
        'wrapped_module': 'src.features.Pipelines.Pipeline_Sklearn',
        'interface_implemented': 'ISklearnPipelineManager',
        'responsibilities': [
            'Wrapper le PipelineManager existant dans l\'interface Clean',
            'Gestion des configurations JSON sklearn',
            'Adaptation des méthodes de training/evaluation',
            'Support GridSearch et validation croisée',
            'Sauvegarde/chargement avec pickle'
        ],
        'dependencies': [
            'Pipeline_Sklearn.PipelineManager',
            'sklearn (scikit-learn)',
            'pickle pour persistance'
        ]
    },
    
    'TensorFlowPipelineAdapter': {
        'description': 'Adaptateur pour le TensorFlowPipelineManager existant',
        'wrapped_module': 'src.features.Pipelines.Pipeline_TensorFlow',
        'interface_implemented': 'ITensorFlowPipelineManager',
        'responsibilities': [
            'Wrapper le TensorFlowPipelineManager dans l\'interface Clean',
            'Gestion des configurations JSON tensorflow',
            'Adaptation training avec callbacks et validation',
            'Extraction historique d\'entraînement',
            'Support transfer learning et CNN custom'
        ],
        'dependencies': [
            'Pipeline_TensorFlow.TensorFlowPipelineManager',
            'tensorflow/keras',
            'numpy pour preprocessing'
        ]
    },
    
    'CovidDataLoaderAdapter': {
        'description': 'Adaptateur pour le covid_data_loader existant',
        'wrapped_module': 'src.features.Data_Loaders.covid_data_loader',
        'interface_implemented': 'ICovidDataLoader',
        'responsibilities': [
            'Wrapper le covid_data_loader dans l\'interface Clean',
            'Chargement dataset COVID-19 Radiography',
            'Gestion cache et validation intégrité',
            'Split automatique train/val/test',
            'Preprocessing et normalisation images'
        ],
        'dependencies': [
            'covid_data_loader module',
            'PIL/OpenCV pour images',
            'sklearn pour data splitting'
        ]
    }
}


def get_adapter_info(adapter_name: str = None) -> dict:
    """
    Retourne la documentation d'un adaptateur ou de tous les adaptateurs.
    
    Args:
        adapter_name: Nom de l'adaptateur (optionnel)
        
    Returns:
        dict: Documentation du/des adaptateur(s)
    """
    if adapter_name:
        return ADAPTER_DOCUMENTATION.get(adapter_name, {})
    return ADAPTER_DOCUMENTATION


# Configuration par défaut pour les adaptateurs
DEFAULT_ADAPTER_CONFIG = {
    'caching': {
        'enable_cache': True,
        'max_cache_size': 1000,  # MB
        'cache_cleanup_threshold': 0.8
    },
    'error_handling': {
        'fallback_enabled': True,
        'retry_attempts': 3,
        'log_level': 'INFO'
    },
    'performance': {
        'lazy_loading': True,
        'batch_processing': True,
        'parallel_workers': 4
    }
}


# Factory pour créer les adaptateurs avec configuration
class AdapterFactory:
    """Factory pour instancier les adaptateurs avec configuration."""
    
    @staticmethod
    def create_sklearn_adapter(config_path: str = None) -> SklearnPipelineAdapter:
        """Crée un adaptateur Sklearn configuré."""
        try:
            return SklearnPipelineAdapter(config_path)
        except ImportError as e:
            raise ImportError(f"Sklearn adapter unavailable: {e}")
    
    @staticmethod
    def create_tensorflow_adapter(config_path: str = None) -> TensorFlowPipelineAdapter:
        """Crée un adaptateur TensorFlow configuré."""
        try:
            return TensorFlowPipelineAdapter(config_path)
        except ImportError as e:
            raise ImportError(f"TensorFlow adapter unavailable: {e}")
    
    @staticmethod
    def create_data_adapter(project_root: str = None) -> CovidDataLoaderAdapter:
        """Crée un adaptateur de données configuré."""
        try:
            return CovidDataLoaderAdapter(project_root)
        except ImportError as e:
            raise ImportError(f"Data adapter unavailable: {e}")
    
    @staticmethod
    def create_all_adapters(config_dir: str = None) -> dict:
        """
        Crée tous les adaptateurs disponibles.
        
        Args:
            config_dir: Dossier contenant les configurations
            
        Returns:
            dict: Dictionnaire des adaptateurs créés
        """
        adapters = {}
        
        try:
            adapters['sklearn'] = AdapterFactory.create_sklearn_adapter()
        except Exception as e:
            print(f"Sklearn adapter not created: {e}")
        
        try:
            adapters['tensorflow'] = AdapterFactory.create_tensorflow_adapter()
        except Exception as e:
            print(f"TensorFlow adapter not created: {e}")
        
        try:
            adapters['data'] = AdapterFactory.create_data_adapter()
        except Exception as e:
            print(f"Data adapter not created: {e}")
        
        return adapters


# Validation des adaptateurs au moment de l'import
def validate_adapters():
    """Valide la disponibilité des modules wrappés."""
    validation_results = {
        'sklearn': False,
        'tensorflow': False,
        'data_loader': False
    }
    
    # Test Sklearn
    try:
        AdapterFactory.create_sklearn_adapter()
        validation_results['sklearn'] = True
    except Exception:
        pass
    
    # Test TensorFlow
    try:
        AdapterFactory.create_tensorflow_adapter()
        validation_results['tensorflow'] = True
    except Exception:
        pass
    
    # Test Data Loader
    try:
        AdapterFactory.create_data_adapter()
        validation_results['data_loader'] = True
    except Exception:
        pass
    
    return validation_results


# Exécution de la validation à l'import (optionnel)
# adapter_availability = validate_adapters()
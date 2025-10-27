"""
Module core de l'architecture Clean pour Widget_Streamlit.
Contient les couches métier, interfaces, entités, services et adaptateurs.
"""

# Import des modules principaux
from . import entities
from . import interfaces
from . import services
from . import adapters

# Export des modules
__all__ = [
    'entities',
    'interfaces', 
    'services',
    'adapters'
]

# Documentation de l'architecture
ARCHITECTURE_INFO = {
    'pattern': 'Clean Architecture + MVVM',
    'layers': {
        'entities': 'Entités métier (modèles de données)',
        'interfaces': 'Contrats et abstractions', 
        'services': 'Logique métier pure',
        'adapters': 'Adaptation du code existant'
    },
    'dependencies': {
        'entities': [],  # Aucune dépendance
        'interfaces': ['entities'],  # Utilise les entités
        'services': ['entities', 'interfaces'],  # Utilise entités et interfaces
        'adapters': ['interfaces']  # Implémente les interfaces
    },
    'benefits': [
        'Séparation claire des responsabilités',
        'Testabilité améliorée',
        'Indépendance vis-à-vis des frameworks',
        'Facilité de maintenance et évolution'
    ]
}


def get_architecture_info() -> dict:
    """Retourne les informations sur l'architecture."""
    return ARCHITECTURE_INFO


def validate_architecture():
    """
    Valide que l'architecture respecte les principes Clean Architecture.
    
    Returns:
        dict: Résultats de validation
    """
    validation = {
        'is_valid': True,
        'violations': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Vérification de l'existence des modules
    required_modules = ['entities', 'interfaces', 'services', 'adapters']
    for module in required_modules:
        try:
            globals()[module]
        except KeyError:
            validation['is_valid'] = False
            validation['violations'].append(f"Module manquant: {module}")
    
    # Vérification des dépendances (simplifiée)
    try:
        # Les services doivent utiliser les interfaces
        if hasattr(services, 'TrainingService'):
            validation['recommendations'].append("Services correctement définis")
        
        # Les adaptateurs doivent implémenter les interfaces
        if hasattr(adapters, 'SklearnPipelineAdapter'):
            validation['recommendations'].append("Adaptateurs correctement définis")
            
    except Exception as e:
        validation['warnings'].append(f"Impossible de valider complètement: {e}")
    
    return validation


# Configuration globale de l'architecture
ARCHITECTURE_CONFIG = {
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    },
    'dependency_injection': {
        'auto_wire': True,
        'singleton_services': True
    },
    'validation': {
        'strict_interfaces': True,
        'runtime_checks': False  # Pour les performances
    }
}


# Version de l'architecture
__version__ = '1.0.0'
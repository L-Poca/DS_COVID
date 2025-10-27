"""
Module de gestion d'état pour l'architecture Clean Streamlit.
Implémentations concrètes des gestionnaires d'état, navigation, session et configuration.
"""

from .streamlit_state_manager import (
    StreamlitStateManager,
    StreamlitNavigationManager, 
    StreamlitSessionManager,
    StreamlitConfigManager
)

# Export des gestionnaires d'état
__all__ = [
    'StreamlitStateManager',
    'StreamlitNavigationManager',
    'StreamlitSessionManager', 
    'StreamlitConfigManager'
]


# Factory pour créer les gestionnaires configurés
class StateManagerFactory:
    """Factory pour créer les gestionnaires d'état avec leurs dépendances."""
    
    @staticmethod
    def create_complete_state_system(config_file: str = None) -> dict:
        """
        Crée un système d'état complet avec tous les gestionnaires.
        
        Args:
            config_file: Chemin optionnel vers le fichier de configuration
            
        Returns:
            dict: Dictionnaire contenant tous les gestionnaires
        """
        # Gestionnaire d'état principal
        state_manager = StreamlitStateManager()
        
        # Gestionnaires dépendants
        navigation_manager = StreamlitNavigationManager(state_manager)
        session_manager = StreamlitSessionManager(state_manager)
        config_manager = StreamlitConfigManager(config_file)
        
        return {
            'state': state_manager,
            'navigation': navigation_manager,
            'session': session_manager,
            'config': config_manager
        }
    
    @staticmethod
    def create_state_manager() -> StreamlitStateManager:
        """Crée un gestionnaire d'état simple."""
        return StreamlitStateManager()
    
    @staticmethod
    def create_navigation_manager(state_manager: StreamlitStateManager) -> StreamlitNavigationManager:
        """Crée un gestionnaire de navigation."""
        return StreamlitNavigationManager(state_manager)
    
    @staticmethod
    def create_session_manager(state_manager: StreamlitStateManager) -> StreamlitSessionManager:
        """Crée un gestionnaire de sessions."""
        return StreamlitSessionManager(state_manager)
    
    @staticmethod
    def create_config_manager(config_file: str = None) -> StreamlitConfigManager:
        """Crée un gestionnaire de configuration."""
        return StreamlitConfigManager(config_file)


# Configuration par défaut du système d'état
DEFAULT_STATE_CONFIG = {
    'caching': {
        'enable_persistent_cache': True,
        'cache_dir': '~/.streamlit_covid_cache',
        'max_cache_age_hours': 24,
        'cleanup_on_startup': True
    },
    'session': {
        'auto_cleanup': True,
        'session_timeout_hours': 24,
        'max_active_sessions': 100
    },
    'navigation': {
        'history_max_size': 10,
        'auto_rerun_on_navigate': True
    },
    'state_persistence': {
        'save_on_change': False,
        'auto_backup_interval_minutes': 30
    }
}


def get_default_config() -> dict:
    """Retourne la configuration par défaut du système d'état."""
    return DEFAULT_STATE_CONFIG.copy()


# Documentation des gestionnaires d'état
STATE_MANAGER_DOCUMENTATION = {
    'StreamlitStateManager': {
        'description': 'Gestionnaire d\'état centralisé avec support multi-portée',
        'scopes': ['SESSION', 'GLOBAL', 'WIDGET', 'CACHE'],
        'features': [
            'Gestion d\'état Streamlit session_state',
            'État global partagé entre sessions',
            'États locaux aux widgets',
            'Cache persistant sur disque',
            'Système de souscription aux changements'
        ],
        'use_cases': [
            'Conservation état entre interactions utilisateur',
            'Partage de données entre widgets',
            'Cache de résultats coûteux',
            'Synchronisation état interface'
        ]
    },
    
    'StreamlitNavigationManager': {
        'description': 'Gestionnaire de navigation avec historique',
        'features': [
            'Navigation entre pages avec st.rerun()',
            'Historique de navigation limité',
            'Fonction retour en arrière',
            'Passage de paramètres entre pages'
        ],
        'use_cases': [
            'Navigation fluide entre pages Streamlit',
            'Breadcrumb et navigation contextuelle',
            'Passage de données entre vues'
        ]
    },
    
    'StreamlitSessionManager': {
        'description': 'Gestionnaire de sessions utilisateur',
        'features': [
            'Création et gestion de sessions',
            'Nettoyage automatique sessions expirées',
            'Suivi de l\'activité utilisateur',
            'Association utilisateur-session'
        ],
        'use_cases': [
            'Suivi utilisateurs multiples',
            'Gestion timeout et sécurité',
            'Analytics et monitoring'
        ]
    },
    
    'StreamlitConfigManager': {
        'description': 'Gestionnaire de configuration application',
        'features': [
            'Configuration JSON hiérarchique',
            'Sauvegarde/chargement fichiers config',
            'Valeurs par défaut intégrées',
            'Accès par clés pointées'
        ],
        'use_cases': [
            'Configuration centralisée de l\'app',
            'Préférences utilisateur persistantes',
            'Paramétrage modèles et pipelines'
        ]
    }
}
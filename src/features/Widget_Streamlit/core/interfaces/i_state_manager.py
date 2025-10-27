"""
Interface pour la gestion de l'état de l'application.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable
from enum import Enum


class StateScope(Enum):
    """Portée des états dans l'application."""
    SESSION = "session"      # État de session Streamlit
    GLOBAL = "global"        # État global de l'application
    WIDGET = "widget"        # État local au widget
    CACHE = "cache"          # État en cache persistant


class IStateManager(ABC):
    """Interface pour la gestion centralisée de l'état."""
    
    @abstractmethod
    def get_state(self, key: str, scope: StateScope = StateScope.SESSION) -> Any:
        """
        Récupère une valeur d'état.
        
        Args:
            key (str): Clé de l'état
            scope (StateScope): Portée de l'état
            
        Returns:
            Any: Valeur stockée ou None
        """
        pass
    
    @abstractmethod
    def set_state(self, key: str, value: Any, scope: StateScope = StateScope.SESSION) -> None:
        """
        Définit une valeur d'état.
        
        Args:
            key (str): Clé de l'état
            value (Any): Valeur à stocker
            scope (StateScope): Portée de l'état
        """
        pass
    
    @abstractmethod
    def delete_state(self, key: str, scope: StateScope = StateScope.SESSION) -> bool:
        """
        Supprime un état.
        
        Args:
            key (str): Clé à supprimer
            scope (StateScope): Portée de l'état
            
        Returns:
            bool: True si la suppression a réussi
        """
        pass
    
    @abstractmethod
    def clear_scope(self, scope: StateScope) -> int:
        """
        Vide tous les états d'une portée.
        
        Args:
            scope (StateScope): Portée à vider
            
        Returns:
            int: Nombre d'états supprimés
        """
        pass
    
    @abstractmethod
    def get_all_keys(self, scope: StateScope) -> List[str]:
        """
        Retourne toutes les clés d'une portée.
        
        Args:
            scope (StateScope): Portée à examiner
            
        Returns:
            List[str]: Liste des clés
        """
        pass
    
    @abstractmethod
    def subscribe_to_changes(self, 
                           key: str, 
                           callback: Callable[[str, Any], None], 
                           scope: StateScope = StateScope.SESSION) -> str:
        """
        S'abonne aux changements d'un état.
        
        Args:
            key (str): Clé à surveiller
            callback: Fonction appelée lors des changements
            scope (StateScope): Portée de l'état
            
        Returns:
            str: ID de l'abonnement
        """
        pass
    
    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Se désabonne des changements.
        
        Args:
            subscription_id (str): ID de l'abonnement
            
        Returns:
            bool: True si le désabonnement a réussi
        """
        pass


class INavigationManager(ABC):
    """Interface pour la gestion de la navigation."""
    
    @abstractmethod
    def get_current_page(self) -> str:
        """
        Retourne la page actuelle.
        
        Returns:
            str: Nom de la page actuelle
        """
        pass
    
    @abstractmethod
    def navigate_to(self, page_name: str, **kwargs) -> None:
        """
        Navigue vers une page.
        
        Args:
            page_name (str): Nom de la page cible
            **kwargs: Paramètres de navigation
        """
        pass
    
    @abstractmethod
    def get_navigation_history(self) -> List[str]:
        """
        Retourne l'historique de navigation.
        
        Returns:
            List[str]: Liste des pages visitées
        """
        pass
    
    @abstractmethod
    def can_go_back(self) -> bool:
        """
        Vérifie s'il est possible de revenir en arrière.
        
        Returns:
            bool: True si retour possible
        """
        pass
    
    @abstractmethod
    def go_back(self) -> Optional[str]:
        """
        Revient à la page précédente.
        
        Returns:
            Optional[str]: Page précédente ou None
        """
        pass


class ISessionManager(ABC):
    """Interface pour la gestion des sessions utilisateur."""
    
    @abstractmethod
    def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Crée une nouvelle session.
        
        Args:
            user_id (Optional[str]): ID utilisateur optionnel
            
        Returns:
            str: ID de la session créée
        """
        pass
    
    @abstractmethod
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Récupère les informations d'une session.
        
        Args:
            session_id (str): ID de la session
            
        Returns:
            Dict[str, Any]: Informations de session
        """
        pass
    
    @abstractmethod
    def extend_session(self, session_id: str, duration_minutes: int = 30) -> bool:
        """
        Prolonge la durée d'une session.
        
        Args:
            session_id (str): ID de la session
            duration_minutes (int): Durée d'extension en minutes
            
        Returns:
            bool: True si l'extension a réussi
        """
        pass
    
    @abstractmethod
    def cleanup_expired_sessions(self) -> int:
        """
        Nettoie les sessions expirées.
        
        Returns:
            int: Nombre de sessions supprimées
        """
        pass
    
    @abstractmethod
    def get_active_sessions(self) -> List[str]:
        """
        Retourne la liste des sessions actives.
        
        Returns:
            List[str]: IDs des sessions actives
        """
        pass


class IConfigManager(ABC):
    """Interface pour la gestion de la configuration."""
    
    @abstractmethod
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur de configuration.
        
        Args:
            key (str): Clé de configuration
            default (Any): Valeur par défaut
            
        Returns:
            Any: Valeur de configuration
        """
        pass
    
    @abstractmethod
    def set_config(self, key: str, value: Any) -> None:
        """
        Définit une valeur de configuration.
        
        Args:
            key (str): Clé de configuration
            value (Any): Valeur à définir
        """
        pass
    
    @abstractmethod
    def load_config_file(self, file_path: str) -> Dict[str, Any]:
        """
        Charge un fichier de configuration.
        
        Args:
            file_path (str): Chemin du fichier
            
        Returns:
            Dict[str, Any]: Configuration chargée
        """
        pass
    
    @abstractmethod
    def save_config_file(self, file_path: str, config: Dict[str, Any]) -> bool:
        """
        Sauvegarde la configuration dans un fichier.
        
        Args:
            file_path (str): Chemin de sauvegarde
            config (Dict[str, Any]): Configuration à sauvegarder
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        pass
    
    @abstractmethod
    def reset_to_defaults(self) -> None:
        """Remet la configuration aux valeurs par défaut."""
        pass
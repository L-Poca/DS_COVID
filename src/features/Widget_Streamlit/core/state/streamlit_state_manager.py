"""
Implémentation concrète du gestionnaire d'état pour Streamlit.
Gère l'état de l'application de manière centralisée avec support des sessions.
"""

import streamlit as st
from typing import Any, Dict, List, Optional, Callable
import logging
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

from ..interfaces import IStateManager, INavigationManager, ISessionManager, IConfigManager
from ..interfaces.i_state_manager import StateScope


class StreamlitStateManager(IStateManager):
    """Gestionnaire d'état centralisé pour Streamlit."""
    
    def __init__(self):
        """Initialise le gestionnaire d'état."""
        self._logger = logging.getLogger(__name__)
        self._subscribers = {}  # Callbacks pour les changements d'état
        self._global_state = {}  # État global partagé
        self._widget_states = {}  # États locaux aux widgets
        
        # Initialisation de l'état Streamlit si nécessaire
        self._init_streamlit_state()
    
    def get_state(self, key: str, scope: StateScope = StateScope.SESSION) -> Any:
        """
        Récupère une valeur d'état.
        
        Args:
            key: Clé de l'état
            scope: Portée de l'état
            
        Returns:
            Any: Valeur stockée ou None
        """
        try:
            if scope == StateScope.SESSION:
                return st.session_state.get(key)
            elif scope == StateScope.GLOBAL:
                return self._global_state.get(key)
            elif scope == StateScope.WIDGET:
                return self._widget_states.get(key)
            elif scope == StateScope.CACHE:
                return self._get_cached_state(key)
            else:
                self._logger.warning(f"Portée d'état inconnue: {scope}")
                return None
                
        except Exception as e:
            self._logger.error(f"Erreur récupération état {key}: {e}")
            return None
    
    def set_state(self, key: str, value: Any, scope: StateScope = StateScope.SESSION) -> None:
        """
        Définit une valeur d'état.
        
        Args:
            key: Clé de l'état
            value: Valeur à stocker
            scope: Portée de l'état
        """
        try:
            old_value = self.get_state(key, scope)
            
            if scope == StateScope.SESSION:
                st.session_state[key] = value
            elif scope == StateScope.GLOBAL:
                self._global_state[key] = value
            elif scope == StateScope.WIDGET:
                self._widget_states[key] = value
            elif scope == StateScope.CACHE:
                self._set_cached_state(key, value)
            else:
                self._logger.warning(f"Portée d'état inconnue: {scope}")
                return
            
            # Notification des subscribers
            self._notify_subscribers(key, value, old_value, scope)
            
        except Exception as e:
            self._logger.error(f"Erreur définition état {key}: {e}")
    
    def delete_state(self, key: str, scope: StateScope = StateScope.SESSION) -> bool:
        """
        Supprime un état.
        
        Args:
            key: Clé à supprimer
            scope: Portée de l'état
            
        Returns:
            bool: True si la suppression a réussi
        """
        try:
            if scope == StateScope.SESSION:
                if key in st.session_state:
                    del st.session_state[key]
                    return True
            elif scope == StateScope.GLOBAL:
                if key in self._global_state:
                    del self._global_state[key]
                    return True
            elif scope == StateScope.WIDGET:
                if key in self._widget_states:
                    del self._widget_states[key]
                    return True
            elif scope == StateScope.CACHE:
                return self._delete_cached_state(key)
            
            return False
            
        except Exception as e:
            self._logger.error(f"Erreur suppression état {key}: {e}")
            return False
    
    def clear_scope(self, scope: StateScope) -> int:
        """
        Vide tous les états d'une portée.
        
        Args:
            scope: Portée à vider
            
        Returns:
            int: Nombre d'états supprimés
        """
        try:
            count = 0
            
            if scope == StateScope.SESSION:
                keys_to_delete = list(st.session_state.keys())
                for key in keys_to_delete:
                    del st.session_state[key]
                    count += 1
            elif scope == StateScope.GLOBAL:
                count = len(self._global_state)
                self._global_state.clear()
            elif scope == StateScope.WIDGET:
                count = len(self._widget_states)
                self._widget_states.clear()
            elif scope == StateScope.CACHE:
                count = self._clear_cached_states()
            
            self._logger.info(f"Portée {scope} vidée: {count} états supprimés")
            return count
            
        except Exception as e:
            self._logger.error(f"Erreur vidage portée {scope}: {e}")
            return 0
    
    def get_all_keys(self, scope: StateScope) -> List[str]:
        """
        Retourne toutes les clés d'une portée.
        
        Args:
            scope: Portée à examiner
            
        Returns:
            List[str]: Liste des clés
        """
        try:
            if scope == StateScope.SESSION:
                return list(st.session_state.keys())
            elif scope == StateScope.GLOBAL:
                return list(self._global_state.keys())
            elif scope == StateScope.WIDGET:
                return list(self._widget_states.keys())
            elif scope == StateScope.CACHE:
                return self._get_cached_keys()
            else:
                return []
                
        except Exception as e:
            self._logger.error(f"Erreur récupération clés {scope}: {e}")
            return []
    
    def subscribe_to_changes(self, 
                           key: str, 
                           callback: Callable[[str, Any], None], 
                           scope: StateScope = StateScope.SESSION) -> str:
        """
        S'abonne aux changements d'un état.
        
        Args:
            key: Clé à surveiller
            callback: Fonction appelée lors des changements
            scope: Portée de l'état
            
        Returns:
            str: ID de l'abonnement
        """
        try:
            subscription_id = f"{scope.value}_{key}_{id(callback)}"
            
            if subscription_id not in self._subscribers:
                self._subscribers[subscription_id] = {
                    'key': key,
                    'callback': callback,
                    'scope': scope,
                    'created': datetime.now()
                }
            
            self._logger.debug(f"Abonnement créé: {subscription_id}")
            return subscription_id
            
        except Exception as e:
            self._logger.error(f"Erreur création abonnement: {e}")
            return ""
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Se désabonne des changements.
        
        Args:
            subscription_id: ID de l'abonnement
            
        Returns:
            bool: True si le désabonnement a réussi
        """
        try:
            if subscription_id in self._subscribers:
                del self._subscribers[subscription_id]
                self._logger.debug(f"Désabonnement: {subscription_id}")
                return True
            return False
            
        except Exception as e:
            self._logger.error(f"Erreur désabonnement: {e}")
            return False
    
    def _init_streamlit_state(self):
        """Initialise l'état Streamlit avec les valeurs par défaut."""
        default_states = {
            'app_initialized': True,
            'current_page': 'home',
            'user_session_id': f"session_{datetime.now().timestamp()}",
            'navigation_history': []
        }
        
        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _notify_subscribers(self, key: str, new_value: Any, old_value: Any, scope: StateScope):
        """Notifie les subscribers d'un changement d'état."""
        try:
            for sub_id, sub_info in self._subscribers.items():
                if sub_info['key'] == key and sub_info['scope'] == scope:
                    try:
                        sub_info['callback'](key, new_value)
                    except Exception as e:
                        self._logger.error(f"Erreur callback subscriber {sub_id}: {e}")
                        
        except Exception as e:
            self._logger.error(f"Erreur notification subscribers: {e}")
    
    def _get_cached_state(self, key: str) -> Any:
        """Récupère un état depuis le cache persistent."""
        try:
            cache_file = self._get_cache_file_path(key)
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            self._logger.debug(f"Erreur lecture cache {key}: {e}")
            return None
    
    def _set_cached_state(self, key: str, value: Any):
        """Sauvegarde un état dans le cache persistent."""
        try:
            cache_file = self._get_cache_file_path(key)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            self._logger.error(f"Erreur écriture cache {key}: {e}")
    
    def _delete_cached_state(self, key: str) -> bool:
        """Supprime un état du cache."""
        try:
            cache_file = self._get_cache_file_path(key)
            if cache_file.exists():
                cache_file.unlink()
                return True
            return False
        except Exception as e:
            self._logger.error(f"Erreur suppression cache {key}: {e}")
            return False
    
    def _clear_cached_states(self) -> int:
        """Vide tous les états en cache."""
        try:
            cache_dir = self._get_cache_dir()
            if cache_dir.exists():
                files = list(cache_dir.glob('*.cache'))
                for file in files:
                    file.unlink()
                return len(files)
            return 0
        except Exception as e:
            self._logger.error(f"Erreur vidage cache: {e}")
            return 0
    
    def _get_cached_keys(self) -> List[str]:
        """Retourne les clés des états en cache."""
        try:
            cache_dir = self._get_cache_dir()
            if cache_dir.exists():
                files = cache_dir.glob('*.cache')
                return [f.stem for f in files]
            return []
        except Exception:
            return []
    
    def _get_cache_dir(self) -> Path:
        """Retourne le dossier de cache."""
        return Path.home() / '.streamlit_covid_cache'
    
    def _get_cache_file_path(self, key: str) -> Path:
        """Retourne le chemin du fichier de cache pour une clé."""
        return self._get_cache_dir() / f"{key}.cache"


class StreamlitNavigationManager(INavigationManager):
    """Gestionnaire de navigation pour Streamlit."""
    
    def __init__(self, state_manager: IStateManager):
        """
        Initialise le gestionnaire de navigation.
        
        Args:
            state_manager: Gestionnaire d'état à utiliser
        """
        self._state_manager = state_manager
        self._logger = logging.getLogger(__name__)
    
    def get_current_page(self) -> str:
        """Retourne la page actuelle."""
        return self._state_manager.get_state('current_page', StateScope.SESSION) or 'home'
    
    def navigate_to(self, page_name: str, **kwargs) -> None:
        """
        Navigue vers une page.
        
        Args:
            page_name: Nom de la page cible
            **kwargs: Paramètres de navigation
        """
        try:
            current_page = self.get_current_page()
            
            # Mise à jour de l'historique
            history = self.get_navigation_history()
            history.append(current_page)
            
            # Limitation de l'historique
            if len(history) > 10:
                history = history[-10:]
            
            # Mise à jour de l'état
            self._state_manager.set_state('current_page', page_name, StateScope.SESSION)
            self._state_manager.set_state('navigation_history', history, StateScope.SESSION)
            
            # Paramètres de navigation
            if kwargs:
                self._state_manager.set_state('navigation_params', kwargs, StateScope.SESSION)
            
            self._logger.info(f"Navigation: {current_page} → {page_name}")
            
            # Forcer le rerun de Streamlit
            st.rerun()
            
        except Exception as e:
            self._logger.error(f"Erreur navigation vers {page_name}: {e}")
    
    def get_navigation_history(self) -> List[str]:
        """Retourne l'historique de navigation."""
        return self._state_manager.get_state('navigation_history', StateScope.SESSION) or []
    
    def can_go_back(self) -> bool:
        """Vérifie s'il est possible de revenir en arrière."""
        history = self.get_navigation_history()
        return len(history) > 0
    
    def go_back(self) -> Optional[str]:
        """
        Revient à la page précédente.
        
        Returns:
            Optional[str]: Page précédente ou None
        """
        try:
            history = self.get_navigation_history()
            
            if history:
                previous_page = history.pop()
                self._state_manager.set_state('navigation_history', history, StateScope.SESSION)
                self.navigate_to(previous_page)
                return previous_page
            
            return None
            
        except Exception as e:
            self._logger.error(f"Erreur retour en arrière: {e}")
            return None


class StreamlitSessionManager(ISessionManager):
    """Gestionnaire de sessions pour Streamlit."""
    
    def __init__(self, state_manager: IStateManager):
        """
        Initialise le gestionnaire de sessions.
        
        Args:
            state_manager: Gestionnaire d'état à utiliser
        """
        self._state_manager = state_manager
        self._logger = logging.getLogger(__name__)
        self._sessions = {}
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Crée une nouvelle session.
        
        Args:
            user_id: ID utilisateur optionnel
            
        Returns:
            str: ID de la session créée
        """
        try:
            session_id = f"session_{datetime.now().timestamp()}"
            
            session_info = {
                'id': session_id,
                'user_id': user_id,
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'data': {}
            }
            
            self._sessions[session_id] = session_info
            
            # Mise à jour de l'état Streamlit
            self._state_manager.set_state('user_session_id', session_id, StateScope.SESSION)
            
            self._logger.info(f"Session créée: {session_id}")
            return session_id
            
        except Exception as e:
            self._logger.error(f"Erreur création session: {e}")
            return ""
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Récupère les informations d'une session.
        
        Args:
            session_id: ID de la session
            
        Returns:
            Dict[str, Any]: Informations de session
        """
        session = self._sessions.get(session_id, {})
        
        # Conversion des datetime en string pour sérialisation
        if session and 'created_at' in session:
            session_copy = session.copy()
            session_copy['created_at'] = session['created_at'].isoformat()
            session_copy['last_activity'] = session['last_activity'].isoformat()
            return session_copy
        
        return session
    
    def extend_session(self, session_id: str, duration_minutes: int = 30) -> bool:
        """
        Prolonge la durée d'une session.
        
        Args:
            session_id: ID de la session
            duration_minutes: Durée d'extension en minutes
            
        Returns:
            bool: True si l'extension a réussi
        """
        try:
            if session_id in self._sessions:
                self._sessions[session_id]['last_activity'] = datetime.now()
                self._logger.debug(f"Session prolongée: {session_id}")
                return True
            return False
            
        except Exception as e:
            self._logger.error(f"Erreur prolongation session {session_id}: {e}")
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Nettoie les sessions expirées.
        
        Returns:
            int: Nombre de sessions supprimées
        """
        try:
            now = datetime.now()
            expired_sessions = []
            
            for session_id, session_info in self._sessions.items():
                # Session expirée après 24h d'inactivité
                if now - session_info['last_activity'] > timedelta(hours=24):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self._sessions[session_id]
            
            if expired_sessions:
                self._logger.info(f"Sessions expirées supprimées: {len(expired_sessions)}")
            
            return len(expired_sessions)
            
        except Exception as e:
            self._logger.error(f"Erreur nettoyage sessions: {e}")
            return 0
    
    def get_active_sessions(self) -> List[str]:
        """
        Retourne la liste des sessions actives.
        
        Returns:
            List[str]: IDs des sessions actives
        """
        try:
            now = datetime.now()
            active_sessions = []
            
            for session_id, session_info in self._sessions.items():
                # Session active si activité dans les dernières 4h
                if now - session_info['last_activity'] < timedelta(hours=4):
                    active_sessions.append(session_id)
            
            return active_sessions
            
        except Exception as e:
            self._logger.error(f"Erreur récupération sessions actives: {e}")
            return []


class StreamlitConfigManager(IConfigManager):
    """Gestionnaire de configuration pour Streamlit."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialise le gestionnaire de configuration.
        
        Args:
            config_file: Chemin vers le fichier de configuration
        """
        self._logger = logging.getLogger(__name__)
        self._config = {}
        self._config_file = Path(config_file) if config_file else self._get_default_config_path()
        
        # Chargement de la configuration
        self._load_default_config()
        if self._config_file.exists():
            self.load_config_file(str(self._config_file))
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur de configuration.
        
        Args:
            key: Clé de configuration
            default: Valeur par défaut
            
        Returns:
            Any: Valeur de configuration
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_config(self, key: str, value: Any) -> None:
        """
        Définit une valeur de configuration.
        
        Args:
            key: Clé de configuration
            value: Valeur à définir
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def load_config_file(self, file_path: str) -> Dict[str, Any]:
        """
        Charge un fichier de configuration.
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            Dict[str, Any]: Configuration chargée
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self._config.update(config)
            self._logger.info(f"Configuration chargée: {file_path}")
            return config
            
        except Exception as e:
            self._logger.error(f"Erreur chargement config {file_path}: {e}")
            return {}
    
    def save_config_file(self, file_path: str, config: Dict[str, Any]) -> bool:
        """
        Sauvegarde la configuration dans un fichier.
        
        Args:
            file_path: Chemin de sauvegarde
            config: Configuration à sauvegarder
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self._logger.info(f"Configuration sauvegardée: {file_path}")
            return True
            
        except Exception as e:
            self._logger.error(f"Erreur sauvegarde config {file_path}: {e}")
            return False
    
    def reset_to_defaults(self) -> None:
        """Remet la configuration aux valeurs par défaut."""
        self._config.clear()
        self._load_default_config()
        self._logger.info("Configuration remise aux valeurs par défaut")
    
    def _load_default_config(self):
        """Charge la configuration par défaut."""
        self._config = {
            'app': {
                'title': 'COVID-19 Detection App',
                'theme': 'light',
                'sidebar_expanded': True
            },
            'ui': {
                'show_progress': True,
                'auto_refresh': False,
                'page_size': 20
            },
            'model': {
                'default_pipeline': 'sklearn',
                'cache_models': True,
                'max_cache_size_mb': 1000
            },
            'data': {
                'default_image_size': (224, 224),
                'normalize_images': True,
                'cache_datasets': True
            }
        }
    
    def _get_default_config_path(self) -> Path:
        """Retourne le chemin par défaut du fichier de configuration."""
        return Path.home() / '.streamlit_covid_config.json'
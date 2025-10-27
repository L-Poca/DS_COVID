"""
Container d'injection de dépendances pour l'architecture Clean.
Assemble automatiquement les composants avec leurs dépendances.
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar, Generic
from abc import ABC, abstractmethod

# Imports des interfaces et services
from ..interfaces import (
    IPipelineManager, IDataLoader, IModelValidator,
    IMetricsCalculator, IStateManager
)
from ..services import TrainingService, EvaluationService, PredictionService
from ..adapters import SklearnPipelineAdapter, TensorFlowPipelineAdapter, CovidDataLoaderAdapter
from ..state import StreamlitStateManager, StateManagerFactory

T = TypeVar('T')


class IDependencyContainer(ABC):
    """Interface pour les conteneurs d'injection de dépendances."""
    
    @abstractmethod
    def register(self, interface: Type[T], implementation: Type[T], singleton: bool = True):
        """Enregistre une implémentation pour une interface."""
        pass
    
    @abstractmethod
    def resolve(self, interface: Type[T]) -> T:
        """Résout une dépendance et retourne l'instance."""
        pass
    
    @abstractmethod
    def is_registered(self, interface: Type[T]) -> bool:
        """Vérifie si une interface est enregistrée."""
        pass


class DependencyContainer(IDependencyContainer):
    """
    Conteneur d'injection de dépendances simple.
    Gère le cycle de vie des objets et leurs dépendances.
    """
    
    def __init__(self):
        """Initialise le conteneur."""
        self._registrations: Dict[Type, Dict[str, Any]] = {}
        self._instances: Dict[Type, Any] = {}
        self._logger = logging.getLogger(__name__)
    
    def register(self, interface: Type[T], implementation: Type[T], singleton: bool = True):
        """
        Enregistre une implémentation pour une interface.
        
        Args:
            interface: Interface ou type abstrait
            implementation: Implémentation concrète
            singleton: Si True, une seule instance sera créée
        """
        self._registrations[interface] = {
            'implementation': implementation,
            'singleton': singleton
        }
        
        self._logger.debug(f"Enregistré: {interface.__name__} -> {implementation.__name__}")
    
    def register_instance(self, interface: Type[T], instance: T):
        """
        Enregistre une instance déjà créée.
        
        Args:
            interface: Interface
            instance: Instance à enregistrer
        """
        self._registrations[interface] = {
            'implementation': type(instance),
            'singleton': True
        }
        self._instances[interface] = instance
        
        self._logger.debug(f"Instance enregistrée: {interface.__name__}")
    
    def resolve(self, interface: Type[T]) -> T:
        """
        Résout une dépendance et retourne l'instance.
        
        Args:
            interface: Interface à résoudre
            
        Returns:
            T: Instance de l'implémentation
        """
        if interface not in self._registrations:
            raise ValueError(f"Interface non enregistrée: {interface.__name__}")
        
        registration = self._registrations[interface]
        
        # Si singleton et instance existe, la retourner
        if registration['singleton'] and interface in self._instances:
            return self._instances[interface]
        
        # Créer nouvelle instance
        implementation = registration['implementation']
        instance = self._create_instance(implementation)
        
        # Sauvegarder si singleton
        if registration['singleton']:
            self._instances[interface] = instance
        
        return instance
    
    def is_registered(self, interface: Type[T]) -> bool:
        """
        Vérifie si une interface est enregistrée.
        
        Args:
            interface: Interface à vérifier
            
        Returns:
            bool: True si enregistrée
        """
        return interface in self._registrations
    
    def clear(self):
        """Vide le conteneur."""
        self._registrations.clear()
        self._instances.clear()
        self._logger.info("Conteneur vidé")
    
    def _create_instance(self, implementation_class: Type[T]) -> T:
        """
        Crée une instance avec injection de dépendances automatique.
        
        Args:
            implementation_class: Classe à instancier
            
        Returns:
            T: Instance créée
        """
        try:
            # Inspection des paramètres du constructeur
            import inspect
            signature = inspect.signature(implementation_class.__init__)
            parameters = signature.parameters
            
            # Construction des arguments
            kwargs = {}
            for param_name, param in parameters.items():
                if param_name == 'self':
                    continue
                
                # Tentative de résolution automatique
                param_type = param.annotation
                if param_type != inspect.Parameter.empty and self.is_registered(param_type):
                    kwargs[param_name] = self.resolve(param_type)
            
            # Création de l'instance
            instance = implementation_class(**kwargs)
            self._logger.debug(f"Instance créée: {implementation_class.__name__}")
            return instance
            
        except Exception as e:
            self._logger.error(f"Erreur création instance {implementation_class.__name__}: {e}")
            # Fallback: création sans injection
            try:
                return implementation_class()
            except Exception as fallback_error:
                self._logger.error(f"Fallback échoué: {fallback_error}")
                raise


class CovidAppContainer:
    """
    Container spécialisé pour l'application COVID-19.
    Configure automatiquement toutes les dépendances de l'application.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise le container de l'application.
        
        Args:
            config: Configuration optionnelle
        """
        self._container = DependencyContainer()
        self._config = config or {}
        self._logger = logging.getLogger(__name__)
        
        # Configuration automatique
        self._setup_container()
    
    def _setup_container(self):
        """Configure toutes les dépendances de l'application."""
        try:
            # 1. Gestionnaires d'état
            self._register_state_managers()
            
            # 2. Adaptateurs (wrapping du code existant)
            self._register_adapters()
            
            # 3. Services métier
            self._register_services()
            
            self._logger.info("Container COVID configuré avec succès")
            
        except Exception as e:
            self._logger.error(f"Erreur configuration container: {e}")
            raise
    
    def _register_state_managers(self):
        """Enregistre les gestionnaires d'état."""
        try:
            # Gestionnaire d'état principal
            state_system = StateManagerFactory.create_complete_state_system(
                self._config.get('config_file')
            )
            
            # Enregistrement des instances
            self._container.register_instance(IStateManager, state_system['state'])
            
            self._logger.debug("Gestionnaires d'état enregistrés")
            
        except Exception as e:
            self._logger.warning(f"Gestionnaires d'état non disponibles: {e}")
    
    def _register_adapters(self):
        """Enregistre les adaptateurs du code existant."""
        adapters_registered = 0
        
        # Adaptateur Sklearn
        try:
            sklearn_config_path = self._config.get('sklearn_config_path')
            sklearn_adapter = SklearnPipelineAdapter(sklearn_config_path)
            self._container.register_instance(IPipelineManager, sklearn_adapter)
            adapters_registered += 1
            self._logger.debug("Adaptateur Sklearn enregistré")
        except Exception as e:
            self._logger.warning(f"Adaptateur Sklearn non disponible: {e}")
        
        # Adaptateur TensorFlow (enregistrement séparé si nécessaire)
        try:
            tf_config_path = self._config.get('tensorflow_config_path')
            tf_adapter = TensorFlowPipelineAdapter(tf_config_path)
            # Note: Même interface IPipelineManager, mais différenciation par nom si nécessaire
            self._logger.debug("Adaptateur TensorFlow disponible")
        except Exception as e:
            self._logger.warning(f"Adaptateur TensorFlow non disponible: {e}")
        
        # Adaptateur Data Loader
        try:
            project_root = self._config.get('project_root')
            data_adapter = CovidDataLoaderAdapter(project_root)
            self._container.register_instance(IDataLoader, data_adapter)
            adapters_registered += 1
            self._logger.debug("Adaptateur Data Loader enregistré")
        except Exception as e:
            self._logger.warning(f"Adaptateur Data Loader non disponible: {e}")
        
        if adapters_registered == 0:
            self._logger.warning("Aucun adaptateur n'a pu être enregistré")
    
    def _register_services(self):
        """Enregistre les services métier."""
        # Services avec injection automatique de dépendances
        self._container.register(TrainingService, TrainingService, singleton=True)
        self._container.register(EvaluationService, EvaluationService, singleton=True)
        self._container.register(PredictionService, PredictionService, singleton=True)
        
        self._logger.debug("Services métier enregistrés")
    
    def get_service(self, service_type: Type[T]) -> T:
        """
        Récupère un service du container.
        
        Args:
            service_type: Type du service
            
        Returns:
            T: Instance du service
        """
        return self._container.resolve(service_type)
    
    def get_training_service(self) -> TrainingService:
        """Retourne le service d'entraînement configuré."""
        return self.get_service(TrainingService)
    
    def get_evaluation_service(self) -> EvaluationService:
        """Retourne le service d'évaluation configuré."""
        return self.get_service(EvaluationService)
    
    def get_prediction_service(self) -> PredictionService:
        """Retourne le service de prédiction configuré."""
        return self.get_service(PredictionService)
    
    def get_state_manager(self) -> IStateManager:
        """Retourne le gestionnaire d'état configuré."""
        return self.get_service(IStateManager)
    
    def get_data_loader(self) -> IDataLoader:
        """Retourne le chargeur de données configuré."""
        return self.get_service(IDataLoader)
    
    def get_pipeline_manager(self) -> IPipelineManager:
        """Retourne le gestionnaire de pipelines configuré."""
        return self.get_service(IPipelineManager)
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Valide que le container est correctement configuré.
        
        Returns:
            Dict: Rapport de validation
        """
        validation = {
            'is_valid': True,
            'services_available': {},
            'errors': [],
            'warnings': []
        }
        
        # Test des services critiques
        critical_services = [
            (IStateManager, "State Manager"),
            (IDataLoader, "Data Loader"),
            (TrainingService, "Training Service"),
            (PredictionService, "Prediction Service")
        ]
        
        for service_type, service_name in critical_services:
            try:
                self.get_service(service_type)
                validation['services_available'][service_name] = True
            except Exception as e:
                validation['is_valid'] = False
                validation['services_available'][service_name] = False
                validation['errors'].append(f"{service_name}: {str(e)}")
        
        # Services optionnels
        optional_services = [
            (EvaluationService, "Evaluation Service")
        ]
        
        for service_type, service_name in optional_services:
            try:
                self.get_service(service_type)
                validation['services_available'][service_name] = True
            except Exception as e:
                validation['services_available'][service_name] = False
                validation['warnings'].append(f"{service_name}: {str(e)}")
        
        return validation


# Factory global pour l'application
class AppContainerFactory:
    """Factory global pour créer le container de l'application."""
    
    _instance: Optional[CovidAppContainer] = None
    
    @classmethod
    def create_container(cls, config: Optional[Dict[str, Any]] = None, force_new: bool = False) -> CovidAppContainer:
        """
        Crée ou retourne le container de l'application.
        
        Args:
            config: Configuration pour le container
            force_new: Force la création d'un nouveau container
            
        Returns:
            CovidAppContainer: Instance du container
        """
        if cls._instance is None or force_new:
            cls._instance = CovidAppContainer(config)
        
        return cls._instance
    
    @classmethod
    def get_container(cls) -> Optional[CovidAppContainer]:
        """Retourne le container existant ou None."""
        return cls._instance
    
    @classmethod
    def reset_container(cls):
        """Remet à zéro le container."""
        cls._instance = None


# Configuration par défaut du container
DEFAULT_CONTAINER_CONFIG = {
    'project_root': None,  # Auto-détecté
    'sklearn_config_path': None,  # Auto-détecté
    'tensorflow_config_path': None,  # Auto-détecté
    'config_file': None,  # Configuration Streamlit
    'logging_level': 'INFO',
    'enable_validation': True,
    'lazy_loading': True
}
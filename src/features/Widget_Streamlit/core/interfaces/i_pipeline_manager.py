"""
Interfaces pour les gestionnaires de pipelines.
Définit les contrats que doivent respecter les différents types de pipelines.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..entities import TrainingConfig, TrainingResult


class IPipelineManager(ABC):
    """Interface pour les gestionnaires de pipelines de machine learning."""
    
    @abstractmethod
    def get_available_configs(self) -> List[Dict[str, Any]]:
        """
        Retourne la liste des configurations disponibles.
        
        Returns:
            List[Dict]: Liste des configurations avec leurs métadonnées
        """
        pass
    
    @abstractmethod
    def create_pipeline(self, config_name: str) -> Any:
        """
        Crée un pipeline à partir d'un nom de configuration.
        
        Args:
            config_name (str): Nom de la configuration
            
        Returns:
            Any: Pipeline configuré
        """
        pass
    
    @abstractmethod
    def train_pipeline(self, pipeline: Any, X_train, y_train, **kwargs) -> Dict[str, Any]:
        """
        Entraîne un pipeline avec les données fournies.
        
        Args:
            pipeline: Pipeline à entraîner
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
            **kwargs: Paramètres additionnels
            
        Returns:
            Dict: Résultats de l'entraînement
        """
        pass
    
    @abstractmethod
    def evaluate_pipeline(self, pipeline: Any, X_test, y_test, **kwargs) -> Dict[str, Any]:
        """
        Évalue un pipeline entraîné.
        
        Args:
            pipeline: Pipeline entraîné
            X_test: Données de test
            y_test: Labels de test
            **kwargs: Paramètres additionnels
            
        Returns:
            Dict: Métriques d'évaluation
        """
        pass
    
    @abstractmethod
    def save_pipeline(self, pipeline: Any, path: str) -> bool:
        """
        Sauvegarde un pipeline entraîné.
        
        Args:
            pipeline: Pipeline à sauvegarder
            path (str): Chemin de sauvegarde
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        pass
    
    @abstractmethod
    def load_pipeline(self, path: str) -> Optional[Any]:
        """
        Charge un pipeline sauvegardé.
        
        Args:
            path (str): Chemin du pipeline à charger
            
        Returns:
            Optional[Any]: Pipeline chargé ou None si échec
        """
        pass


class ISklearnPipelineManager(IPipelineManager):
    """Interface spécialisée pour les pipelines Sklearn."""
    
    @abstractmethod
    def get_feature_importance(self, pipeline: Any) -> Optional[Dict[str, float]]:
        """
        Extrait l'importance des features du pipeline.
        
        Args:
            pipeline: Pipeline sklearn entraîné
            
        Returns:
            Optional[Dict]: Importance des features ou None
        """
        pass
    
    @abstractmethod
    def cross_validate(self, pipeline: Any, X, y, cv: int = 5) -> Dict[str, Any]:
        """
        Effectue une validation croisée.
        
        Args:
            pipeline: Pipeline à valider
            X: Données d'entrée
            y: Labels
            cv (int): Nombre de folds
            
        Returns:
            Dict: Résultats de la validation croisée
        """
        pass


class ITensorFlowPipelineManager(IPipelineManager):
    """Interface spécialisée pour les pipelines TensorFlow."""
    
    @abstractmethod
    def compile_model(self, model: Any, **compile_kwargs) -> Any:
        """
        Compile un modèle TensorFlow.
        
        Args:
            model: Modèle à compiler
            **compile_kwargs: Paramètres de compilation
            
        Returns:
            Any: Modèle compilé
        """
        pass
    
    @abstractmethod
    def get_training_history(self, history: Any) -> Dict[str, List[float]]:
        """
        Extrait l'historique d'entraînement.
        
        Args:
            history: Historique d'entraînement TensorFlow
            
        Returns:
            Dict: Métriques par époque
        """
        pass
    
    @abstractmethod
    def predict_proba(self, model: Any, X) -> Any:
        """
        Prédit les probabilités avec un modèle TensorFlow.
        
        Args:
            model: Modèle entraîné
            X: Données d'entrée
            
        Returns:
            Any: Probabilités prédites
        """
        pass


class IDataAugmentationManager(ABC):
    """Interface pour les gestionnaires d'augmentation de données."""
    
    @abstractmethod
    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """
        Retourne les stratégies d'augmentation disponibles.
        
        Returns:
            List[Dict]: Stratégies avec leurs paramètres
        """
        pass
    
    @abstractmethod
    def apply_augmentation(self, data, strategy_name: str, **kwargs) -> Any:
        """
        Applique une stratégie d'augmentation aux données.
        
        Args:
            data: Données à augmenter
            strategy_name (str): Nom de la stratégie
            **kwargs: Paramètres de la stratégie
            
        Returns:
            Any: Données augmentées
        """
        pass
    
    @abstractmethod
    def preview_augmentation(self, data, strategy_name: str, n_samples: int = 5) -> Any:
        """
        Génère un aperçu de l'augmentation.
        
        Args:
            data: Données d'exemple
            strategy_name (str): Stratégie à prévisualiser
            n_samples (int): Nombre d'échantillons à générer
            
        Returns:
            Any: Échantillons augmentés pour prévisualisation
        """
        pass
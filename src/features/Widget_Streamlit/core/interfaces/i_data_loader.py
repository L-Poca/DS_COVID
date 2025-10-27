"""
Interfaces pour les chargeurs de données.
Définit les contrats pour l'accès et la manipulation des données COVID.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from ..entities import DataConfig


class IDataLoader(ABC):
    """Interface pour les chargeurs de données."""
    
    @abstractmethod
    def load_data(self, config: DataConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Charge les données selon la configuration spécifiée.
        
        Args:
            config (DataConfig): Configuration du chargement
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (données, labels)
        """
        pass
    
    @abstractmethod
    def get_available_sizes(self) -> List[Tuple[int, int]]:
        """
        Retourne les tailles d'images disponibles.
        
        Returns:
            List[Tuple[int, int]]: Liste des tailles (width, height)
        """
        pass
    
    @abstractmethod
    def get_class_distribution(self, config: DataConfig) -> Dict[str, int]:
        """
        Analyse la distribution des classes dans le dataset.
        
        Args:
            config (DataConfig): Configuration du dataset
            
        Returns:
            Dict[str, int]: Nombre d'échantillons par classe
        """
        pass
    
    @abstractmethod
    def validate_data_integrity(self, config: DataConfig) -> Dict[str, Any]:
        """
        Valide l'intégrité des données.
        
        Args:
            config (DataConfig): Configuration à valider
            
        Returns:
            Dict[str, Any]: Rapport de validation
        """
        pass


class ICovidDataLoader(IDataLoader):
    """Interface spécialisée pour le dataset COVID-19."""
    
    @abstractmethod
    def get_sample_images(self, class_name: str, n_samples: int = 5) -> List[np.ndarray]:
        """
        Récupère des échantillons d'images pour une classe donnée.
        
        Args:
            class_name (str): Nom de la classe (COVID, Normal, Viral Pneumonia)
            n_samples (int): Nombre d'échantillons à récupérer
            
        Returns:
            List[np.ndarray]: Liste d'images échantillons
        """
        pass
    
    @abstractmethod
    def split_data(self, 
                   config: DataConfig, 
                   train_size: float = 0.7, 
                   val_size: float = 0.15, 
                   test_size: float = 0.15,
                   random_state: Optional[int] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Divise les données en ensembles d'entraînement, validation et test.
        
        Args:
            config (DataConfig): Configuration des données
            train_size (float): Proportion pour l'entraînement
            val_size (float): Proportion pour la validation
            test_size (float): Proportion pour le test
            random_state (Optional[int]): Seed pour la reproductibilité
            
        Returns:
            Dict: Ensembles divisés {'train': (X, y), 'val': (X, y), 'test': (X, y)}
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Retourne les métadonnées du dataset COVID.
        
        Returns:
            Dict[str, Any]: Informations sur le dataset
        """
        pass
    
    @abstractmethod
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Préprocesse une image individuelle.
        
        Args:
            image (np.ndarray): Image à préprocesser
            target_size (Tuple[int, int]): Taille cible
            
        Returns:
            np.ndarray: Image préprocessée
        """
        pass


class IDataValidator(ABC):
    """Interface pour la validation des données."""
    
    @abstractmethod
    def validate_image_format(self, image_path: str) -> bool:
        """
        Valide le format d'une image.
        
        Args:
            image_path (str): Chemin vers l'image
            
        Returns:
            bool: True si le format est valide
        """
        pass
    
    @abstractmethod
    def validate_dataset_structure(self, dataset_path: str) -> Dict[str, Any]:
        """
        Valide la structure d'un dataset.
        
        Args:
            dataset_path (str): Chemin vers le dataset
            
        Returns:
            Dict[str, Any]: Rapport de validation
        """
        pass
    
    @abstractmethod
    def check_data_leakage(self, 
                          train_data: np.ndarray, 
                          test_data: np.ndarray, 
                          threshold: float = 0.95) -> Dict[str, Any]:
        """
        Vérifie s'il y a une fuite de données entre les ensembles.
        
        Args:
            train_data (np.ndarray): Données d'entraînement
            test_data (np.ndarray): Données de test
            threshold (float): Seuil de similarité
            
        Returns:
            Dict[str, Any]: Résultats de la vérification
        """
        pass


class ICacheManager(ABC):
    """Interface pour la gestion du cache des données."""
    
    @abstractmethod
    def cache_exists(self, cache_key: str) -> bool:
        """
        Vérifie si un cache existe pour une clé donnée.
        
        Args:
            cache_key (str): Clé de cache
            
        Returns:
            bool: True si le cache existe
        """
        pass
    
    @abstractmethod
    def load_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Charge des données depuis le cache.
        
        Args:
            cache_key (str): Clé de cache
            
        Returns:
            Optional[Any]: Données cachées ou None
        """
        pass
    
    @abstractmethod
    def save_to_cache(self, cache_key: str, data: Any) -> bool:
        """
        Sauvegarde des données dans le cache.
        
        Args:
            cache_key (str): Clé de cache
            data (Any): Données à cacher
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        pass
    
    @abstractmethod
    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Vide le cache selon un motif optionnel.
        
        Args:
            pattern (Optional[str]): Motif pour filtrer les clés
            
        Returns:
            int: Nombre d'éléments supprimés
        """
        pass
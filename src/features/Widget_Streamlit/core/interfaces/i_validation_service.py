"""
Interfaces pour les services de validation et d'évaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from ..entities import TrainingResult, PredictionResult


class IModelValidator(ABC):
    """Interface pour la validation des modèles."""
    
    @abstractmethod
    def validate_model_performance(self, 
                                 model: Any, 
                                 X_test: np.ndarray, 
                                 y_test: np.ndarray) -> Dict[str, Any]:
        """
        Valide les performances d'un modèle.
        
        Args:
            model: Modèle à valider
            X_test: Données de test
            y_test: Labels de test
            
        Returns:
            Dict[str, Any]: Métriques de validation
        """
        pass
    
    @abstractmethod
    def cross_validate_model(self, 
                           model: Any, 
                           X: np.ndarray, 
                           y: np.ndarray, 
                           cv_folds: int = 5) -> Dict[str, Any]:
        """
        Effectue une validation croisée.
        
        Args:
            model: Modèle à valider
            X: Données complètes
            y: Labels complets
            cv_folds: Nombre de folds
            
        Returns:
            Dict[str, Any]: Résultats de validation croisée
        """
        pass
    
    @abstractmethod
    def detect_overfitting(self, 
                          training_history: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Détecte le surapprentissage à partir de l'historique.
        
        Args:
            training_history: Historique d'entraînement
            
        Returns:
            Dict[str, Any]: Analyse du surapprentissage
        """
        pass


class IMetricsCalculator(ABC):
    """Interface pour le calcul des métriques."""
    
    @abstractmethod
    def calculate_classification_metrics(self, 
                                       y_true: np.ndarray, 
                                       y_pred: np.ndarray, 
                                       y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calcule les métriques de classification.
        
        Args:
            y_true: Labels réels
            y_pred: Prédictions
            y_proba: Probabilités prédites (optionnel)
            
        Returns:
            Dict[str, float]: Métriques calculées
        """
        pass
    
    @abstractmethod
    def generate_confusion_matrix(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray, 
                                class_names: List[str]) -> Dict[str, Any]:
        """
        Génère une matrice de confusion.
        
        Args:
            y_true: Labels réels
            y_pred: Prédictions
            class_names: Noms des classes
            
        Returns:
            Dict[str, Any]: Matrice et statistiques associées
        """
        pass
    
    @abstractmethod
    def calculate_roc_metrics(self, 
                            y_true: np.ndarray, 
                            y_proba: np.ndarray, 
                            class_names: List[str]) -> Dict[str, Any]:
        """
        Calcule les métriques ROC/AUC.
        
        Args:
            y_true: Labels réels
            y_proba: Probabilités prédites
            class_names: Noms des classes
            
        Returns:
            Dict[str, Any]: Métriques ROC/AUC
        """
        pass


class IResultsExporter(ABC):
    """Interface pour l'export des résultats."""
    
    @abstractmethod
    def export_training_results(self, 
                              results: TrainingResult, 
                              export_path: str, 
                              format_type: str = 'json') -> bool:
        """
        Exporte les résultats d'entraînement.
        
        Args:
            results: Résultats à exporter
            export_path: Chemin d'export
            format_type: Format d'export (json, csv, xlsx)
            
        Returns:
            bool: True si l'export a réussi
        """
        pass
    
    @abstractmethod
    def export_predictions(self, 
                         predictions: List[PredictionResult], 
                         export_path: str, 
                         format_type: str = 'csv') -> bool:
        """
        Exporte les prédictions.
        
        Args:
            predictions: Prédictions à exporter
            export_path: Chemin d'export
            format_type: Format d'export
            
        Returns:
            bool: True si l'export a réussi
        """
        pass
    
    @abstractmethod
    def generate_report(self, 
                       training_results: List[TrainingResult], 
                       output_path: str) -> bool:
        """
        Génère un rapport complet.
        
        Args:
            training_results: Résultats d'entraînement
            output_path: Chemin de sortie
            
        Returns:
            bool: True si la génération a réussi
        """
        pass


class IVisualizationService(ABC):
    """Interface pour les services de visualisation."""
    
    @abstractmethod
    def plot_training_history(self, 
                            history: Dict[str, List[float]], 
                            save_path: Optional[str] = None) -> Any:
        """
        Trace l'historique d'entraînement.
        
        Args:
            history: Historique d'entraînement
            save_path: Chemin de sauvegarde optionnel
            
        Returns:
            Any: Figure générée
        """
        pass
    
    @abstractmethod
    def plot_confusion_matrix(self, 
                            confusion_matrix: np.ndarray, 
                            class_names: List[str], 
                            save_path: Optional[str] = None) -> Any:
        """
        Trace une matrice de confusion.
        
        Args:
            confusion_matrix: Matrice de confusion
            class_names: Noms des classes
            save_path: Chemin de sauvegarde optionnel
            
        Returns:
            Any: Figure générée
        """
        pass
    
    @abstractmethod
    def plot_roc_curves(self, 
                       roc_data: Dict[str, Any], 
                       save_path: Optional[str] = None) -> Any:
        """
        Trace les courbes ROC.
        
        Args:
            roc_data: Données ROC
            save_path: Chemin de sauvegarde optionnel
            
        Returns:
            Any: Figure générée
        """
        pass
    
    @abstractmethod
    def plot_feature_importance(self, 
                              importance_data: Dict[str, float], 
                              save_path: Optional[str] = None) -> Any:
        """
        Trace l'importance des features.
        
        Args:
            importance_data: Données d'importance
            save_path: Chemin de sauvegarde optionnel
            
        Returns:
            Any: Figure générée
        """
        pass
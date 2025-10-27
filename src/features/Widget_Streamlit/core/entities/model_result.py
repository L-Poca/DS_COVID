"""
Entités métier pour les résultats de modélisation.
Définit les structures de données pour les résultats d'entraînement, évaluation et prédiction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class ModelStatus(Enum):
    """États possibles d'un modèle."""
    PENDING = "En attente"
    TRAINING = "En cours d'entraînement"
    COMPLETED = "Terminé"
    FAILED = "Échec"


class ResultType(Enum):
    """Types de résultats."""
    TRAINING = "training"
    EVALUATION = "evaluation"
    PREDICTION = "prediction"


@dataclass
class TrainingResult:
    """Résultat d'entraînement d'un modèle."""
    model_name: str
    test_accuracy: float
    test_f1: float
    cv_mean: float
    cv_std: float
    training_time: timedelta
    pipeline: Any  # Le pipeline entraîné (sklearn, tensorflow, etc.)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Métriques détaillées optionnelles
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None
    test_classification_report: Optional[Dict] = None
    confusion_matrix: Optional[np.ndarray] = None
    cv_scores: Optional[List[float]] = None
    best_params: Optional[Dict] = None
    best_score: Optional[float] = None
    
    # Métadonnées
    status: ModelStatus = ModelStatus.COMPLETED
    config_used: Optional[Dict] = None
    data_info: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation."""
        result_dict = {
            'model_name': self.model_name,
            'test_accuracy': self.test_accuracy,
            'test_f1': self.test_f1,
            'cv_mean': self.cv_mean,
            'cv_std': self.cv_std,
            'training_time_seconds': self.training_time.total_seconds(),
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value
        }
        
        # Ajouter les métriques optionnelles si disponibles
        if self.test_precision is not None:
            result_dict['test_precision'] = self.test_precision
        if self.test_recall is not None:
            result_dict['test_recall'] = self.test_recall
        if self.test_classification_report is not None:
            result_dict['test_classification_report'] = self.test_classification_report
        if self.cv_scores is not None:
            result_dict['cv_scores'] = self.cv_scores
        if self.best_params is not None:
            result_dict['best_params'] = self.best_params
        if self.best_score is not None:
            result_dict['best_score'] = self.best_score
        if self.config_used is not None:
            result_dict['config_used'] = self.config_used
        if self.data_info is not None:
            result_dict['data_info'] = self.data_info
            
        return result_dict


@dataclass
class EvaluationResult:
    """Résultat d'évaluation d'un modèle."""
    model_name: str
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    classification_report: Dict
    confusion_matrix: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Métriques avancées optionnelles
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    log_loss: Optional[float] = None
    
    def get_summary(self) -> Dict[str, float]:
        """Retourne un résumé des métriques principales."""
        summary = {
            'accuracy': self.accuracy,
            'f1_score': self.f1_score,
            'precision': self.precision,
            'recall': self.recall
        }
        
        if self.roc_auc is not None:
            summary['roc_auc'] = self.roc_auc
        if self.pr_auc is not None:
            summary['pr_auc'] = self.pr_auc
            
        return summary


@dataclass
class PredictionResult:
    """Résultat de prédiction."""
    model_name: str
    predictions: np.ndarray
    probabilities: np.ndarray
    n_samples: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Métriques de confiance
    mean_confidence: Optional[float] = field(init=False)
    confidence_scores: Optional[List[float]] = field(init=False)
    
    def __post_init__(self):
        """Calcule les métriques de confiance après initialisation."""
        if self.probabilities is not None:
            self.confidence_scores = [max(prob) for prob in self.probabilities]
            self.mean_confidence = np.mean(self.confidence_scores) if self.confidence_scores else 0.0
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Retourne la distribution des classes prédites."""
        unique, counts = np.unique(self.predictions, return_counts=True)
        return dict(zip(unique, counts))
    
    def get_low_confidence_samples(self, threshold: float = 0.7) -> List[int]:
        """Retourne les indices des échantillons avec faible confiance."""
        if self.confidence_scores is None:
            return []
        
        return [i for i, conf in enumerate(self.confidence_scores) if conf < threshold]


@dataclass
class BatchPredictionResult:
    """Résultat de prédiction en lot."""
    results: List[PredictionResult]
    total_samples: int
    total_files: int
    overall_mean_confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques résumées du batch."""
        if not self.results:
            return {}
        
        all_predictions = []
        all_confidences = []
        
        for result in self.results:
            all_predictions.extend(result.predictions)
            if result.confidence_scores:
                all_confidences.extend(result.confidence_scores)
        
        return {
            'total_samples': self.total_samples,
            'total_files': self.total_files,
            'overall_mean_confidence': self.overall_mean_confidence,
            'min_confidence': min(all_confidences) if all_confidences else 0,
            'max_confidence': max(all_confidences) if all_confidences else 0,
            'class_distribution': dict(zip(*np.unique(all_predictions, return_counts=True))) if all_predictions else {}
        }


@dataclass
class ComparisonData:
    """Données pour la comparaison de modèles."""
    configs: List[str]
    pipeline_type: str
    data_source: str
    data_info: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


# Fonctions utilitaires

def create_training_result_from_dict(data: Dict[str, Any]) -> TrainingResult:
    """Crée un TrainingResult à partir d'un dictionnaire."""
    return TrainingResult(
        model_name=data['model_name'],
        test_accuracy=data['test_accuracy'],
        test_f1=data['test_f1'],
        cv_mean=data['cv_mean'],
        cv_std=data['cv_std'],
        training_time=timedelta(seconds=data['training_time_seconds']),
        pipeline=data.get('pipeline'),  # Peut être None lors de la désérialisation
        timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
        test_precision=data.get('test_precision'),
        test_recall=data.get('test_recall'),
        test_classification_report=data.get('test_classification_report'),
        cv_scores=data.get('cv_scores'),
        best_params=data.get('best_params'),
        best_score=data.get('best_score'),
        status=ModelStatus(data.get('status', ModelStatus.COMPLETED.value)),
        config_used=data.get('config_used'),
        data_info=data.get('data_info')
    )
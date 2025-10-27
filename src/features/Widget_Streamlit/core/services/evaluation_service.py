"""
Service d'évaluation des modèles.
Encapsule la logique métier d'évaluation et de validation.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
from datetime import datetime

from ..entities import TrainingResult, PredictionResult, DataConfig
from ..interfaces import (
    IPipelineManager, IDataLoader, IModelValidator, 
    IMetricsCalculator, IVisualizationService
)


class EvaluationService:
    """Service pour l'évaluation et la validation des modèles."""
    
    def __init__(self,
                 sklearn_manager: IPipelineManager,
                 tensorflow_manager: IPipelineManager,
                 data_loader: IDataLoader,
                 validator: IModelValidator,
                 metrics_calculator: IMetricsCalculator,
                 visualization_service: Optional[IVisualizationService] = None):
        """
        Initialise le service d'évaluation.
        
        Args:
            sklearn_manager: Gestionnaire de pipelines Sklearn
            tensorflow_manager: Gestionnaire de pipelines TensorFlow
            data_loader: Chargeur de données
            validator: Validateur de modèles
            metrics_calculator: Calculateur de métriques
            visualization_service: Service de visualisation (optionnel)
        """
        self._sklearn_manager = sklearn_manager
        self._tensorflow_manager = tensorflow_manager
        self._data_loader = data_loader
        self._validator = validator
        self._metrics_calculator = metrics_calculator
        self._visualization_service = visualization_service
        self._logger = logging.getLogger(__name__)
    
    def evaluate_model_comprehensive(self, 
                                   pipeline: Any, 
                                   training_result: TrainingResult,
                                   test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Effectue une évaluation complète d'un modèle.
        
        Args:
            pipeline: Modèle entraîné à évaluer
            training_result: Résultats d'entraînement associés
            test_data: Données de test optionnelles (X_test, y_test)
            
        Returns:
            Dict: Évaluation complète avec métriques et analyses
        """
        try:
            self._logger.info(f"Évaluation complète du modèle {training_result.model_id}")
            
            # Préparation des données de test
            if test_data is None:
                X_test, y_test = self._load_test_data(training_result.config.data_config)
            else:
                X_test, y_test = test_data
            
            # Sélection du gestionnaire approprié
            manager = self._get_pipeline_manager(training_result.config.pipeline_type)
            
            # Évaluation de base
            base_metrics = manager.evaluate_pipeline(pipeline, X_test, y_test)
            
            # Prédictions détaillées
            y_pred = pipeline.predict(X_test)
            
            # Prédictions probabilistes si possible
            y_proba = None
            try:
                if hasattr(pipeline, 'predict_proba'):
                    y_proba = pipeline.predict_proba(X_test)
                elif hasattr(manager, 'predict_proba'):
                    y_proba = manager.predict_proba(pipeline, X_test)
            except Exception as e:
                self._logger.warning(f"Impossible d'obtenir les probabilités: {e}")
            
            # Calcul des métriques détaillées
            detailed_metrics = self._metrics_calculator.calculate_classification_metrics(
                y_test, y_pred, y_proba
            )
            
            # Matrice de confusion
            class_names = self._get_class_names(training_result.config.data_config)
            confusion_data = self._metrics_calculator.generate_confusion_matrix(
                y_test, y_pred, class_names
            )
            
            # Métriques ROC/AUC si probabilités disponibles
            roc_data = None
            if y_proba is not None:
                roc_data = self._metrics_calculator.calculate_roc_metrics(
                    y_test, y_proba, class_names
                )
            
            # Analyse du surapprentissage
            overfitting_analysis = None
            if training_result.training_history:
                overfitting_analysis = self._validator.detect_overfitting(
                    training_result.training_history
                )
            
            # Validation croisée additionnelle
            cv_results = None
            if hasattr(self._validator, 'cross_validate_model'):
                try:
                    # Rechargement des données complètes pour CV
                    X_full, y_full = self._data_loader.load_data(training_result.config.data_config)
                    cv_results = self._validator.cross_validate_model(pipeline, X_full, y_full)
                except Exception as e:
                    self._logger.warning(f"Validation croisée échouée: {e}")
            
            # Assemblage des résultats
            evaluation_result = {
                'model_id': training_result.model_id,
                'evaluation_date': datetime.now().isoformat(),
                'base_metrics': base_metrics,
                'detailed_metrics': detailed_metrics,
                'confusion_matrix': confusion_data,
                'roc_analysis': roc_data,
                'overfitting_analysis': overfitting_analysis,
                'cross_validation': cv_results,
                'test_samples': len(X_test),
                'predictions_summary': {
                    'total_predictions': len(y_pred),
                    'correct_predictions': np.sum(y_pred == y_test),
                    'accuracy': np.mean(y_pred == y_test),
                    'prediction_distribution': {
                        str(class_name): int(np.sum(y_pred == i)) 
                        for i, class_name in enumerate(class_names)
                    }
                }
            }
            
            # Ajout des visualisations si service disponible
            if self._visualization_service:
                evaluation_result['visualizations'] = self._generate_evaluation_visualizations(
                    evaluation_result, training_result
                )
            
            self._logger.info("Évaluation complète terminée")
            return evaluation_result
            
        except Exception as e:
            self._logger.error(f"Erreur lors de l'évaluation complète: {e}")
            return {
                'model_id': training_result.model_id,
                'evaluation_date': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            }
    
    def compare_models(self, 
                      models_data: List[Tuple[Any, TrainingResult]],
                      comparison_metrics: List[str] = None) -> Dict[str, Any]:
        """
        Compare plusieurs modèles selon des métriques spécifiées.
        
        Args:
            models_data: Liste de tuples (pipeline, training_result)
            comparison_metrics: Métriques à comparer (défaut: accuracy, f1, precision, recall)
            
        Returns:
            Dict: Analyse comparative des modèles
        """
        if comparison_metrics is None:
            comparison_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        
        try:
            self._logger.info(f"Comparaison de {len(models_data)} modèles")
            
            comparison_results = {
                'models_count': len(models_data),
                'comparison_date': datetime.now().isoformat(),
                'metrics_compared': comparison_metrics,
                'models_performance': {},
                'ranking': {},
                'best_model': None,
                'performance_summary': {}
            }
            
            # Évaluation de chaque modèle
            for pipeline, training_result in models_data:
                model_id = training_result.model_id
                
                try:
                    # Évaluation complète
                    evaluation = self.evaluate_model_comprehensive(pipeline, training_result)
                    
                    # Extraction des métriques de comparaison
                    model_metrics = {}
                    for metric in comparison_metrics:
                        if metric in evaluation.get('detailed_metrics', {}):
                            model_metrics[metric] = evaluation['detailed_metrics'][metric]
                        elif metric in evaluation.get('base_metrics', {}):
                            model_metrics[metric] = evaluation['base_metrics'][metric]
                        else:
                            model_metrics[metric] = None
                    
                    comparison_results['models_performance'][model_id] = {
                        'metrics': model_metrics,
                        'training_time': training_result.training_time,
                        'pipeline_type': training_result.config.pipeline_type.value,
                        'pipeline_name': training_result.config.pipeline_name,
                        'evaluation_details': evaluation
                    }
                    
                except Exception as e:
                    self._logger.error(f"Erreur évaluation modèle {model_id}: {e}")
                    comparison_results['models_performance'][model_id] = {
                        'error': str(e),
                        'metrics': {metric: None for metric in comparison_metrics}
                    }
            
            # Calcul du ranking pour chaque métrique
            for metric in comparison_metrics:
                valid_scores = {}
                for model_id, perf in comparison_results['models_performance'].items():
                    score = perf.get('metrics', {}).get(metric)
                    if score is not None:
                        valid_scores[model_id] = score
                
                if valid_scores:
                    # Tri par score décroissant (meilleur = plus élevé)
                    sorted_models = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)
                    comparison_results['ranking'][metric] = [
                        {'model_id': model_id, 'score': score, 'rank': i+1}
                        for i, (model_id, score) in enumerate(sorted_models)
                    ]
            
            # Détermination du meilleur modèle (basé sur la métrique principale)
            primary_metric = comparison_metrics[0]
            if primary_metric in comparison_results['ranking']:
                best_model_info = comparison_results['ranking'][primary_metric][0]
                comparison_results['best_model'] = {
                    'model_id': best_model_info['model_id'],
                    'metric_used': primary_metric,
                    'score': best_model_info['score']
                }
            
            # Résumé statistique des performances
            for metric in comparison_metrics:
                scores = [
                    perf['metrics'].get(metric) 
                    for perf in comparison_results['models_performance'].values()
                    if perf['metrics'].get(metric) is not None
                ]
                
                if scores:
                    comparison_results['performance_summary'][metric] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores),
                        'range': np.max(scores) - np.min(scores)
                    }
            
            self._logger.info("Comparaison des modèles terminée")
            return comparison_results
            
        except Exception as e:
            self._logger.error(f"Erreur lors de la comparaison: {e}")
            return {
                'models_count': len(models_data),
                'comparison_date': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            }
    
    def generate_evaluation_report(self, 
                                 evaluation_results: Dict[str, Any],
                                 include_visualizations: bool = True) -> Dict[str, Any]:
        """
        Génère un rapport d'évaluation structuré.
        
        Args:
            evaluation_results: Résultats d'évaluation
            include_visualizations: Inclure les graphiques
            
        Returns:
            Dict: Rapport structuré pour export
        """
        try:
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'model_id': evaluation_results.get('model_id'),
                    'report_type': 'model_evaluation'
                },
                'executive_summary': self._generate_executive_summary(evaluation_results),
                'detailed_analysis': evaluation_results,
                'recommendations': self._generate_recommendations(evaluation_results)
            }
            
            if include_visualizations and self._visualization_service:
                report['visualizations'] = evaluation_results.get('visualizations', {})
            
            return report
            
        except Exception as e:
            self._logger.error(f"Erreur génération rapport: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def _load_test_data(self, data_config: DataConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Charge les données de test selon la configuration."""
        X, y = self._data_loader.load_data(data_config)
        # TODO: Implémenter la logique de division test si nécessaire
        return X, y
    
    def _get_pipeline_manager(self, pipeline_type):
        """Retourne le gestionnaire approprié selon le type."""
        if pipeline_type.value == 'sklearn':
            return self._sklearn_manager
        elif pipeline_type.value == 'tensorflow':
            return self._tensorflow_manager
        else:
            raise ValueError(f"Type de pipeline non supporté: {pipeline_type}")
    
    def _get_class_names(self, data_config: DataConfig) -> List[str]:
        """Récupère les noms des classes du dataset."""
        # TODO: Implémenter selon la structure du data_loader
        return ['COVID', 'Normal', 'Viral Pneumonia']  # Valeurs par défaut
    
    def _generate_evaluation_visualizations(self, 
                                          evaluation_result: Dict[str, Any],
                                          training_result: TrainingResult) -> Dict[str, Any]:
        """Génère les visualisations pour l'évaluation."""
        visualizations = {}
        
        try:
            # Matrice de confusion
            if 'confusion_matrix' in evaluation_result and self._visualization_service:
                cm_data = evaluation_result['confusion_matrix']
                visualizations['confusion_matrix'] = self._visualization_service.plot_confusion_matrix(
                    cm_data.get('matrix'), cm_data.get('class_names', [])
                )
            
            # Courbes ROC
            if 'roc_analysis' in evaluation_result and evaluation_result['roc_analysis']:
                visualizations['roc_curves'] = self._visualization_service.plot_roc_curves(
                    evaluation_result['roc_analysis']
                )
            
            # Historique d'entraînement
            if training_result.training_history:
                visualizations['training_history'] = self._visualization_service.plot_training_history(
                    training_result.training_history
                )
                
        except Exception as e:
            self._logger.warning(f"Erreur génération visualisations: {e}")
        
        return visualizations
    
    def _generate_executive_summary(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Génère un résumé exécutif des résultats."""
        summary = {
            'overall_performance': 'Unknown',
            'key_metrics': {},
            'strengths': [],
            'weaknesses': [],
            'confidence_level': 'Medium'
        }
        
        try:
            metrics = evaluation_results.get('detailed_metrics', {})
            
            # Performance globale
            accuracy = metrics.get('accuracy', 0)
            if accuracy >= 0.9:
                summary['overall_performance'] = 'Excellent'
            elif accuracy >= 0.8:
                summary['overall_performance'] = 'Good'
            elif accuracy >= 0.7:
                summary['overall_performance'] = 'Fair'
            else:
                summary['overall_performance'] = 'Poor'
            
            # Métriques clés
            summary['key_metrics'] = {
                'accuracy': metrics.get('accuracy'),
                'f1_score': metrics.get('f1_macro'),
                'precision': metrics.get('precision_macro'),
                'recall': metrics.get('recall_macro')
            }
            
            # Forces et faiblesses (logique simplifiée)
            if metrics.get('precision_macro', 0) > 0.85:
                summary['strengths'].append('Haute précision - peu de faux positifs')
            if metrics.get('recall_macro', 0) > 0.85:
                summary['strengths'].append('Haute sensibilité - détecte bien les cas positifs')
            
            if metrics.get('precision_macro', 1) < 0.7:
                summary['weaknesses'].append('Précision faible - beaucoup de faux positifs')
            if metrics.get('recall_macro', 1) < 0.7:
                summary['weaknesses'].append('Sensibilité faible - manque des cas positifs')
                
        except Exception as e:
            self._logger.warning(f"Erreur génération résumé: {e}")
        
        return summary
    
    def _generate_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Génère des recommandations basées sur les résultats."""
        recommendations = []
        
        try:
            metrics = evaluation_results.get('detailed_metrics', {})
            overfitting = evaluation_results.get('overfitting_analysis', {})
            
            # Recommandations basées sur les métriques
            if metrics.get('accuracy', 0) < 0.8:
                recommendations.append(
                    "Considérer l'augmentation des données ou l'amélioration des features"
                )
            
            if overfitting and overfitting.get('is_overfitting', False):
                recommendations.append(
                    "Le modèle présente du surapprentissage - considérer la régularisation"
                )
            
            # Recommandations par défaut
            if not recommendations:
                recommendations.append("Le modèle présente des performances satisfaisantes")
                
        except Exception as e:
            self._logger.warning(f"Erreur génération recommandations: {e}")
            recommendations.append("Analyse détaillée recommandée")
        
        return recommendations
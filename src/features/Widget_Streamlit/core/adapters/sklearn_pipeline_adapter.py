"""
Adaptateur pour le PipelineManager Sklearn existant.
Wrap le code existant pour respecter les interfaces Clean Architecture.
"""

from typing import Dict, Any, List, Optional
import logging
import sys
import os
from pathlib import Path

# Import robuste du PipelineManager via le gestionnaire d'imports
from .import_manager import import_manager

# Récupération sécurisée du SklearnPipelineManager
SklearnPipelineManager = import_manager.get_sklearn_pipeline_manager()

from ..interfaces import ISklearnPipelineManager


class SklearnPipelineAdapter(ISklearnPipelineManager):
    """
    Adaptateur pour le PipelineManager Sklearn existant.
    Implémente l'interface ISklearnPipelineManager en wrappant le code legacy.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise l'adaptateur avec le PipelineManager existant.
        
        Args:
            config_path: Chemin vers le fichier de configuration sklearn
        """
        self._logger = logging.getLogger(__name__)
        self._config_path = config_path
        
        if SklearnPipelineManager is None or hasattr(SklearnPipelineManager, '_is_mock'):
            self._logger.warning("⚠️ PipelineManager Sklearn non disponible - mode dégradé")
            self._pipeline_manager = SklearnPipelineManager() if SklearnPipelineManager else None
            self._is_available = False
            self._trained_pipelines = {}
        else:
            try:
                # Recherche du fichier de config si pas spécifié
                if config_path is None:
                    config_path = self._find_sklearn_config()
                
                self._pipeline_manager = SklearnPipelineManager(config_path)
                self._trained_pipelines = {}  # Cache des pipelines entraînés
                self._is_available = True
                self._logger.info(f"Adaptateur Sklearn initialisé avec config: {config_path}")
                
            except Exception as e:
                self._logger.error(f"Erreur initialisation adaptateur Sklearn: {e}")
                self._pipeline_manager = None
                self._is_available = False
                self._trained_pipelines = {}
    
    def get_available_configs(self) -> List[Dict[str, Any]]:
        """
        Retourne la liste des configurations disponibles.
        
        Returns:
            List[Dict]: Liste des configurations avec leurs métadonnées
        """
        try:
            configs = []
            
            if hasattr(self._pipeline_manager, 'config') and 'pipeline_configs' in self._pipeline_manager.config:
                pipeline_configs = self._pipeline_manager.config['pipeline_configs']
                
                for config_name, config_data in pipeline_configs.items():
                    config_info = {
                        'name': config_name,
                        'description': config_data.get('description', 'Configuration Sklearn'),
                        'type': 'sklearn',
                        'steps': config_data.get('steps', []),
                        'grid_search': config_data.get('grid_search', {}),
                        'enabled': config_data.get('enabled', True)
                    }
                    configs.append(config_info)
            
            self._logger.debug(f"Configurations disponibles: {len(configs)}")
            return configs
            
        except Exception as e:
            self._logger.error(f"Erreur récupération configurations: {e}")
            return []
    
    def create_pipeline(self, config_name: str) -> Any:
        """
        Crée un pipeline à partir d'un nom de configuration.
        
        Args:
            config_name: Nom de la configuration
            
        Returns:
            Any: Pipeline sklearn configuré
        """
        try:
            self._logger.info(f"Création pipeline Sklearn: {config_name}")
            
            # Utilisation de la méthode existante
            pipeline = self._pipeline_manager.create_pipeline(config_name)
            
            if pipeline is not None:
                self._logger.debug(f"Pipeline {config_name} créé avec succès")
            else:
                self._logger.error(f"Échec création pipeline {config_name}")
            
            return pipeline
            
        except Exception as e:
            self._logger.error(f"Erreur création pipeline {config_name}: {e}")
            return None
    
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
        try:
            self._logger.info(f"Entraînement pipeline Sklearn")
            
            # Extraction des paramètres optionnels
            X_test = kwargs.get('X_test')
            y_test = kwargs.get('y_test')
            
            # Déduction du nom de config depuis le pipeline si possible
            config_name = kwargs.get('config_name', 'custom_pipeline')
            
            # Si c'est un pipeline existant dans le manager, utiliser la méthode train_pipeline
            if config_name in self._pipeline_manager.config.get('pipeline_configs', {}):
                # Assurer que le pipeline est dans le manager
                self._pipeline_manager.pipelines[config_name] = pipeline
                
                # Entraînement via la méthode existante
                results = self._pipeline_manager.train_pipeline(
                    config_name, X_train, y_train, X_test, y_test
                )
            else:
                # Entraînement direct pour pipelines custom
                from datetime import datetime
                
                start_time = datetime.now()
                pipeline.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Construction des résultats
                results = {
                    'pipeline': pipeline,
                    'training_time': training_time,
                    'config_name': config_name
                }
                
                # Ajout des métriques si données de test disponibles
                if X_test is not None and y_test is not None:
                    y_pred = pipeline.predict(X_test)
                    from sklearn.metrics import accuracy_score, f1_score
                    
                    results.update({
                        'test_accuracy': accuracy_score(y_test, y_pred),
                        'test_f1': f1_score(y_test, y_pred, average='macro'),
                        'predictions': y_pred
                    })
                
                # GridSearch results si disponibles
                if hasattr(pipeline, 'best_params_'):
                    results.update({
                        'best_params': pipeline.best_params_,
                        'best_score': pipeline.best_score_,
                        'cv_results': pipeline.cv_results_ if hasattr(pipeline, 'cv_results_') else None
                    })
            
            # Mise en cache du pipeline entraîné
            self._trained_pipelines[config_name] = pipeline
            
            self._logger.info(f"Entraînement terminé: {config_name}")
            return results
            
        except Exception as e:
            self._logger.error(f"Erreur entraînement pipeline: {e}")
            return {'error': str(e)}
    
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
        try:
            self._logger.debug("Évaluation pipeline Sklearn")
            
            # Prédictions
            y_pred = pipeline.predict(X_test)
            
            # Calcul des métriques
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                classification_report, confusion_matrix
            )
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'predictions': y_pred.tolist(),
                'test_samples': len(X_test)
            }
            
            # Scores de probabilité si disponibles
            try:
                if hasattr(pipeline, 'predict_proba'):
                    y_proba = pipeline.predict_proba(X_test)
                    metrics['probabilities'] = y_proba.tolist()
                    
                    # ROC-AUC si multiclasse
                    from sklearn.metrics import roc_auc_score
                    try:
                        metrics['roc_auc_macro'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                    except Exception:
                        pass  # Ignore si impossible de calculer
                        
            except Exception as e:
                self._logger.debug(f"Probabilités non disponibles: {e}")
            
            self._logger.debug("Évaluation terminée")
            return metrics
            
        except Exception as e:
            self._logger.error(f"Erreur évaluation pipeline: {e}")
            return {'error': str(e)}
    
    def save_pipeline(self, pipeline: Any, path: str) -> bool:
        """
        Sauvegarde un pipeline entraîné.
        
        Args:
            pipeline: Pipeline à sauvegarder
            path: Chemin de sauvegarde
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            import pickle
            from pathlib import Path
            
            # Créer le dossier parent si nécessaire
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarde avec pickle
            with open(path, 'wb') as f:
                pickle.dump(pipeline, f)
            
            self._logger.info(f"Pipeline sauvegardé: {path}")
            return True
            
        except Exception as e:
            self._logger.error(f"Erreur sauvegarde pipeline: {e}")
            return False
    
    def load_pipeline(self, path: str) -> Optional[Any]:
        """
        Charge un pipeline sauvegardé.
        
        Args:
            path: Chemin du pipeline à charger
            
        Returns:
            Optional[Any]: Pipeline chargé ou None si échec
        """
        try:
            import pickle
            from pathlib import Path
            
            if not Path(path).exists():
                self._logger.error(f"Fichier pipeline non trouvé: {path}")
                return None
            
            with open(path, 'rb') as f:
                pipeline = pickle.load(f)
            
            self._logger.info(f"Pipeline chargé: {path}")
            return pipeline
            
        except Exception as e:
            self._logger.error(f"Erreur chargement pipeline: {e}")
            return None
    
    def get_feature_importance(self, pipeline: Any) -> Optional[Dict[str, float]]:
        """
        Extrait l'importance des features du pipeline.
        
        Args:
            pipeline: Pipeline sklearn entraîné
            
        Returns:
            Optional[Dict]: Importance des features ou None
        """
        try:
            # Recherche du composant avec feature_importances_
            if hasattr(pipeline, 'feature_importances_'):
                importances = pipeline.feature_importances_
            elif hasattr(pipeline, 'steps'):
                # Pipeline sklearn - chercher dans les étapes
                for step_name, step_obj in pipeline.steps:
                    if hasattr(step_obj, 'feature_importances_'):
                        importances = step_obj.feature_importances_
                        break
                else:
                    return None
            elif hasattr(pipeline, 'best_estimator_'):
                # GridSearchCV
                return self.get_feature_importance(pipeline.best_estimator_)
            else:
                return None
            
            # Conversion en dictionnaire
            feature_importance = {
                f'feature_{i}': float(importance) 
                for i, importance in enumerate(importances)
            }
            
            return feature_importance
            
        except Exception as e:
            self._logger.debug(f"Impossible d'extraire l'importance des features: {e}")
            return None
    
    def cross_validate(self, pipeline: Any, X, y, cv: int = 5) -> Dict[str, Any]:
        """
        Effectue une validation croisée.
        
        Args:
            pipeline: Pipeline à valider
            X: Données d'entrée
            y: Labels
            cv: Nombre de folds
            
        Returns:
            Dict: Résultats de la validation croisée
        """
        try:
            from sklearn.model_selection import cross_val_score, cross_validate
            
            # Validation croisée avec plusieurs métriques
            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            
            cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True)
            
            # Formatage des résultats
            results = {
                'cv_folds': cv,
                'metrics': {}
            }
            
            for metric in scoring:
                test_scores = cv_results[f'test_{metric}']
                train_scores = cv_results[f'train_{metric}']
                
                results['metrics'][metric] = {
                    'test_mean': float(test_scores.mean()),
                    'test_std': float(test_scores.std()),
                    'train_mean': float(train_scores.mean()),
                    'train_std': float(train_scores.std()),
                    'test_scores': test_scores.tolist(),
                    'train_scores': train_scores.tolist()
                }
            
            # Temps d'exécution
            if 'fit_time' in cv_results:
                results['fit_time'] = {
                    'mean': float(cv_results['fit_time'].mean()),
                    'std': float(cv_results['fit_time'].std()),
                    'times': cv_results['fit_time'].tolist()
                }
            
            return results
            
        except Exception as e:
            self._logger.error(f"Erreur validation croisée: {e}")
            return {'error': str(e)}
    
    def _find_sklearn_config(self) -> str:
        """
        Recherche le fichier de configuration Sklearn dans le projet.
        
        Returns:
            str: Chemin vers le fichier de configuration
        """
        # Chemins possibles pour le fichier de config
        possible_paths = [
            Path(__file__).parent.parent.parent / 'Pipelines' / 'Pipeline_Sklearn_config.json',
            Path(__file__).parent.parent.parent / 'Configs' / 'Pipeline_Sklearn_config.json',
            Path('Pipeline_Sklearn_config.json'),
            Path('configs/Pipeline_Sklearn_config.json')
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        # Fichier par défaut si aucun trouvé
        return "Pipeline_Sklearn_config.json"
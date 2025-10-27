"""
Service d'entraînement des modèles.
Encapsule toute la logique métier liée à l'entraînement.
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from ..entities import TrainingConfig, TrainingResult, PipelineType
from ..interfaces import IPipelineManager, IDataLoader, IModelValidator


class TrainingService:
    """Service pour l'entraînement des modèles de machine learning."""
    
    def __init__(self, 
                 sklearn_manager: IPipelineManager,
                 tensorflow_manager: IPipelineManager,
                 data_loader: IDataLoader,
                 validator: IModelValidator):
        """
        Initialise le service d'entraînement.
        
        Args:
            sklearn_manager: Gestionnaire de pipelines Sklearn
            tensorflow_manager: Gestionnaire de pipelines TensorFlow  
            data_loader: Chargeur de données
            validator: Validateur de modèles
        """
        self._sklearn_manager = sklearn_manager
        self._tensorflow_manager = tensorflow_manager
        self._data_loader = data_loader
        self._validator = validator
        self._logger = logging.getLogger(__name__)
    
    def get_available_configurations(self, pipeline_type: PipelineType) -> List[Dict[str, Any]]:
        """
        Récupère les configurations disponibles pour un type de pipeline.
        
        Args:
            pipeline_type: Type de pipeline souhaité
            
        Returns:
            List[Dict]: Configurations disponibles avec métadonnées
        """
        try:
            if pipeline_type == PipelineType.SKLEARN:
                return self._sklearn_manager.get_available_configs()
            elif pipeline_type == PipelineType.TENSORFLOW:
                return self._tensorflow_manager.get_available_configs()
            else:
                self._logger.error(f"Type de pipeline non supporté: {pipeline_type}")
                return []
        except Exception as e:
            self._logger.error(f"Erreur lors de la récupération des configurations: {e}")
            return []
    
    def validate_training_config(self, config: TrainingConfig) -> Dict[str, Any]:
        """
        Valide une configuration d'entraînement.
        
        Args:
            config: Configuration à valider
            
        Returns:
            Dict: Résultat de la validation avec erreurs éventuelles
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validation des données
            data_integrity = self._data_loader.validate_data_integrity(config.data_config)
            if not data_integrity.get('is_valid', True):
                validation_result['is_valid'] = False
                validation_result['errors'].extend(data_integrity.get('errors', []))
            
            # Validation de la répartition train/validation/test
            total_split = config.train_size + config.validation_size + config.test_size
            if abs(total_split - 1.0) > 0.01:
                validation_result['is_valid'] = False
                validation_result['errors'].append(
                    f"La somme des répartitions doit égaler 1.0 (actuel: {total_split})"
                )
            
            # Vérifications spécifiques au type de pipeline
            if config.pipeline_type == PipelineType.TENSORFLOW:
                if config.epochs <= 0:
                    validation_result['is_valid'] = False
                    validation_result['errors'].append("Le nombre d'époques doit être positif")
                
                if config.batch_size <= 0:
                    validation_result['is_valid'] = False
                    validation_result['errors'].append("La taille de batch doit être positive")
            
            # Avertissements
            if config.train_size < 0.6:
                validation_result['warnings'].append(
                    "Proportion d'entraînement faible (<60%), performances potentiellement limitées"
                )
            
            if config.test_size < 0.15:
                validation_result['warnings'].append(
                    "Proportion de test faible (<15%), évaluation potentiellement biaisée"
                )
        
        except Exception as e:
            self._logger.error(f"Erreur lors de la validation: {e}")
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Erreur de validation: {str(e)}")
        
        return validation_result
    
    def prepare_training_data(self, config: TrainingConfig) -> Dict[str, Any]:
        """
        Prépare les données pour l'entraînement.
        
        Args:
            config: Configuration d'entraînement
            
        Returns:
            Dict: Données préparées et divisées
        """
        try:
            self._logger.info("Chargement et préparation des données...")
            
            # Chargement des données
            X, y = self._data_loader.load_data(config.data_config)
            
            # Division des données
            if hasattr(self._data_loader, 'split_data'):
                splits = self._data_loader.split_data(
                    config.data_config,
                    train_size=config.train_size,
                    val_size=config.validation_size,
                    test_size=config.test_size,
                    random_state=config.random_state
                )
            else:
                # Division manuelle si pas implémentée dans le loader
                from sklearn.model_selection import train_test_split
                
                # Division initiale train+val / test
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=config.test_size, 
                    random_state=config.random_state, stratify=y
                )
                
                # Division train / val
                val_ratio = config.validation_size / (config.train_size + config.validation_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_ratio,
                    random_state=config.random_state, stratify=y_temp
                )
                
                splits = {
                    'train': (X_train, y_train),
                    'val': (X_val, y_val),
                    'test': (X_test, y_test)
                }
            
            # Ajout des métadonnées
            splits['metadata'] = {
                'total_samples': len(X),
                'train_samples': len(splits['train'][0]),
                'val_samples': len(splits['val'][0]),
                'test_samples': len(splits['test'][0]),
                'feature_shape': X.shape[1:] if len(X.shape) > 1 else (X.shape[0],),
                'class_distribution': self._data_loader.get_class_distribution(config.data_config)
            }
            
            self._logger.info(f"Données préparées: {splits['metadata']}")
            return splits
            
        except Exception as e:
            self._logger.error(f"Erreur lors de la préparation des données: {e}")
            raise
    
    def train_model(self, config: TrainingConfig, prepared_data: Dict[str, Any]) -> TrainingResult:
        """
        Entraîne un modèle selon la configuration fournie.
        
        Args:
            config: Configuration d'entraînement
            prepared_data: Données préparées par prepare_training_data
            
        Returns:
            TrainingResult: Résultats de l'entraînement
        """
        start_time = datetime.now()
        
        try:
            self._logger.info(f"Début de l'entraînement avec {config.pipeline_name}")
            
            # Sélection du gestionnaire de pipeline
            if config.pipeline_type == PipelineType.SKLEARN:
                manager = self._sklearn_manager
            elif config.pipeline_type == PipelineType.TENSORFLOW:
                manager = self._tensorflow_manager
            else:
                raise ValueError(f"Type de pipeline non supporté: {config.pipeline_type}")
            
            # Création du pipeline
            pipeline = manager.create_pipeline(config.pipeline_name)
            
            # Extraction des données d'entraînement
            X_train, y_train = prepared_data['train']
            X_val, y_val = prepared_data['val']
            X_test, y_test = prepared_data['test']
            
            # Entraînement
            training_kwargs = config.training_params or {}
            if config.pipeline_type == PipelineType.TENSORFLOW:
                training_kwargs.update({
                    'epochs': config.epochs,
                    'batch_size': config.batch_size,
                    'validation_data': (X_val, y_val)
                })
            
            training_results = manager.train_pipeline(
                pipeline, X_train, y_train, **training_kwargs
            )
            
            # Évaluation
            evaluation_results = manager.evaluate_pipeline(pipeline, X_test, y_test)
            
            # Validation croisée si demandée
            cv_results = None
            if config.cross_validation:
                cv_results = self._validator.cross_validate_model(
                    pipeline, X_train, y_train, cv_folds=config.cv_folds or 5
                )
            
            # Construction du résultat
            end_time = datetime.now()
            training_duration = (end_time - start_time).total_seconds()
            
            result = TrainingResult(
                config=config,
                model_id=f"{config.pipeline_name}_{int(start_time.timestamp())}",
                training_time=training_duration,
                metrics=evaluation_results,
                cross_validation_results=cv_results,
                model_path=None,  # Sera défini lors de la sauvegarde
                training_history=training_results.get('history'),
                metadata={
                    'data_metadata': prepared_data['metadata'],
                    'training_params': training_kwargs,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat()
                }
            )
            
            self._logger.info(f"Entraînement terminé en {training_duration:.2f}s")
            return result
            
        except Exception as e:
            self._logger.error(f"Erreur lors de l'entraînement: {e}")
            # Retour d'un résultat d'échec
            return TrainingResult(
                config=config,
                model_id=f"failed_{int(start_time.timestamp())}",
                training_time=(datetime.now() - start_time).total_seconds(),
                metrics={'error': str(e)},
                cross_validation_results=None,
                model_path=None,
                training_history=None,
                metadata={'error': str(e), 'success': False}
            )
    
    def save_trained_model(self, 
                          pipeline: Any, 
                          result: TrainingResult, 
                          save_path: str) -> bool:
        """
        Sauvegarde un modèle entraîné.
        
        Args:
            pipeline: Pipeline entraîné à sauvegarder
            result: Résultats d'entraînement associés
            save_path: Chemin de sauvegarde
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            # Sélection du gestionnaire approprié
            if result.config.pipeline_type == PipelineType.SKLEARN:
                manager = self._sklearn_manager
            elif result.config.pipeline_type == PipelineType.TENSORFLOW:
                manager = self._tensorflow_manager
            else:
                self._logger.error(f"Type de pipeline non supporté: {result.config.pipeline_type}")
                return False
            
            # Sauvegarde du modèle
            success = manager.save_pipeline(pipeline, save_path)
            
            if success:
                # Mise à jour du chemin dans les résultats
                result.model_path = save_path
                self._logger.info(f"Modèle sauvegardé: {save_path}")
            else:
                self._logger.error(f"Échec de la sauvegarde: {save_path}")
            
            return success
            
        except Exception as e:
            self._logger.error(f"Erreur lors de la sauvegarde: {e}")
            return False
    
    def load_trained_model(self, model_path: str, pipeline_type: PipelineType) -> Optional[Any]:
        """
        Charge un modèle sauvegardé.
        
        Args:
            model_path: Chemin du modèle à charger
            pipeline_type: Type de pipeline du modèle
            
        Returns:
            Optional[Any]: Pipeline chargé ou None si échec
        """
        try:
            if pipeline_type == PipelineType.SKLEARN:
                manager = self._sklearn_manager
            elif pipeline_type == PipelineType.TENSORFLOW:
                manager = self._tensorflow_manager
            else:
                self._logger.error(f"Type de pipeline non supporté: {pipeline_type}")
                return None
            
            pipeline = manager.load_pipeline(model_path)
            
            if pipeline is not None:
                self._logger.info(f"Modèle chargé: {model_path}")
            else:
                self._logger.error(f"Échec du chargement: {model_path}")
            
            return pipeline
            
        except Exception as e:
            self._logger.error(f"Erreur lors du chargement: {e}")
            return None
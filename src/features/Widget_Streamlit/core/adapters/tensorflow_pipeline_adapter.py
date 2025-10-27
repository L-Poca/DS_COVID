"""
Adaptateur pour le PipelineManager TensorFlow existant.
Wrap le code existant pour respecter les interfaces Clean Architecture.
"""

from typing import Dict, Any, List, Optional
import logging
import sys
import os
from pathlib import Path
import numpy as np

# Import robuste du TensorFlowPipelineManager via le gestionnaire d'imports
from .import_manager import import_manager

# Récupération sécurisée du TensorFlowPipelineManager
TensorFlowPipelineManager = import_manager.get_tensorflow_pipeline_manager()

from ..interfaces import ITensorFlowPipelineManager


class TensorFlowPipelineAdapter(ITensorFlowPipelineManager):
    """
    Adaptateur pour le TensorFlowPipelineManager existant.
    Implémente l'interface ITensorFlowPipelineManager en wrappant le code legacy.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise l'adaptateur avec le TensorFlowPipelineManager existant.
        
        Args:
            config_path: Chemin vers le fichier de configuration TensorFlow
        """
        self._logger = logging.getLogger(__name__)
        
        if TensorFlowPipelineManager is None:
            raise ImportError("TensorFlowPipelineManager non disponible")
        
        try:
            # Recherche du fichier de config si pas spécifié
            if config_path is None:
                config_path = self._find_tensorflow_config()
            
            self._pipeline_manager = TensorFlowPipelineManager(config_path)
            self._trained_models = {}  # Cache des modèles entraînés
            self._training_histories = {}  # Cache des historiques
            self._logger.info(f"Adaptateur TensorFlow initialisé avec config: {config_path}")
            
        except Exception as e:
            self._logger.error(f"Erreur initialisation adaptateur TensorFlow: {e}")
            raise
    
    def get_available_configs(self) -> List[Dict[str, Any]]:
        """
        Retourne la liste des configurations disponibles.
        
        Returns:
            List[Dict]: Liste des configurations avec leurs métadonnées
        """
        try:
            configs = []
            
            if hasattr(self._pipeline_manager, 'config') and 'model_configs' in self._pipeline_manager.config:
                model_configs = self._pipeline_manager.config['model_configs']
                
                for config_name, config_data in model_configs.items():
                    config_info = {
                        'name': config_name,
                        'description': config_data.get('description', 'Configuration TensorFlow'),
                        'type': 'tensorflow',
                        'architecture': config_data.get('architecture', 'custom'),
                        'input_shape': config_data.get('input_shape', [224, 224, 3]),
                        'num_classes': config_data.get('num_classes', 3),
                        'enabled': config_data.get('enabled', True),
                        'transfer_learning': config_data.get('base_model') is not None
                    }
                    configs.append(config_info)
            
            self._logger.debug(f"Configurations TensorFlow disponibles: {len(configs)}")
            return configs
            
        except Exception as e:
            self._logger.error(f"Erreur récupération configurations TensorFlow: {e}")
            return []
    
    def create_pipeline(self, config_name: str) -> Any:
        """
        Crée un modèle TensorFlow à partir d'un nom de configuration.
        
        Args:
            config_name: Nom de la configuration
            
        Returns:
            Any: Modèle TensorFlow/Keras configuré
        """
        try:
            self._logger.info(f"Création modèle TensorFlow: {config_name}")
            
            # Utilisation de la méthode existante
            model = self._pipeline_manager.create_model(config_name)
            
            if model is not None:
                self._logger.debug(f"Modèle {config_name} créé avec succès")
                # Compiler le modèle si pas déjà fait
                if not model.built:
                    self._logger.debug("Compilation du modèle...")
                    self.compile_model(model)
            else:
                self._logger.error(f"Échec création modèle {config_name}")
            
            return model
            
        except Exception as e:
            self._logger.error(f"Erreur création modèle {config_name}: {e}")
            return None
    
    def compile_model(self, model: Any, **compile_kwargs) -> Any:
        """
        Compile un modèle TensorFlow.
        
        Args:
            model: Modèle à compiler
            **compile_kwargs: Paramètres de compilation
            
        Returns:
            Any: Modèle compilé
        """
        try:
            # Paramètres par défaut pour la compilation
            default_compile_params = {
                'optimizer': 'adam',
                'loss': 'categorical_crossentropy',
                'metrics': ['accuracy']
            }
            
            # Fusion avec les paramètres fournis
            compile_params = {**default_compile_params, **compile_kwargs}
            
            # Compilation du modèle
            model.compile(**compile_params)
            
            self._logger.debug(f"Modèle compilé avec: {compile_params}")
            return model
            
        except Exception as e:
            self._logger.error(f"Erreur compilation modèle: {e}")
            return model  # Retourner le modèle même en cas d'erreur
    
    def train_pipeline(self, pipeline: Any, X_train, y_train, **kwargs) -> Dict[str, Any]:
        """
        Entraîne un modèle TensorFlow avec les données fournies.
        
        Args:
            pipeline: Modèle à entraîner
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
            **kwargs: Paramètres additionnels
            
        Returns:
            Dict: Résultats de l'entraînement avec historique
        """
        try:
            self._logger.info("Entraînement modèle TensorFlow")
            
            # Extraction des paramètres
            epochs = kwargs.get('epochs', 10)
            batch_size = kwargs.get('batch_size', 32)
            validation_data = kwargs.get('validation_data')
            config_name = kwargs.get('config_name', 'custom_model')
            
            # Préparation des données
            X_train_processed = self._preprocess_data(X_train)
            y_train_processed = self._preprocess_labels(y_train)
            
            validation_data_processed = None
            if validation_data is not None:
                X_val, y_val = validation_data
                validation_data_processed = (
                    self._preprocess_data(X_val),
                    self._preprocess_labels(y_val)
                )
            
            # Callbacks par défaut
            callbacks_list = kwargs.get('callbacks', self._get_default_callbacks())
            
            # Entraînement
            from datetime import datetime
            start_time = datetime.now()
            
            history = pipeline.fit(
                X_train_processed, y_train_processed,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data_processed,
                callbacks=callbacks_list,
                verbose=kwargs.get('verbose', 1)
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Sauvegarde de l'historique
            self._training_histories[config_name] = history
            
            # Construction des résultats
            results = {
                'model': pipeline,
                'history': history.history,
                'training_time': training_time,
                'config_name': config_name,
                'epochs_completed': len(history.history['loss']),
                'best_epoch': np.argmax(history.history.get('val_accuracy', history.history['accuracy'])) + 1
            }
            
            # Ajout des métriques finales
            if 'val_accuracy' in history.history:
                results['final_val_accuracy'] = history.history['val_accuracy'][-1]
                results['best_val_accuracy'] = max(history.history['val_accuracy'])
            
            results['final_train_accuracy'] = history.history['accuracy'][-1]
            results['final_loss'] = history.history['loss'][-1]
            
            # Mise en cache du modèle entraîné
            self._trained_models[config_name] = pipeline
            
            self._logger.info(f"Entraînement terminé: {config_name} ({epochs} époques)")
            return results
            
        except Exception as e:
            self._logger.error(f"Erreur entraînement modèle TensorFlow: {e}")
            return {'error': str(e)}
    
    def evaluate_pipeline(self, pipeline: Any, X_test, y_test, **kwargs) -> Dict[str, Any]:
        """
        Évalue un modèle TensorFlow entraîné.
        
        Args:
            pipeline: Modèle entraîné
            X_test: Données de test
            y_test: Labels de test
            **kwargs: Paramètres additionnels
            
        Returns:
            Dict: Métriques d'évaluation
        """
        try:
            self._logger.debug("Évaluation modèle TensorFlow")
            
            # Préparation des données
            X_test_processed = self._preprocess_data(X_test)
            y_test_processed = self._preprocess_labels(y_test)
            
            # Évaluation avec le modèle
            test_loss, test_accuracy = pipeline.evaluate(
                X_test_processed, y_test_processed, 
                verbose=kwargs.get('verbose', 0)
            )
            
            # Prédictions
            y_pred_proba = pipeline.predict(X_test_processed)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = np.argmax(y_test_processed, axis=1) if len(y_test_processed.shape) > 1 else y_test_processed
            
            # Métriques détaillées
            from sklearn.metrics import (
                precision_score, recall_score, f1_score,
                classification_report, confusion_matrix
            )
            
            metrics = {
                'test_loss': float(test_loss),
                'test_accuracy': float(test_accuracy),
                'accuracy': float(test_accuracy),  # Alias pour compatibilité
                'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
                'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
                'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
                'classification_report': classification_report(y_true, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist(),
                'test_samples': len(X_test)
            }
            
            # ROC-AUC si multiclasse
            try:
                from sklearn.metrics import roc_auc_score
                metrics['roc_auc_macro'] = float(roc_auc_score(
                    y_test_processed if len(y_test_processed.shape) > 1 else 
                    self._to_categorical(y_test_processed, y_pred_proba.shape[1]),
                    y_pred_proba, 
                    multi_class='ovr', 
                    average='macro'
                ))
            except Exception as e:
                self._logger.debug(f"ROC-AUC non calculable: {e}")
            
            self._logger.debug("Évaluation TensorFlow terminée")
            return metrics
            
        except Exception as e:
            self._logger.error(f"Erreur évaluation modèle TensorFlow: {e}")
            return {'error': str(e)}
    
    def save_pipeline(self, pipeline: Any, path: str) -> bool:
        """
        Sauvegarde un modèle TensorFlow entraîné.
        
        Args:
            pipeline: Modèle à sauvegarder
            path: Chemin de sauvegarde
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            from pathlib import Path
            
            # Créer le dossier parent si nécessaire
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarde du modèle TensorFlow
            pipeline.save(path)
            
            self._logger.info(f"Modèle TensorFlow sauvegardé: {path}")
            return True
            
        except Exception as e:
            self._logger.error(f"Erreur sauvegarde modèle TensorFlow: {e}")
            return False
    
    def load_pipeline(self, path: str) -> Optional[Any]:
        """
        Charge un modèle TensorFlow sauvegardé.
        
        Args:
            path: Chemin du modèle à charger
            
        Returns:
            Optional[Any]: Modèle chargé ou None si échec
        """
        try:
            import tensorflow as tf
            from pathlib import Path
            
            if not Path(path).exists():
                self._logger.error(f"Fichier modèle non trouvé: {path}")
                return None
            
            model = tf.keras.models.load_model(path)
            
            self._logger.info(f"Modèle TensorFlow chargé: {path}")
            return model
            
        except Exception as e:
            self._logger.error(f"Erreur chargement modèle TensorFlow: {e}")
            return None
    
    def get_training_history(self, history: Any) -> Dict[str, List[float]]:
        """
        Extrait l'historique d'entraînement.
        
        Args:
            history: Historique d'entraînement TensorFlow
            
        Returns:
            Dict: Métriques par époque
        """
        try:
            if hasattr(history, 'history'):
                return history.history
            elif isinstance(history, dict):
                return history
            else:
                self._logger.warning("Format d'historique non reconnu")
                return {}
                
        except Exception as e:
            self._logger.error(f"Erreur extraction historique: {e}")
            return {}
    
    def predict_proba(self, model: Any, X) -> Any:
        """
        Prédit les probabilités avec un modèle TensorFlow.
        
        Args:
            model: Modèle entraîné
            X: Données d'entrée
            
        Returns:
            Any: Probabilités prédites
        """
        try:
            X_processed = self._preprocess_data(X)
            predictions = model.predict(X_processed)
            return predictions
            
        except Exception as e:
            self._logger.error(f"Erreur prédiction probabilités: {e}")
            return None
    
    def _preprocess_data(self, X) -> np.ndarray:
        """Préprocesse les données d'entrée."""
        try:
            X_processed = np.array(X)
            
            # Normalisation si nécessaire
            if X_processed.dtype != np.float32:
                X_processed = X_processed.astype(np.float32)
            
            if X_processed.max() > 1.0:
                X_processed = X_processed / 255.0
            
            # Assurer la forme correcte pour CNN
            if len(X_processed.shape) == 3:
                X_processed = np.expand_dims(X_processed, axis=0)
            
            return X_processed
            
        except Exception as e:
            self._logger.warning(f"Erreur preprocessing données: {e}")
            return X
    
    def _preprocess_labels(self, y) -> np.ndarray:
        """Préprocesse les labels."""
        try:
            y_processed = np.array(y)
            
            # Conversion en one-hot si nécessaire
            if len(y_processed.shape) == 1:
                from tensorflow.keras.utils import to_categorical
                num_classes = len(np.unique(y_processed))
                y_processed = to_categorical(y_processed, num_classes)
            
            return y_processed
            
        except Exception as e:
            self._logger.warning(f"Erreur preprocessing labels: {e}")
            return y
    
    def _to_categorical(self, y, num_classes):
        """Conversion en one-hot encoding."""
        try:
            from tensorflow.keras.utils import to_categorical
            return to_categorical(y, num_classes)
        except Exception:
            return y
    
    def _get_default_callbacks(self) -> list:
        """Retourne les callbacks par défaut pour l'entraînement."""
        try:
            import tensorflow as tf
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-7
                )
            ]
            
            return callbacks
            
        except Exception as e:
            self._logger.warning(f"Callbacks par défaut non disponibles: {e}")
            return []
    
    def _find_tensorflow_config(self) -> str:
        """
        Recherche le fichier de configuration TensorFlow dans le projet.
        
        Returns:
            str: Chemin vers le fichier de configuration
        """
        # Chemins possibles pour le fichier de config
        possible_paths = [
            Path(__file__).parent.parent.parent / 'Pipelines' / 'Pipeline_TensorFlow_config.json',
            Path(__file__).parent.parent.parent / 'Configs' / 'Pipeline_TensorFlow_config.json',
            Path('Pipeline_TensorFlow_config.json'),
            Path('configs/Pipeline_TensorFlow_config.json')
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        # Fichier par défaut si aucun trouvé
        return "Pipeline_TensorFlow_config.json"
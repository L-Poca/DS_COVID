"""
Service de prédiction.
Encapsule la logique métier de prédiction et d'inférence.
"""

from typing import Dict, Any, List, Optional, Union
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

from ..entities import PredictionResult, PipelineType, DataSource
from ..interfaces import IPipelineManager, IDataLoader


class PredictionService:
    """Service pour la prédiction avec des modèles entraînés."""
    
    def __init__(self,
                 sklearn_manager: IPipelineManager,
                 tensorflow_manager: IPipelineManager,
                 data_loader: IDataLoader):
        """
        Initialise le service de prédiction.
        
        Args:
            sklearn_manager: Gestionnaire de pipelines Sklearn
            tensorflow_manager: Gestionnaire de pipelines TensorFlow
            data_loader: Chargeur de données
        """
        self._sklearn_manager = sklearn_manager
        self._tensorflow_manager = tensorflow_manager
        self._data_loader = data_loader
        self._logger = logging.getLogger(__name__)
        
        # Cache des modèles chargés pour éviter les rechargements
        self._loaded_models = {}
    
    def load_model_for_prediction(self, 
                                model_path: str, 
                                pipeline_type: PipelineType,
                                force_reload: bool = False) -> Optional[Any]:
        """
        Charge un modèle pour la prédiction avec mise en cache.
        
        Args:
            model_path: Chemin du modèle à charger
            pipeline_type: Type de pipeline du modèle
            force_reload: Force le rechargement même si en cache
            
        Returns:
            Optional[Any]: Modèle chargé ou None si échec
        """
        try:
            cache_key = f"{model_path}_{pipeline_type.value}"
            
            # Vérification du cache
            if not force_reload and cache_key in self._loaded_models:
                self._logger.debug(f"Modèle trouvé en cache: {model_path}")
                return self._loaded_models[cache_key]
            
            # Chargement du modèle
            if pipeline_type == PipelineType.SKLEARN:
                manager = self._sklearn_manager
            elif pipeline_type == PipelineType.TENSORFLOW:
                manager = self._tensorflow_manager
            else:
                self._logger.error(f"Type de pipeline non supporté: {pipeline_type}")
                return None
            
            model = manager.load_pipeline(model_path)
            
            if model is not None:
                # Mise en cache
                self._loaded_models[cache_key] = model
                self._logger.info(f"Modèle chargé et mis en cache: {model_path}")
            else:
                self._logger.error(f"Échec du chargement: {model_path}")
            
            return model
            
        except Exception as e:
            self._logger.error(f"Erreur lors du chargement du modèle: {e}")
            return None
    
    def predict_single_image(self, 
                           model: Any, 
                           image_data: np.ndarray,
                           model_info: Dict[str, Any] = None) -> PredictionResult:
        """
        Effectue une prédiction sur une seule image.
        
        Args:
            model: Modèle chargé pour la prédiction
            image_data: Données de l'image à analyser
            model_info: Informations sur le modèle (optionnel)
            
        Returns:
            PredictionResult: Résultat de la prédiction
        """
        try:
            prediction_start = datetime.now()
            
            # Préparation de l'image si nécessaire
            processed_image = self._preprocess_single_image(image_data)
            
            # Prédiction de classe
            predicted_class_idx = model.predict(processed_image.reshape(1, -1))[0]
            
            # Prédiction de probabilités si possible
            confidence_scores = None
            try:
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(processed_image.reshape(1, -1))[0]
                    confidence_scores = {
                        f'class_{i}': float(prob) for i, prob in enumerate(probabilities)
                    }
                elif hasattr(model, 'decision_function'):
                    decision_scores = model.decision_function(processed_image.reshape(1, -1))[0]
                    # Conversion en probabilités approximatives avec softmax
                    exp_scores = np.exp(decision_scores - np.max(decision_scores))
                    probabilities = exp_scores / np.sum(exp_scores)
                    confidence_scores = {
                        f'class_{i}': float(prob) for i, prob in enumerate(probabilities)
                    }
            except Exception as e:
                self._logger.warning(f"Impossible d'obtenir les scores de confiance: {e}")
            
            # Mappage vers les noms de classes COVID
            class_names = ['COVID', 'Normal', 'Viral Pneumonia']
            predicted_class = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else f'Unknown_{predicted_class_idx}'
            
            # Calcul de la confiance finale
            confidence = None
            if confidence_scores:
                confidence = max(confidence_scores.values())
                # Mise à jour avec les vrais noms de classes
                confidence_scores = {
                    class_names[i]: score for i, score in enumerate(confidence_scores.values())
                    if i < len(class_names)
                }
            
            prediction_time = (datetime.now() - prediction_start).total_seconds()
            
            # Construction du résultat
            result = PredictionResult(
                predicted_class=predicted_class,
                confidence=confidence,
                confidence_scores=confidence_scores,
                prediction_time=prediction_time,
                input_shape=image_data.shape,
                model_info=model_info or {},
                metadata={
                    'prediction_date': prediction_start.isoformat(),
                    'image_preprocessing': 'applied',
                    'available_classes': class_names
                }
            )
            
            self._logger.debug(f"Prédiction terminée: {predicted_class} (confiance: {confidence:.3f if confidence else 'N/A'})")
            return result
            
        except Exception as e:
            self._logger.error(f"Erreur lors de la prédiction: {e}")
            return PredictionResult(
                predicted_class="ERROR",
                confidence=0.0,
                confidence_scores=None,
                prediction_time=0.0,
                input_shape=image_data.shape if isinstance(image_data, np.ndarray) else None,
                model_info=model_info or {},
                metadata={'error': str(e)}
            )
    
    def predict_batch(self, 
                     model: Any,
                     images_data: List[np.ndarray],
                     model_info: Dict[str, Any] = None,
                     batch_size: int = 32) -> List[PredictionResult]:
        """
        Effectue des prédictions sur un lot d'images.
        
        Args:
            model: Modèle chargé pour la prédiction
            images_data: Liste des images à analyser
            model_info: Informations sur le modèle
            batch_size: Taille des lots pour le traitement
            
        Returns:
            List[PredictionResult]: Liste des résultats de prédiction
        """
        try:
            self._logger.info(f"Prédiction en lot sur {len(images_data)} images")
            
            results = []
            total_batches = (len(images_data) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(images_data))
                batch_images = images_data[start_idx:end_idx]
                
                self._logger.debug(f"Traitement du lot {batch_idx + 1}/{total_batches}")
                
                # Traitement du lot
                batch_results = []
                for img_idx, image_data in enumerate(batch_images):
                    result = self.predict_single_image(model, image_data, model_info)
                    result.metadata['batch_index'] = batch_idx
                    result.metadata['image_index'] = start_idx + img_idx
                    batch_results.append(result)
                
                results.extend(batch_results)
            
            self._logger.info(f"Prédiction en lot terminée: {len(results)} résultats")
            return results
            
        except Exception as e:
            self._logger.error(f"Erreur lors de la prédiction en lot: {e}")
            # Retourner des résultats d'erreur pour chaque image
            return [
                PredictionResult(
                    predicted_class="ERROR",
                    confidence=0.0,
                    confidence_scores=None,
                    prediction_time=0.0,
                    input_shape=img.shape if isinstance(img, np.ndarray) else None,
                    model_info=model_info or {},
                    metadata={'error': str(e), 'image_index': i}
                )
                for i, img in enumerate(images_data)
            ]
    
    def predict_from_file(self, 
                         model: Any,
                         file_path: str,
                         model_info: Dict[str, Any] = None) -> PredictionResult:
        """
        Effectue une prédiction à partir d'un fichier image.
        
        Args:
            model: Modèle chargé pour la prédiction
            file_path: Chemin vers le fichier image
            model_info: Informations sur le modèle
            
        Returns:
            PredictionResult: Résultat de la prédiction
        """
        try:
            # Vérification de l'existence du fichier
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
            
            # Chargement de l'image via le data_loader
            image_data = self._load_image_from_file(file_path)
            
            # Prédiction
            result = self.predict_single_image(model, image_data, model_info)
            result.metadata['source_file'] = file_path
            
            return result
            
        except Exception as e:
            self._logger.error(f"Erreur lors de la prédiction depuis fichier {file_path}: {e}")
            return PredictionResult(
                predicted_class="ERROR",
                confidence=0.0,
                confidence_scores=None,
                prediction_time=0.0,
                input_shape=None,
                model_info=model_info or {},
                metadata={'error': str(e), 'source_file': file_path}
            )
    
    def analyze_prediction_confidence(self, 
                                   predictions: List[PredictionResult],
                                   confidence_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Analyse la confiance des prédictions.
        
        Args:
            predictions: Liste des résultats de prédiction
            confidence_threshold: Seuil de confiance pour classification
            
        Returns:
            Dict: Analyse de confiance
        """
        try:
            if not predictions:
                return {'error': 'Aucune prédiction à analyser'}
            
            # Filtrage des prédictions valides
            valid_predictions = [
                pred for pred in predictions 
                if pred.predicted_class != "ERROR" and pred.confidence is not None
            ]
            
            if not valid_predictions:
                return {'error': 'Aucune prédiction valide'}
            
            confidences = [pred.confidence for pred in valid_predictions]
            
            # Statistiques de base
            analysis = {
                'total_predictions': len(predictions),
                'valid_predictions': len(valid_predictions),
                'error_predictions': len(predictions) - len(valid_predictions),
                'confidence_stats': {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences),
                    'median': np.median(confidences)
                },
                'confidence_distribution': {
                    'high_confidence': len([c for c in confidences if c >= confidence_threshold]),
                    'medium_confidence': len([c for c in confidences if 0.5 <= c < confidence_threshold]),
                    'low_confidence': len([c for c in confidences if c < 0.5])
                }
            }
            
            # Analyse par classe
            class_analysis = {}
            for pred in valid_predictions:
                class_name = pred.predicted_class
                if class_name not in class_analysis:
                    class_analysis[class_name] = {
                        'count': 0,
                        'confidences': [],
                        'avg_confidence': 0.0
                    }
                
                class_analysis[class_name]['count'] += 1
                class_analysis[class_name]['confidences'].append(pred.confidence)
            
            # Calcul des moyennes par classe
            for class_name, data in class_analysis.items():
                data['avg_confidence'] = np.mean(data['confidences'])
                del data['confidences']  # Suppression pour alléger
            
            analysis['by_class'] = class_analysis
            
            # Recommandations
            recommendations = []
            if analysis['confidence_stats']['mean'] < 0.7:
                recommendations.append("Confiance moyenne faible - vérifier la qualité du modèle")
            if analysis['confidence_distribution']['low_confidence'] > len(valid_predictions) * 0.2:
                recommendations.append("Beaucoup de prédictions à faible confiance - revoir les seuils")
            if analysis['error_predictions'] > 0:
                recommendations.append(f"{analysis['error_predictions']} prédictions ont échoué")
            
            analysis['recommendations'] = recommendations
            
            return analysis
            
        except Exception as e:
            self._logger.error(f"Erreur lors de l'analyse de confiance: {e}")
            return {'error': str(e)}
    
    def clear_model_cache(self) -> int:
        """
        Vide le cache des modèles chargés.
        
        Returns:
            int: Nombre de modèles supprimés du cache
        """
        cleared_count = len(self._loaded_models)
        self._loaded_models.clear()
        self._logger.info(f"Cache des modèles vidé: {cleared_count} modèles supprimés")
        return cleared_count
    
    def get_cached_models_info(self) -> Dict[str, Any]:
        """
        Retourne des informations sur les modèles en cache.
        
        Returns:
            Dict: Informations sur le cache
        """
        return {
            'cached_models_count': len(self._loaded_models),
            'cached_models': list(self._loaded_models.keys())
        }
    
    def _preprocess_single_image(self, image_data: np.ndarray) -> np.ndarray:
        """
        Préprocesse une image pour la prédiction.
        
        Args:
            image_data: Données image brutes
            
        Returns:
            np.ndarray: Image préprocessée
        """
        try:
            # Utilisation du data_loader si méthode disponible
            if hasattr(self._data_loader, 'preprocess_image'):
                return self._data_loader.preprocess_image(image_data, target_size=(224, 224))
            else:
                # Préprocessing basique
                # Normalisation
                if image_data.dtype != np.float32:
                    image_data = image_data.astype(np.float32)
                
                # Normalisation des pixels [0, 255] -> [0, 1]
                if image_data.max() > 1.0:
                    image_data = image_data / 255.0
                
                return image_data
                
        except Exception as e:
            self._logger.warning(f"Erreur preprocessing, utilisation données brutes: {e}")
            return image_data
    
    def _load_image_from_file(self, file_path: str) -> np.ndarray:
        """
        Charge une image depuis un fichier.
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            np.ndarray: Données de l'image
        """
        try:
            # Tentative avec PIL/OpenCV selon disponibilité
            try:
                from PIL import Image
                img = Image.open(file_path)
                return np.array(img)
            except ImportError:
                try:
                    import cv2
                    img = cv2.imread(file_path)
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except ImportError:
                    raise ImportError("PIL ou OpenCV requis pour charger les images")
                    
        except Exception as e:
            self._logger.error(f"Erreur chargement image {file_path}: {e}")
            raise
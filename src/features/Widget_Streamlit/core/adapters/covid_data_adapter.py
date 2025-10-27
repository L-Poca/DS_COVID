"""
Adaptateur pour le covid_data_loader existant.
Wrap le chargeur de données existant pour respecter les interfaces Clean Architecture.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Import robuste du covid_data_loader via le gestionnaire d'imports
from .import_manager import import_manager

# Récupération sécurisée du covid_data_loader
covid_data_loader = import_manager.get_covid_data_loader()
COVID_DATA_LOADER_AVAILABLE = covid_data_loader is not None and not hasattr(covid_data_loader, '_is_mock')

from ..interfaces import ICovidDataLoader
from ..entities import DataConfig, DataSource


class CovidDataLoaderAdapter(ICovidDataLoader):
    """
    Adaptateur pour le covid_data_loader existant.
    Implémente l'interface ICovidDataLoader en wrappant le code legacy.
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialise l'adaptateur avec le data loader existant.
        
        Args:
            project_root: Chemin racine du projet (auto-détecté si None)
        """
        self._logger = logging.getLogger(__name__)
        
        if not COVID_DATA_LOADER_AVAILABLE:
            raise ImportError("covid_data_loader non disponible")
        
        try:
            # Configuration du project root
            if project_root is None:
                project_root = self._find_project_root()
            
            self._project_root = Path(project_root)
            self._data_paths = covid_data_loader.get_data_paths(project_root)
            self._cached_data = {}  # Cache pour éviter les rechargements
            
            # Validation des chemins
            self._validate_data_paths()
            
            self._logger.info(f"Adaptateur COVID data loader initialisé: {project_root}")
            
        except Exception as e:
            self._logger.error(f"Erreur initialisation adaptateur data loader: {e}")
            raise
    
    def load_data(self, config: DataConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Charge les données selon la configuration spécifiée.
        
        Args:
            config: Configuration du chargement
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (données, labels)
        """
        try:
            self._logger.info(f"Chargement données COVID - source: {config.data_source}, taille: {config.image_size}")
            
            # Vérification du cache
            cache_key = self._generate_cache_key(config)
            if cache_key in self._cached_data:
                self._logger.debug("Données trouvées en cache")
                return self._cached_data[cache_key]
            
            # Utilisation du data loader existant
            if hasattr(covid_data_loader, 'load_covid_dataset'):
                # Fonction principale de chargement
                X, y, class_names = covid_data_loader.load_covid_dataset(
                    data_paths=self._data_paths,
                    target_size=config.image_size,
                    load_masks=config.load_masks,
                    sample_size=config.sample_size,
                    classes=config.classes if config.classes else None,
                    balance_classes=config.balance_classes,
                    random_state=config.random_state
                )
            else:
                # Fallback avec fonctions individuelles
                X, y = self._load_data_fallback(config)
            
            # Normalisation si demandée
            if config.normalize:
                X = self._normalize_images(X)
            
            # Mise en cache
            self._cached_data[cache_key] = (X, y)
            
            self._logger.info(f"Données chargées: {X.shape}, classes: {np.unique(y)}")
            return X, y
            
        except Exception as e:
            self._logger.error(f"Erreur chargement données: {e}")
            raise
    
    def get_available_sizes(self) -> List[Tuple[int, int]]:
        """
        Retourne les tailles d'images disponibles.
        
        Returns:
            List[Tuple[int, int]]: Liste des tailles (width, height)
        """
        # Tailles standard supportées par le projet
        standard_sizes = [
            (64, 64),
            (128, 128),
            (224, 224),
            (256, 256),
            (299, 299),
            (512, 512)
        ]
        
        # Vérifier quelles tailles ont des données preprocessées
        available_sizes = []
        data_processed_path = self._project_root / "data" / "processed"
        
        if data_processed_path.exists():
            for size in standard_sizes:
                size_folder = data_processed_path / f"{size[0]}x{size[1]}"
                if size_folder.exists():
                    available_sizes.append(size)
        
        # Si aucune donnée preprocessée, retourner les tailles standard
        if not available_sizes:
            available_sizes = standard_sizes
        
        return available_sizes
    
    def get_class_distribution(self, config: DataConfig) -> Dict[str, int]:
        """
        Analyse la distribution des classes dans le dataset.
        
        Args:
            config: Configuration du dataset
            
        Returns:
            Dict[str, int]: Nombre d'échantillons par classe
        """
        try:
            if hasattr(covid_data_loader, 'get_class_distribution'):
                return covid_data_loader.get_class_distribution(self._data_paths)
            else:
                # Calcul manuel de la distribution
                distribution = {}
                
                for class_name, class_info in self._data_paths.items():
                    if 'images' in class_info:
                        images_path = Path(class_info['images'])
                        if images_path.exists():
                            image_count = len(list(images_path.glob('*.png'))) + \
                                        len(list(images_path.glob('*.jpg'))) + \
                                        len(list(images_path.glob('*.jpeg')))
                            distribution[class_name] = image_count
                        else:
                            distribution[class_name] = 0
                
                return distribution
                
        except Exception as e:
            self._logger.error(f"Erreur calcul distribution classes: {e}")
            return {}
    
    def validate_data_integrity(self, config: DataConfig) -> Dict[str, Any]:
        """
        Valide l'intégrité des données.
        
        Args:
            config: Configuration à valider
            
        Returns:
            Dict[str, Any]: Rapport de validation
        """
        try:
            validation_report = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'statistics': {}
            }
            
            # Vérification de l'existence des chemins
            for class_name, class_info in self._data_paths.items():
                if 'images' in class_info:
                    images_path = Path(class_info['images'])
                    if not images_path.exists():
                        validation_report['is_valid'] = False
                        validation_report['errors'].append(f"Dossier images manquant: {class_name}")
                    else:
                        # Compter les images
                        image_files = list(images_path.glob('*.png')) + \
                                    list(images_path.glob('*.jpg')) + \
                                    list(images_path.glob('*.jpeg'))
                        validation_report['statistics'][f'{class_name}_images'] = len(image_files)
                        
                        if len(image_files) == 0:
                            validation_report['warnings'].append(f"Aucune image trouvée: {class_name}")
            
            # Validation de la taille demandée
            available_sizes = self.get_available_sizes()
            if config.image_size not in available_sizes:
                validation_report['warnings'].append(
                    f"Taille {config.image_size} non preprocessée, redimensionnement nécessaire"
                )
            
            # Distribution des classes
            distribution = self.get_class_distribution(config)
            validation_report['statistics']['class_distribution'] = distribution
            
            # Vérification équilibre des classes
            if distribution:
                counts = list(distribution.values())
                if max(counts) / min(counts) > 5:  # Ratio > 5:1
                    validation_report['warnings'].append("Dataset très déséquilibré détecté")
            
            return validation_report
            
        except Exception as e:
            self._logger.error(f"Erreur validation intégrité: {e}")
            return {
                'is_valid': False,
                'errors': [str(e)],
                'warnings': [],
                'statistics': {}
            }
    
    def get_sample_images(self, class_name: str, n_samples: int = 5) -> List[np.ndarray]:
        """
        Récupère des échantillons d'images pour une classe donnée.
        
        Args:
            class_name: Nom de la classe
            n_samples: Nombre d'échantillons à récupérer
            
        Returns:
            List[np.ndarray]: Liste d'images échantillons
        """
        try:
            if hasattr(covid_data_loader, 'get_sample_images'):
                return covid_data_loader.get_sample_images(
                    self._data_paths, class_name, n_samples
                )
            else:
                # Implémentation manuelle
                samples = []
                
                if class_name in self._data_paths and 'images' in self._data_paths[class_name]:
                    images_path = Path(self._data_paths[class_name]['images'])
                    
                    if images_path.exists():
                        image_files = list(images_path.glob('*.png'))[:n_samples]
                        
                        for img_file in image_files:
                            try:
                                img = self._load_single_image(str(img_file))
                                if img is not None:
                                    samples.append(img)
                            except Exception as e:
                                self._logger.warning(f"Erreur chargement {img_file}: {e}")
                
                return samples[:n_samples]
                
        except Exception as e:
            self._logger.error(f"Erreur récupération échantillons {class_name}: {e}")
            return []
    
    def split_data(self, 
                   config: DataConfig,
                   train_size: float = 0.7,
                   val_size: float = 0.15,
                   test_size: float = 0.15,
                   random_state: Optional[int] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Divise les données en ensembles d'entraînement, validation et test.
        
        Args:
            config: Configuration des données
            train_size: Proportion pour l'entraînement
            val_size: Proportion pour la validation
            test_size: Proportion pour le test
            random_state: Seed pour la reproductibilité
            
        Returns:
            Dict: Ensembles divisés
        """
        try:
            # Vérification des proportions
            if abs(train_size + val_size + test_size - 1.0) > 0.01:
                raise ValueError("Les proportions doivent sommer à 1.0")
            
            # Chargement des données
            X, y = self.load_data(config)
            
            # Division initiale: (train+val) / test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )
            
            # Division: train / val
            val_ratio = val_size / (train_size + val_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio,
                random_state=random_state,
                stratify=y_temp
            )
            
            return {
                'train': (X_train, y_train),
                'val': (X_val, y_val),
                'test': (X_test, y_test)
            }
            
        except Exception as e:
            self._logger.error(f"Erreur division données: {e}")
            raise
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Retourne les métadonnées du dataset COVID.
        
        Returns:
            Dict[str, Any]: Informations sur le dataset
        """
        try:
            metadata = {
                'dataset_name': 'COVID-19 Radiography Dataset',
                'classes': list(covid_data_loader.COVID_CLASSES.keys()) if hasattr(covid_data_loader, 'COVID_CLASSES') else ['COVID', 'Normal', 'Viral Pneumonia'],
                'project_root': str(self._project_root),
                'data_paths': self._data_paths,
                'available_sizes': self.get_available_sizes(),
                'class_distribution': self.get_class_distribution(DataConfig())
            }
            
            return metadata
            
        except Exception as e:
            self._logger.error(f"Erreur récupération métadonnées: {e}")
            return {'error': str(e)}
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Préprocesse une image individuelle.
        
        Args:
            image: Image à préprocesser
            target_size: Taille cible
            
        Returns:
            np.ndarray: Image préprocessée
        """
        try:
            if hasattr(covid_data_loader, 'preprocess_image'):
                return covid_data_loader.preprocess_image(image, target_size)
            else:
                # Préprocessing basique
                return self._resize_image(image, target_size)
                
        except Exception as e:
            self._logger.error(f"Erreur preprocessing image: {e}")
            return image
    
    def clear_cache(self) -> int:
        """Vide le cache des données chargées."""
        cleared_count = len(self._cached_data)
        self._cached_data.clear()
        self._logger.info(f"Cache vidé: {cleared_count} entrées supprimées")
        return cleared_count
    
    def _find_project_root(self) -> str:
        """Trouve automatiquement la racine du projet."""
        current_path = Path(__file__).parent
        
        # Remonte jusqu'à trouver le dossier data ou DS_COVID
        while current_path.parent != current_path:
            if (current_path / 'data').exists() or current_path.name == 'DS_COVID':
                return str(current_path)
            current_path = current_path.parent
        
        # Fallback
        return str(Path(__file__).parent.parent.parent.parent)
    
    def _validate_data_paths(self):
        """Valide que les chemins de données existent."""
        if not self._data_paths:
            raise ValueError("Aucun chemin de données trouvé")
        
        for class_name, paths in self._data_paths.items():
            if 'images' in paths:
                if not Path(paths['images']).exists():
                    self._logger.warning(f"Chemin images manquant: {class_name}")
    
    def _generate_cache_key(self, config: DataConfig) -> str:
        """Génère une clé de cache basée sur la configuration."""
        return f"{config.data_source}_{config.image_size}_{config.sample_size}_{config.normalize}_{config.balance_classes}"
    
    def _load_data_fallback(self, config: DataConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Chargement de données en fallback si fonction principale indisponible."""
        # TODO: Implémenter un chargement de base
        raise NotImplementedError("Fallback data loading pas encore implémenté")
    
    def _normalize_images(self, images: np.ndarray) -> np.ndarray:
        """Normalise les images."""
        if images.dtype != np.float32:
            images = images.astype(np.float32)
        
        if images.max() > 1.0:
            images = images / 255.0
            
        return images
    
    def _load_single_image(self, image_path: str) -> Optional[np.ndarray]:
        """Charge une seule image."""
        try:
            from PIL import Image
            img = Image.open(image_path)
            return np.array(img)
        except Exception:
            try:
                import cv2
                img = cv2.imread(image_path)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                self._logger.error(f"Impossible de charger {image_path}: {e}")
                return None
    
    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Redimensionne une image."""
        try:
            import cv2
            return cv2.resize(image, target_size)
        except Exception:
            try:
                from PIL import Image
                img = Image.fromarray(image)
                img = img.resize(target_size)
                return np.array(img)
            except Exception as e:
                self._logger.error(f"Erreur redimensionnement: {e}")
                return image
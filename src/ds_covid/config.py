"""
Configuration Package pour ds-covid
Gestion centralisée des paramètres pour le package installable

Utilisation:
    from ds_covid.config import Settings, get_settings
    
    settings = get_settings()
    print(settings.batch_size)
"""

from pathlib import Path
from typing import List, Optional, Union
import os
from dataclasses import dataclass, field
import json
from importlib import resources

@dataclass
class MLSettings:
    """Configuration pour les modèles ML traditionnels - Optimisée pour Colab Pro"""
    random_forest: dict = field(default_factory=lambda: {
        'n_estimators': 500,     # Plus d'arbres
        'max_depth': 20,         # Plus profond
        'min_samples_split': 3,
        'min_samples_leaf': 1,
        'max_features': 'sqrt'
    })
    xgboost: dict = field(default_factory=lambda: {
        'n_estimators': 300,     # Plus d'estimateurs
        'learning_rate': 0.05,   # Plus fin
        'max_depth': 6,          # Plus profond
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    })
    gradient_boosting: dict = field(default_factory=lambda: {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_samples_split': 3,
        'min_samples_leaf': 1,
        'subsample': 0.8
    })
    extra_trees: dict = field(default_factory=lambda: {
        'n_estimators': 400,
        'max_depth': 25,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    })
    cv_folds: int = 5            # Plus de folds
    n_jobs: int = -1             # Utilise tous les CPU

@dataclass
class DeepLearningSettings:
    """Configuration pour le deep learning - Optimisée pour Colab Pro"""
    pretrained_weights: str = 'imagenet'
    freeze_base_layers: bool = True
    fine_tune_layers: int = 20  # Plus de couches à fine-tuner
    data_augmentation: bool = True  # Augmentation de données
    mixed_precision: bool = True   # Precision mixte pour Colab Pro
    callbacks: dict = field(default_factory=lambda: {
        'early_stopping_patience': 15,  # Plus de patience avec plus d'epochs
        'reduce_lr_patience': 7,
        'reduce_lr_factor': 0.3,
        'min_lr': 1e-8,
        'model_checkpoint': True,  # Sauvegarde des meilleurs modèles
        'tensorboard': True       # Logs pour TensorBoard
    })
    
    # Architectures à tester (pour ensemble methods)
    architectures: List[str] = field(default_factory=lambda: [
        'EfficientNetB3',  # Plus puissant qu'B0
        'ResNet152V2',     # Plus profond que ResNet50
        'VGG19',           # Plus profond que VGG16
        'DenseNet201'      # Architecture dense
    ])

@dataclass
class TrainingSettings:
    """Configuration pour l'entraînement - Optimisée pour Colab Pro"""
    batch_size: int = 64  # Plus gros batch pour Colab Pro
    epochs: int = 100     # Plus d'epochs avec GPU puissant
    learning_rate: float = 0.001
    validation_split: float = 0.2
    test_split: float = 0.2
    random_seed: int = 42
    img_size: tuple = (256, 256)  # Puissance de 2, optimal pour GPU
    img_channels: int = 3

@dataclass
class DataSettings:
    """Configuration pour les données - Optimisée pour Colab Pro"""
    class_names: List[str] = field(default_factory=lambda: [
        'COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia'
    ])
    max_images_per_class: int = 5000  # Plus d'images avec Colab Pro
    sample_size_analysis: int = 500   # Échantillon plus large
    
    # Paramètres d'augmentation de données
    augmentation_params: dict = field(default_factory=lambda: {
        'rotation_range': 20,
        'width_shift_range': 0.15,
        'height_shift_range': 0.15,
        'shear_range': 0.15,
        'zoom_range': 0.15,
        'horizontal_flip': True,
        'brightness_range': [0.8, 1.2],
        'fill_mode': 'nearest'
    })
    
    # Préprocessing avancé
    preprocessing: dict = field(default_factory=lambda: {
        'normalize': True,
        'standardize': True,
        'histogram_equalization': False,  # Optionnel pour images médicales
        'noise_reduction': False          # Optionnel pour améliorer qualité
    })
    
    @property
    def num_classes(self) -> int:
        return len(self.class_names)
    
    @property
    def class_mapping(self) -> dict:
        return {name: idx for idx, name in enumerate(self.class_names)}

@dataclass
class VisualizationSettings:
    """Configuration pour la visualisation"""
    plot_style: str = 'seaborn-v0_8'
    color_palette: str = 'husl'
    figure_size: tuple = (12, 8)
    dpi: int = 100

@dataclass
class Settings:
    """Configuration principale du package ds-covid"""
    
    # Sous-configurations
    training: TrainingSettings = field(default_factory=TrainingSettings)
    data: DataSettings = field(default_factory=DataSettings)
    ml: MLSettings = field(default_factory=MLSettings)
    deep_learning: DeepLearningSettings = field(default_factory=DeepLearningSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)
    
    # Paramètres généraux
    verbose: int = 1
    log_level: str = 'INFO'
    
    # Chemins (résolus dynamiquement)
    _data_dir: Optional[Path] = None
    _models_dir: Optional[Path] = None
    _results_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Initialisation post-création"""
        # Chargement depuis variables d'environnement si disponibles
        self._load_from_env()
    
    def _load_dotenv(self):
        """Charge le fichier .env s'il existe"""
        try:
            # Recherche du .env dans les emplacements possibles
            possible_env_paths = [
                Path.cwd() / '.env',
                Path.cwd().parent / '.env',  # Si on est dans notebooks/
                Path(__file__).parent.parent.parent / '.env'  # Racine du package
            ]
            
            for env_path in possible_env_paths:
                if env_path.exists():
                    with open(env_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                key, value = key.strip(), value.strip()
                                # Supprime les guillemets si présents
                                if value.startswith('"') and value.endswith('"'):
                                    value = value[1:-1]
                                os.environ[key] = value
                    print(f"📄 Fichier .env chargé: {env_path}")
                    break
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement .env: {e}")
    
    def _load_from_env(self):
        """Charge les paramètres depuis les variables d'environnement"""
        # Chargement du fichier .env si présent
        self._load_dotenv()
        
        # Training
        self.training.batch_size = int(os.getenv('BATCH_SIZE', self.training.batch_size))
        self.training.epochs = int(os.getenv('EPOCHS', self.training.epochs))
        self.training.learning_rate = float(os.getenv('LEARNING_RATE', self.training.learning_rate))
        
        # Data
        if os.getenv('CLASS_NAMES'):
            self.data.class_names = os.getenv('CLASS_NAMES').split(',')
        self.data.max_images_per_class = int(os.getenv('MAX_IMAGES_PER_CLASS', self.data.max_images_per_class))
        
        # Verbose
        self.verbose = int(os.getenv('VERBOSE', self.verbose))
    
    @property
    def project_root(self) -> Path:
        """Racine du projet (depuis PROJECT_ROOT ou auto-détection)"""
        project_root = os.getenv('PROJECT_ROOT', '.')
        if project_root == '.':
            # Auto-détection de la racine (contient pyproject.toml)
            current_dir = Path.cwd()
            while current_dir != current_dir.parent:
                if (current_dir / 'pyproject.toml').exists():
                    return current_dir
                current_dir = current_dir.parent
            return Path.cwd()  # Fallback
        else:
            return Path(project_root).expanduser().resolve()
    
    @property
    def data_dir(self) -> Path:
        """Répertoire des données (depuis DATA_DIR ou relatif à PROJECT_ROOT)"""
        if self._data_dir is None:
            env_path = os.getenv('DATA_DIR')
            if env_path:
                if env_path.startswith('.'):
                    # Chemin relatif à PROJECT_ROOT
                    self._data_dir = (self.project_root / env_path).resolve()
                else:
                    # Chemin absolu
                    self._data_dir = Path(env_path).expanduser().resolve()
            else:
                # Valeur par défaut
                self._data_dir = self.project_root / 'data'
        return self._data_dir
    
    @data_dir.setter
    def data_dir(self, value: Union[str, Path]):
        """Définit le répertoire des données"""
        self._data_dir = Path(value).expanduser().resolve()
    
    @property
    def models_dir(self) -> Path:
        """Répertoire des modèles (depuis MODELS_DIR ou relatif à PROJECT_ROOT)"""
        if self._models_dir is None:
            env_path = os.getenv('MODELS_DIR')
            if env_path:
                if env_path.startswith('.'):
                    # Chemin relatif à PROJECT_ROOT
                    self._models_dir = (self.project_root / env_path).resolve()
                else:
                    # Chemin absolu
                    self._models_dir = Path(env_path).expanduser().resolve()
            else:
                # Valeur par défaut
                self._models_dir = self.project_root / 'models'
        return self._models_dir
    
    @models_dir.setter
    def models_dir(self, value: Union[str, Path]):
        self._models_dir = Path(value).expanduser().resolve()
    
    @property
    def results_dir(self) -> Path:
        """Répertoire des résultats (depuis RESULTS_DIR ou relatif à PROJECT_ROOT)"""
        if self._results_dir is None:
            env_path = os.getenv('RESULTS_DIR')  
            if env_path:
                if env_path.startswith('.'):
                    # Chemin relatif à PROJECT_ROOT
                    self._results_dir = (self.project_root / env_path).resolve()
                else:
                    # Chemin absolu
                    self._results_dir = Path(env_path).expanduser().resolve()
            else:
                # Valeur par défaut
                self._results_dir = self.project_root / 'results'
        return self._results_dir
    
    @results_dir.setter
    def results_dir(self, value: Union[str, Path]):
        self._results_dir = Path(value).expanduser().resolve()
    
    def create_directories(self):
        """Crée les répertoires nécessaires"""
        for dir_path in [self.models_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                print(f"📁 Créé/vérifié: {dir_path}")
    
    def setup_environment(self):
        """Configure l'environnement pour l'entraînement"""
        import warnings
        import numpy as np
        
        # Warnings
        warnings.filterwarnings('ignore')
        
        # Seeds
        np.random.seed(self.training.random_seed)
        
        # TensorFlow si disponible
        try:
            import tensorflow as tf
            tf.random.set_seed(self.training.random_seed)
            
            # Configuration GPU
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                if self.verbose:
                    print(f"🚀 GPU configuré: {len(physical_devices)} device(s)")
            else:
                if self.verbose:
                    print("💻 Configuration CPU")
        except ImportError:
            if self.verbose:
                print("⚠️ TensorFlow non disponible")
        
        # Matplotlib/Seaborn si disponibles
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.style.use(self.visualization.plot_style)
            sns.set_palette(self.visualization.color_palette)
        except ImportError:
            pass
        
        if self.verbose:
            print("✅ Environnement configuré")
    
    def save_config(self, path: Union[str, Path]):
        """Sauvegarde la configuration dans un fichier JSON"""
        config_dict = {
            'training': {
                'batch_size': self.training.batch_size,
                'epochs': self.training.epochs,
                'learning_rate': self.training.learning_rate,
                'validation_split': self.training.validation_split,
                'test_split': self.training.test_split,
                'random_seed': self.training.random_seed,
                'img_size': self.training.img_size,
                'img_channels': self.training.img_channels
            },
            'data': {
                'class_names': self.data.class_names,
                'max_images_per_class': self.data.max_images_per_class,
                'sample_size_analysis': self.data.sample_size_analysis
            },
            'ml': {
                'random_forest': self.ml.random_forest,
                'xgboost': self.ml.xgboost,
                'cv_folds': self.ml.cv_folds,
                'n_jobs': self.ml.n_jobs
            },
            'deep_learning': {
                'pretrained_weights': self.deep_learning.pretrained_weights,
                'freeze_base_layers': self.deep_learning.freeze_base_layers,
                'fine_tune_layers': self.deep_learning.fine_tune_layers,
                'callbacks': self.deep_learning.callbacks
            },
            'visualization': {
                'plot_style': self.visualization.plot_style,
                'color_palette': self.visualization.color_palette,
                'figure_size': self.visualization.figure_size,
                'dpi': self.visualization.dpi
            }
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        if self.verbose:
            print(f"💾 Configuration sauvegardée: {path}")
    
    def print_summary(self):
        """Affiche un résumé de la configuration"""
        print("=" * 60)
        print("🔧 CONFIGURATION DU PACKAGE DS-COVID")
        print("=" * 60)
        
        print(f"\n🎯 ENTRAÎNEMENT:")
        print(f"   📊 Batch size: {self.training.batch_size}")
        print(f"   🔄 Époques: {self.training.epochs}")
        print(f"   📈 Learning rate: {self.training.learning_rate}")
        print(f"   🖼️ Taille d'image: {self.training.img_size}")
        
        print(f"\n🏷️ DONNÉES:")
        print(f"   📋 Classes ({self.data.num_classes}): {', '.join(self.data.class_names)}")
        print(f"   📊 Max images/classe: {self.data.max_images_per_class}")
        
        print(f"\n📁 CHEMINS:")
        print(f"   📂 Données: {self.data_dir}")
        print(f"   🤖 Modèles: {self.models_dir}")
        print(f"   📊 Résultats: {self.results_dir}")
        
        print("=" * 60)

# Instance globale des paramètres
_global_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Retourne l'instance globale des paramètres"""
    global _global_settings
    if _global_settings is None:
        _global_settings = Settings()
    return _global_settings

def configure_package(
    data_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    results_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> Settings:
    """
    Configure le package ds-covid
    
    Args:
        data_dir: Répertoire des données
        models_dir: Répertoire des modèles
        results_dir: Répertoire des résultats
        **kwargs: Autres paramètres de configuration
    
    Returns:
        Instance de Settings configurée
    """
    settings = get_settings()
    
    # Configuration des chemins
    if data_dir:
        settings.data_dir = data_dir
    if models_dir:
        settings.models_dir = models_dir
    if results_dir:
        settings.results_dir = results_dir
    
    # Configuration des autres paramètres
    for key, value in kwargs.items():
        if hasattr(settings.training, key):
            setattr(settings.training, key, value)
        elif hasattr(settings.data, key):
            setattr(settings.data, key, value)
        elif hasattr(settings, key):
            setattr(settings, key, value)
    
    # Setup de l'environnement
    settings.setup_environment()
    settings.create_directories()
    
    return settings
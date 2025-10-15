"""
Configuration Manager pour le projet DS_COVID
Charge et gère les variables d'environnement depuis le fichier .env

Auteur: Équipe DS_COVID
Date: 15 octobre 2025
Branche: ReVamp
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any
import ast

class ConfigManager:
    """Gestionnaire de configuration centralisé"""
    
    def __init__(self, env_path: str = None):
        """
        Initialise le gestionnaire de configuration
        Détecte automatiquement la racine du projet
        
        Args:
            env_path: Chemin vers le fichier .env (optionnel)
        """
        # Détection automatique de la racine du projet
        if env_path is None:
            # Recherche du dossier contenant .env en remontant depuis ce fichier
            current_path = Path(__file__).parent
            while current_path != current_path.parent:
                if (current_path / '.env').exists() or (current_path / '.env.example').exists():
                    self.project_root = current_path
                    break
                current_path = current_path.parent
            else:
                # Fallback : dossier parent du dossier src
                self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(env_path).parent
        
        self.env_path = self.project_root / '.env'
        
        # Charger les variables d'environnement
        self.load_env_file()
        
        # Configuration chargée
        self.config = self._load_config()
    
    def load_env_file(self):
        """Charge le fichier .env dans les variables d'environnement"""
        if not self.env_path.exists():
            print(f"⚠️ Fichier .env non trouvé: {self.env_path}")
            return
        
        with open(self.env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Ignorer les commentaires et lignes vides
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        
        print(f"✅ Variables d'environnement chargées depuis: {self.env_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis les variables d'environnement"""
        
        # Résolution des chemins relatifs depuis la racine du projet
        def resolve_path(env_var: str, default_relative_path: str) -> Path:
            """Résout un chemin depuis la variable d'environnement ou le défaut"""
            env_value = os.getenv(env_var, default_relative_path)
            if env_value.startswith('./') or env_value == '.':
                # Chemin relatif depuis la racine du projet
                return self.project_root / env_value.lstrip('./')
            elif not os.path.isabs(env_value):
                # Chemin relatif sans ./ 
                return self.project_root / env_value
            else:
                # Chemin absolu
                return Path(env_value)
        
        # Chemins
        paths = {
            'project_root': self.project_root,
            'data_dir': resolve_path('DATA_DIR', './data/raw/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset'),
            'models_dir': resolve_path('MODELS_DIR', './models'),
            'results_dir': resolve_path('RESULTS_DIR', './reports'),
            'notebooks_dir': resolve_path('NOTEBOOKS_DIR', './notebooks')
        }
        
        # Configuration des images
        image_config = {
            'img_size': (
                int(os.getenv('IMG_WIDTH', 224)),
                int(os.getenv('IMG_HEIGHT', 224))
            ),
            'img_channels': int(os.getenv('IMG_CHANNELS', 3)),
        }
        
        # Paramètres d'entraînement
        training_config = {
            'batch_size': int(os.getenv('BATCH_SIZE', 32)),
            'epochs': int(os.getenv('EPOCHS', 50)),
            'learning_rate': float(os.getenv('LEARNING_RATE', 0.001)),
            'validation_split': float(os.getenv('VALIDATION_SPLIT', 0.2)),
            'test_split': float(os.getenv('TEST_SPLIT', 0.2)),
            'random_seed': int(os.getenv('RANDOM_SEED', 42))
        }
        
        # Classes
        class_names_str = os.getenv('CLASS_NAMES', 'COVID,Lung_Opacity,Normal,Viral Pneumonia')
        classes_config = {
            'class_names': class_names_str.split(','),
            'num_classes': int(os.getenv('NUM_CLASSES', 4)),
            'class_mapping': {name.strip(): idx for idx, name in enumerate(class_names_str.split(','))}
        }
        
        # ML Traditionnel
        ml_config = {
            'random_forest': {
                'n_estimators': int(os.getenv('RF_N_ESTIMATORS', 200)),
                'max_depth': int(os.getenv('RF_MAX_DEPTH', 15)),
                'min_samples_split': int(os.getenv('RF_MIN_SAMPLES_SPLIT', 5)),
                'min_samples_leaf': int(os.getenv('RF_MIN_SAMPLES_LEAF', 2))
            },
            'xgboost': {
                'n_estimators': int(os.getenv('XGB_N_ESTIMATORS', 100)),
                'learning_rate': float(os.getenv('XGB_LEARNING_RATE', 0.1)),
                'max_depth': int(os.getenv('XGB_MAX_DEPTH', 3)),
                'min_child_weight': int(os.getenv('XGB_MIN_CHILD_WEIGHT', 1))
            },
            'cv_folds': int(os.getenv('CV_FOLDS', 3)),
            'n_jobs': int(os.getenv('N_JOBS', -1))
        }
        
        # Deep Learning
        dl_config = {
            'pretrained_weights': os.getenv('PRETRAINED_WEIGHTS', 'imagenet'),
            'freeze_base_layers': os.getenv('FREEZE_BASE_LAYERS', 'True').lower() == 'true',
            'fine_tune_layers': int(os.getenv('FINE_TUNE_LAYERS', 10)),
            'callbacks': {
                'early_stopping_patience': int(os.getenv('EARLY_STOPPING_PATIENCE', 10)),
                'reduce_lr_patience': int(os.getenv('REDUCE_LR_PATIENCE', 5)),
                'reduce_lr_factor': float(os.getenv('REDUCE_LR_FACTOR', 0.5)),
                'min_lr': float(os.getenv('MIN_LR', 1e-7))
            }
        }
        
        # Performance
        performance_config = {
            'max_images_per_class': int(os.getenv('MAX_IMAGES_PER_CLASS', 1000)),
            'sample_size_analysis': int(os.getenv('SAMPLE_SIZE_ANALYSIS', 200)),
            'verbose': int(os.getenv('VERBOSE', 1))
        }
        
        # Visualisation
        viz_config = {
            'plot_style': os.getenv('PLOT_STYLE', 'seaborn-v0_8'),
            'color_palette': os.getenv('COLOR_PALETTE', 'husl'),
            'figure_size': (
                int(os.getenv('FIGURE_SIZE_WIDTH', 12)),
                int(os.getenv('FIGURE_SIZE_HEIGHT', 8))
            ),
            'dpi': int(os.getenv('DPI', 100))
        }
        
        return {
            'paths': paths,
            'image': image_config,
            'training': training_config,
            'classes': classes_config,
            'ml': ml_config,
            'deep_learning': dl_config,
            'performance': performance_config,
            'visualization': viz_config
        }
    
    def get(self, section: str, key: str = None, default: Any = None):
        """
        Récupère une valeur de configuration
        
        Args:
            section: Section de configuration
            key: Clé spécifique (optionnel)
            default: Valeur par défaut
        
        Returns:
            Valeur de configuration
        """
        if section not in self.config:
            return default
        
        if key is None:
            return self.config[section]
        
        return self.config[section].get(key, default)
    
    def create_directories(self):
        """Crée les répertoires nécessaires"""
        paths = self.get('paths')
        for path_name, path_obj in paths.items():
            if path_name != 'project_root':  # Le projet root existe déjà
                path_obj.mkdir(parents=True, exist_ok=True)
                print(f"📁 {path_name}: {path_obj}")
    
    def print_summary(self):
        """Affiche un résumé de la configuration"""
        print("=" * 60)
        print("🔧 CONFIGURATION DU PROJET DS_COVID")
        print("=" * 60)
        
        print(f"\n📁 CHEMINS:")
        paths = self.get('paths')
        for name, path in paths.items():
            status = "✅" if path.exists() else "❌"
            print(f"   {status} {name}: {path}")
        
        print(f"\n🖼️ IMAGES:")
        img_config = self.get('image')
        print(f"   📐 Taille: {img_config['img_size']}")
        print(f"   🎨 Canaux: {img_config['img_channels']}")
        
        print(f"\n🎯 ENTRAÎNEMENT:")
        train_config = self.get('training')
        print(f"   📊 Batch size: {train_config['batch_size']}")
        print(f"   🔄 Époques: {train_config['epochs']}")
        print(f"   📈 Learning rate: {train_config['learning_rate']}")
        
        print(f"\n🏷️ CLASSES:")
        classes = self.get('classes', 'class_names')
        print(f"   📋 {len(classes)} classes: {', '.join(classes)}")
        
        print("=" * 60)

# Instance globale du gestionnaire de configuration
config_manager = ConfigManager()

# Fonctions utilitaires pour accès rapide
def get_config(section: str, key: str = None, default: Any = None):
    """Accès rapide à la configuration"""
    return config_manager.get(section, key, default)

def setup_environment():
    """Configure l'environnement de développement"""
    import warnings
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Configuration des warnings
    warnings.filterwarnings('ignore')
    
    # Configuration matplotlib/seaborn
    viz_config = get_config('visualization')
    plt.style.use(viz_config['plot_style'])
    sns.set_palette(viz_config['color_palette'])
    
    # Configuration numpy
    np.random.seed(get_config('training', 'random_seed'))
    
    # Configuration TensorFlow si disponible
    try:
        import tensorflow as tf
        tf.random.set_seed(get_config('training', 'random_seed'))
        
        # Configuration GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"🚀 GPU configuré: {len(physical_devices)} device(s)")
        else:
            print("💻 Configuration CPU")
            
    except ImportError:
        print("⚠️ TensorFlow non disponible")
    
    print("✅ Environnement configuré")

if __name__ == "__main__":
    # Test de la configuration
    config_manager.print_summary()
    config_manager.create_directories()
    setup_environment()
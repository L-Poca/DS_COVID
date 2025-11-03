"""
╔════════════════════════════════════════════════════════════════════════════╗
║  🎯 CELLULE DE CONFIGURATION STANDALONE - COPIER-COLLER DANS VOS NOTEBOOKS ║
╚════════════════════════════════════════════════════════════════════════════╝

INSTRUCTIONS:
-------------
1. Copiez TOUT le contenu de cette cellule
2. Collez-le comme PREMIÈRE CELLULE de votre notebook
3. Exécutez la cellule
4. La variable 'config' est prête à l'emploi !

Cette cellule est 100% autonome et fonctionne partout :
✅ Google Colab (clone + installe automatiquement)
✅ WSL / Linux Local
✅ Tout environnement Jupyter

APRÈS EXÉCUTION, VOUS POUVEZ UTILISER:
- config: Objet de configuration (config.batch_size, config.data_dir, etc.)
- ENV: Environnement détecté ('colab', 'wsl', 'local')
- from features.raf.*: Tous les imports du projet

"""

# =============================================================================
# IMPORTS STANDARDS
# =============================================================================

import os
import sys
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


# =============================================================================
# DÉTECTION AUTOMATIQUE DE L'ENVIRONNEMENT
# =============================================================================

def detect_environment():
    """Détecte l'environnement (colab, wsl, local)"""
    try:
        import google.colab
        return "colab"
    except ImportError:
        is_wsl = os.path.exists('/proc/version') and 'microsoft' in open('/proc/version').read().lower()
        return "wsl" if is_wsl else "local"

ENV = detect_environment()
print(f"🌍 Environnement: {ENV.upper()}")


# =============================================================================
# BOOTSTRAP COLAB (Clone + Install si nécessaire)
# =============================================================================

if ENV == "colab":
    print("\n🚀 Bootstrap Colab...")
    
    os.chdir('/content')
    if not os.path.exists('/content/DS_COVID'):
        print("📥 Clonage du repository...")
        subprocess.run(['git', 'clone', 'https://github.com/L-Poca/DS_COVID.git'], check=True)
    
    os.chdir('/content/DS_COVID')
    subprocess.run(['git', 'checkout', 'copilot/data-viz-exploratory-analysis-again'], 
                   capture_output=True, check=False)
    
    print("📦 Installation des dépendances...")
    subprocess.run(['pip', 'install', '-r', 'requirements-colab.txt', '--quiet'], check=True)
    
    print("📦 Installation du package...")
    result = subprocess.run(['pip', 'install', '-e', '.', '--quiet'], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️ Erreur installation: {result.stderr}")
    else:
        print("✅ Package installé")
    
    print("💾 Montage Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Extraction dataset
    for archive in ['/content/drive/MyDrive/DS_COVID/archive_covid.zip']:
        if os.path.exists(archive):
            print("📦 Extraction dataset...")
            os.makedirs('./data/raw/', exist_ok=True)
            subprocess.run(['unzip', '-o', '-q', archive, '-d', './data/raw/COVID-19_Radiography_Dataset/'])
            break
    
    print("✅ Bootstrap terminé")


# =============================================================================
# AJOUT DU CHEMIN src/ POUR LES IMPORTS
# =============================================================================

# Déterminer project_root selon l'environnement
if ENV == "colab":
    project_root = Path('/content/DS_COVID')
else:
    project_root = Path('/home/cepa/DST/projet_DS/DS_COVID')

# Ajouter src/ au sys.path pour permettre "from features.raf.*"
src_path = str(project_root / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    print(f"✅ Chemin src/ ajouté: {src_path}")


# =============================================================================
# FONCTIONS DE CONFIGURATION (autonomes, sans import externe)
# =============================================================================


@dataclass
class Config:
    """Configuration centralisée du projet"""
    project_root: Path
    data_dir: Path
    models_dir: Path
    results_dir: Path
    
    img_width: int = 256
    img_height: int = 256
    img_channels: int = 3
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    validation_split: float = 0.2
    test_split: float = 0.2
    random_seed: int = 42
    classes: list = field(default_factory=list)
    
    # ML models
    rf_n_estimators: int = 200
    rf_max_depth: int = 15
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2
    xgb_n_estimators: int = 100
    xgb_learning_rate: float = 0.1
    xgb_max_depth: int = 3
    xgb_min_child_weight: int = 1

    # Transfer learning
    pretrained_weights: str = "imagenet"
    freeze_base_layers: bool = True
    fine_tune_layers: int = 10
    
    # Callbacks
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7

    # Validation croisée
    cv_folds: int = 3
    n_jobs: int = -1
    
    # Système
    verbose: int = 1
    log_level: str = "INFO"
    
    # Gestion mémoire
    max_images_per_class: int = 1000
    sample_size_analysis: int = 200
    
    # Visualization
    plot_style: str = "seaborn-v0_8"
    color_palette: str = "husl"
    figure_width: int = 12
    figure_height: int = 8
    dpi: int = 100
    
    # Export
    model_save_format: str = "h5"
    results_format: str = "csv"
    export_predictions: bool = True
    save_plots: bool = True

    # Interpretability
    gradcam_alpha: float = 0.4
    gradcam_colormap: str = "jet"
    gradcam_layer_auto: bool = True
    shap_max_evals: int = 100
    shap_background_size: int = 50
    shap_model_type: str = "auto"
    confidence_high_threshold: float = 0.8
    confidence_medium_threshold: float = 0.6
    
    # Interprétabilité - Rapports
    reports_format: str = "csv"
    reports_save_heatmaps: bool = True
    reports_generate_dashboard: bool = True

    
    def __post_init__(self):
        self.num_classes = len(self.classes) if self.classes else 4
        self.img_size = (self.img_width, self.img_height)
        self.figure_size = (self.figure_width, self.figure_height)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


def deep_merge(base: dict, override: dict) -> dict:
    """Fusionne récursivement deux dictionnaires"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    """Applatie un dictionnaire imbriqué"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_config_files(project_root: Path, environment: str) -> dict:
    """Charge et merge les fichiers de configuration JSON"""
    config_dir = project_root / 'src/features/raf/config'
    default_config_path = config_dir / 'default_config.json'
    colab_config_path = config_dir / 'colab_config.json'
    
    # Charger config par défaut
    try:
        with open(default_config_path, 'r') as f:
            config_data = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Fichier {default_config_path} non trouvé, utilisation config minimale")
        config_data = {}
    
    # Merger avec config Colab si nécessaire
    if environment == "colab" and colab_config_path.exists():
        try:
            with open(colab_config_path, 'r') as f:
                colab_overrides = json.load(f)
            config_data = deep_merge(config_data, colab_overrides)
            print("✅ Configuration Colab chargée et mergée")
        except Exception as e:
            print(f"⚠️ Erreur chargement config Colab: {e}")
    
    return config_data


def build_config(environment: str) -> Config:
    """Construit l'objet Config depuis les fichiers JSON"""
    # project_root déjà défini globalement
    global project_root
    
    # Charger les fichiers JSON
    config_data = load_config_files(project_root, environment)
    flat_config = flatten_dict(config_data)
    
    # Construire les chemins
    data_dir_str = flat_config.get('paths_data_dir', 'data')
    data_dir = project_root / data_dir_str if not Path(data_dir_str).is_absolute() else Path(data_dir_str)
    
    models_dir = project_root / flat_config.get('paths_models_dir', 'models')
    results_dir = project_root / flat_config.get('paths_results_dir', 'results')
    
    # Mapping des champs JSON vers Config
    config_kwargs = {
        'project_root': project_root,
        'data_dir': data_dir,
        'models_dir': models_dir,
        'results_dir': results_dir,
        'img_width': flat_config.get('images_width', 256),
        'img_height': flat_config.get('images_height', 256),
        'img_channels': flat_config.get('images_channels', 3),
        'batch_size': flat_config.get('training_batch_size', 32),
        'epochs': flat_config.get('training_epochs', 50),
        'learning_rate': flat_config.get('training_learning_rate', 0.001),
        'validation_split': flat_config.get('training_validation_split', 0.2),
        'test_split': flat_config.get('training_test_split', 0.2),
        'random_seed': flat_config.get('training_random_seed', 42),
        'classes': flat_config.get('dataset_classes', []),
        'rf_n_estimators': flat_config.get('models_random_forest_n_estimators', 200),
        'rf_max_depth': flat_config.get('models_random_forest_max_depth', 15),
        'xgb_n_estimators': flat_config.get('models_xgboost_n_estimators', 100),
        'xgb_learning_rate': flat_config.get('models_xgboost_learning_rate', 0.1),
        'pretrained_weights': flat_config.get('models_transfer_learning_pretrained_weights', 'imagenet'),
        'freeze_base_layers': flat_config.get('models_transfer_learning_freeze_base_layers', True),
        'fine_tune_layers': flat_config.get('models_transfer_learning_fine_tune_layers', 10),
        'early_stopping_patience': flat_config.get('callbacks_early_stopping_patience', 10),
        'reduce_lr_patience': flat_config.get('callbacks_reduce_lr_patience', 5),
        'reduce_lr_factor': flat_config.get('callbacks_reduce_lr_factor', 0.5),
        'min_lr': flat_config.get('callbacks_min_lr', 1e-7),
        'plot_style': flat_config.get('visualization_plot_style', 'seaborn-v0_8'),
        'color_palette': flat_config.get('visualization_color_palette', 'husl'),
        'figure_width': flat_config.get('visualization_figure_width', 12),
        'figure_height': flat_config.get('visualization_figure_height', 8),
        'dpi': flat_config.get('visualization_dpi', 100),
        'gradcam_alpha': flat_config.get('interpretability_gradcam_alpha', 0.4),
        'gradcam_colormap': flat_config.get('interpretability_gradcam_colormap', 'jet'),
        'shap_max_evals': flat_config.get('interpretability_shap_max_evals', 100),
        'shap_background_size': flat_config.get('interpretability_shap_background_size', 50),
        'confidence_high_threshold': flat_config.get('interpretability_confidence_high_threshold', 0.8),
        'confidence_medium_threshold': flat_config.get('interpretability_confidence_medium_threshold', 0.6),
    }
    
    return Config(**config_kwargs)


# =============================================================================
# CHARGEMENT DE LA CONFIGURATION
# =============================================================================

config = build_config(ENV)


# =============================================================================
# AFFICHAGE DU RÉSUMÉ
# =============================================================================

print("\n" + "=" * 60)
print("✅ CONFIGURATION PRÊTE")
print("=" * 60)
print(f"📂 Projet: {config.project_root}")
print(f"📊 Dataset: {config.data_dir}")
print(f"🏷️ Classes: {', '.join(config.classes)}")
print(f"🎛️ Images: {config.img_size}")
print(f"🔧 Batch: {config.batch_size} | Époques: {config.epochs}")
print(f"📐 Dataset accessible: {'✅' if config.data_dir.exists() else '❌'}")
print("=" * 60)
print("\n💡 Variables disponibles:")
print("   • config: Configuration du projet (Config)")
print("   • ENV: Environnement ('colab', 'wsl', 'local')")
print("\n🎯 Imports disponibles:")
print("   • from features.raf.data import ...")
print("   • from features.raf.interpretability import ...")
print("   • from features.raf.models import ...")
print("=" * 60)

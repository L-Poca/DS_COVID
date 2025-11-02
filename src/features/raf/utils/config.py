# =================================
# CONFIGURATION UNIVERSELLE RAF
# =================================
"""
Configuration universelle simplifiée pour Colab/WSL
Utilise uniquement des fichiers JSON pour la configuration (pas de .env)
"""

import os
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


# Chemins vers les fichiers de configuration
CONFIG_DIR = Path(__file__).parent.parent / 'config'
DEFAULT_CONFIG_PATH = CONFIG_DIR / 'default_config.json'
COLAB_CONFIG_PATH = CONFIG_DIR / 'colab_config.json'


def detect_environment() -> str:
    """
    Détecte l'environnement d'exécution
    
    Returns:
        str: 'colab', 'wsl', ou 'local'
    """
    try:
        import google.colab  # type: ignore
        return "colab"
    except ImportError:
        is_wsl = os.path.exists('/proc/version') and 'microsoft' in open('/proc/version').read().lower()
        return "wsl" if is_wsl else "local"


def _run_command(cmd: list[str], description: str, quiet: bool = True) -> bool:
    """
    Exécute une commande et gère les erreurs
    
    Args:
        cmd: Commande à exécuter
        description: Description de l'action
        quiet: Mode silencieux
        
    Returns:
        bool: Succès de l'exécution
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description}")
            return True
        else:
            print(f"❌ {description}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description}: {e}")
        return False


def _setup_colab_repository() -> bool:
    """Clone et configure le repository sur Colab"""
    os.chdir('/content')
    
    if os.path.exists('/content/DS_COVID'):
        print("✅ Repository déjà présent")
    else:
        print("📥 Clonage du repository...")
        if not _run_command(
            ['git', 'clone', 'https://github.com/L-Poca/DS_COVID.git', '/content/DS_COVID'],
            "Repository cloné"
        ):
            return False
    
    os.chdir('/content/DS_COVID')
    _run_command(
        ['git', 'checkout', 'copilot/data-viz-exploratory-analysis-again'],
        "Branche activée",
        quiet=True
    )
    return True


def _install_dependencies() -> bool:
    """Installe les dépendances Python"""
    requirements_files = ['./requirements-colab.txt', './requirements.txt']
    
    for req_file in requirements_files:
        if os.path.exists(req_file):
            print(f"📦 Installation {req_file}...")
            return _run_command(
                ['pip', 'install', '-r', req_file, '--quiet'],
                f"Dépendances installées depuis {req_file}"
            )
    
    print("⚠️ Aucun fichier requirements trouvé")
    return True


def _install_project() -> bool:
    """Installe le projet en mode éditable"""
    if os.path.exists('./setup.py') or os.path.exists('./pyproject.toml'):
        print("🔧 Installation du projet...")
        return _run_command(
            ['pip', 'install', '-e', '.', '--quiet'],
            "Projet installé"
        )
    return True


def _mount_google_drive():
    """Monte Google Drive si nécessaire"""
    if not os.path.exists('/content/drive'):
        print("💾 Montage Google Drive...")
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive')
    else:
        print("✅ Google Drive déjà monté")


def _setup_dataset():
    """Configure le dataset (extraction si nécessaire)"""
    # Chemins possibles du dataset
    dataset_paths = [
        './data/raw/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset',
        './data/raw/COVID-19_Radiography_Dataset',
    ]
    
    # Vérifier si dataset existe
    for path in dataset_paths:
        if os.path.exists(path) and os.path.exists(f"{path}/COVID"):
            print(f"✅ Dataset trouvé: {path}")
            return
    
    # Extraction depuis Drive
    archive_path = '/content/drive/MyDrive/archive_covid.zip'
    if os.path.exists(archive_path):
        print("📦 Extraction du dataset...")
        os.makedirs('./data/raw/', exist_ok=True)
        _run_command(
            ['unzip', '-o', '-q', archive_path, '-d', './data/raw/'],
            "Dataset extrait"
        )
    else:
        print("⚠️ Archive dataset non trouvée dans Drive")


def setup_colab_environment() -> bool:
    """
    Configure automatiquement l'environnement Colab (simplifié)
    
    Returns:
        bool: Succès de la configuration
    """
    try:
        print("🔄 === CONFIGURATION COLAB ===\n")
        
        if not _setup_colab_repository():
            return False
        
        _install_dependencies()
        _install_project()
        _mount_google_drive()
        _setup_dataset()
        
        print("\n✅ Configuration Colab terminée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur configuration Colab: {e}")
        return False


@dataclass
class Config:
    """Configuration centralisée du projet (chargée depuis JSON)"""
    
    # Chemins de base
    project_root: Path
    data_dir: Path
    models_dir: Path
    results_dir: Path
    
    # Configuration images
    img_width: int = 256
    img_height: int = 256
    img_channels: int = 3
    
    # Paramètres d'entraînement
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    validation_split: float = 0.2
    test_split: float = 0.2
    random_seed: int = 42
    
    # Classes du dataset
    classes: list[str] = field(default_factory=list)
    
    # Random Forest
    rf_n_estimators: int = 200
    rf_max_depth: int = 15
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2
    
    # XGBoost
    xgb_n_estimators: int = 100
    xgb_learning_rate: float = 0.1
    xgb_max_depth: int = 3
    xgb_min_child_weight: int = 1
    
    # Transfer Learning
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
    
    # Visualisation
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

    def __post_init__(self):
        """Calculs dérivés après initialisation"""
        self.num_classes = len(self.classes) if self.classes else 4
        self.img_size = (self.img_width, self.img_height)
        self.figure_size = (self.figure_width, self.figure_height)
        
        # Créer les répertoires nécessaires
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convertit la config en dictionnaire (sans les Paths)"""
        data = asdict(self)
        # Convertir les Path en str
        data['project_root'] = str(self.project_root)
        data['data_dir'] = str(self.data_dir)
        data['models_dir'] = str(self.models_dir)
        data['results_dir'] = str(self.results_dir)
        return data
    
    def save(self, filepath: Path):
        """Sauvegarde la configuration en JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def load_json_config(config_path: Path) -> dict:
    """
    Charge un fichier de configuration JSON
    
    Args:
        config_path: Chemin vers le fichier JSON
        
    Returns:
        dict: Configuration chargée
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Fichier de config non trouvé: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ Erreur de parsing JSON: {e}")
        return {}


def _find_project_root() -> Path:
    """Trouve la racine du projet en cherchant le fichier .env"""
    current = Path.cwd()
    
    for parent in [current] + list(current.parents):
        if (parent / '.env').exists():
            return parent
    
    return current


# Configuration globale (singleton)
_global_config: Optional[Config] = None


def deep_merge(base: dict, override: dict) -> dict:
    """
    Fusionne récursivement deux dictionnaires
    
    Args:
        base: Dictionnaire de base
        override: Dictionnaire de surcharge
        
    Returns:
        dict: Dictionnaire fusionné
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    """
    Applatie un dictionnaire imbriqué en utilisant des clés séparées
    
    Args:
        d: Dictionnaire à aplatir
        parent_key: Clé parent (pour récursion)
        sep: Séparateur pour les clés
        
    Returns:
        dict: Dictionnaire aplati
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_project_config(environment: Optional[str] = None) -> Config:
    """
    Charge la configuration du projet depuis JSON avec surcharges environnement
    
    Args:
        environment: Environnement ('colab' ou None pour local)
        
    Returns:
        Config: Instance de configuration chargée
    """
    # 1. Charger la config par défaut
    config_data = load_json_config(DEFAULT_CONFIG_PATH)
    
    # 2. Appliquer les surcharges environnement si nécessaire
    if environment == "colab" and COLAB_CONFIG_PATH.exists():
        colab_overrides = load_json_config(COLAB_CONFIG_PATH)
        config_data = deep_merge(config_data, colab_overrides)
        print("✅ Configuration Colab chargée")
    
    # 3. Aplatir le dictionnaire pour correspondre aux attributs de Config
    flat_config = flatten_dict(config_data)
    
    # 4. Construire les chemins de base depuis JSON uniquement
    project_root = Path(flat_config.get('paths_project_root', '/home/cepa/DST/projet_DS/DS_COVID'))
    data_dir = Path(flat_config.get('paths_data_dir', str(project_root / 'data')))
    models_dir = Path(flat_config.get('paths_models_dir', str(project_root / 'models')))
    results_dir = Path(flat_config.get('paths_results_dir', str(project_root / 'results')))
    
    # 5. Préparer les arguments pour Config (ne garder que les champs valides)
    config_kwargs: dict = {
        'project_root': project_root,
        'data_dir': data_dir,
        'models_dir': models_dir,
        'results_dir': results_dir,
    }
    
    # Mapping des clés JSON aplaties vers les attributs de Config
    field_mapping = {
        'images_width': ('img_width', int),
        'images_height': ('img_height', int),
        'images_channels': ('img_channels', int),
        'training_batch_size': ('batch_size', int),
        'training_epochs': ('epochs', int),
        'training_learning_rate': ('learning_rate', float),
        'training_validation_split': ('validation_split', float),
        'training_test_split': ('test_split', float),
        'training_random_seed': ('random_seed', int),
        'dataset_classes': ('classes', list),
        'models_random_forest_n_estimators': ('rf_n_estimators', int),
        'models_random_forest_max_depth': ('rf_max_depth', int),
        'models_random_forest_min_samples_split': ('rf_min_samples_split', int),
        'models_random_forest_min_samples_leaf': ('rf_min_samples_leaf', int),
        'models_xgboost_n_estimators': ('xgb_n_estimators', int),
        'models_xgboost_learning_rate': ('xgb_learning_rate', float),
        'models_xgboost_max_depth': ('xgb_max_depth', int),
        'models_xgboost_min_child_weight': ('xgb_min_child_weight', int),
        'models_transfer_learning_pretrained_weights': ('pretrained_weights', str),
        'models_transfer_learning_freeze_base_layers': ('freeze_base_layers', bool),
        'models_transfer_learning_fine_tune_layers': ('fine_tune_layers', int),
        'callbacks_early_stopping_patience': ('early_stopping_patience', int),
        'callbacks_reduce_lr_patience': ('reduce_lr_patience', int),
        'callbacks_reduce_lr_factor': ('reduce_lr_factor', float),
        'callbacks_min_lr': ('min_lr', float),
        'cross_validation_cv_folds': ('cv_folds', int),
        'cross_validation_n_jobs': ('n_jobs', int),
        'system_verbose': ('verbose', int),
        'system_log_level': ('log_level', str),
        'memory_max_images_per_class': ('max_images_per_class', int),
        'memory_sample_size_analysis': ('sample_size_analysis', int),
        'visualization_plot_style': ('plot_style', str),
        'visualization_color_palette': ('color_palette', str),
        'visualization_figure_width': ('figure_width', int),
        'visualization_figure_height': ('figure_height', int),
        'visualization_dpi': ('dpi', int),
        'export_model_save_format': ('model_save_format', str),
        'export_results_format': ('results_format', str),
        'export_export_predictions': ('export_predictions', bool),
        'export_save_plots': ('save_plots', bool),
    }
    
    # Appliquer le mapping avec conversion de type
    for json_key, (config_attr, expected_type) in field_mapping.items():
        if json_key in flat_config:
            value = flat_config[json_key]
            # Convertir au bon type si nécessaire
            if expected_type == list and not isinstance(value, list):
                value = [value]
            elif expected_type != list and not isinstance(value, expected_type):
                try:
                    value = expected_type(value)
                except (ValueError, TypeError):
                    print(f"⚠️ Impossible de convertir {json_key}={value} en {expected_type.__name__}")
                    continue
            config_kwargs[config_attr] = value
    
    # 7. Créer l'instance de Config
    return Config(**config_kwargs)


def get_config() -> Config:
    """Récupère la configuration globale (singleton)"""
    global _global_config
    if _global_config is None:
        env = detect_environment()
        _global_config = get_project_config(environment=env if env == "colab" else None)
    return _global_config


def set_config(config: Config):
    """Définit la configuration globale"""
    global _global_config
    _global_config = config


def setup_universal_environment(check_dataset: bool = False) -> Config:
    """
    Configuration universelle complète - REMPLACE la cellule 1 du notebook
    Point d'entrée principal pour la configuration
    
    Args:
        check_dataset: Si True, compte les images du dataset (peut être lent)
    
    Returns:
        Config: Configuration prête à l'emploi
    """
    print("=" * 60)
    print("🚀 CONFIGURATION UNIVERSELLE RAF")
    print("=" * 60)
    
    # Détecter et configurer l'environnement
    env = detect_environment()
    
    if env == "colab":
        setup_colab_environment()
    
    # Charger la configuration
    config = get_config()
    
    # Affichage récapitulatif
    env_name = {"colab": "Colab", "wsl": "WSL", "local": "Local"}[env]
    
    print(f"\n📊 === RÉCAPITULATIF ===")
    print(f"🌍 Environnement: {env_name}")
    print(f"📂 Projet: {config.project_root}")
    print(f"📊 Dataset: {config.data_dir}")
    print(f"🎛️ Images: {config.img_size}")
    print(f"🏷️ Classes: {config.classes}")
    print(f"🔧 Batch size: {config.batch_size}")
    print(f"🎯 Époques: {config.epochs}")
    
    # Vérification dataset (optionnel, peut être lent)
    if check_dataset and config.data_dir.exists():
        print(f"\n🔍 Vérification du dataset...")
        total_images = 0
        
        for cls in config.classes:
            class_path = config.data_dir / cls
            if class_path.exists():
                images = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg"))
                count = len(images)
                print(f"  {cls}: {count:,} images")
                total_images += count
            else:
                print(f"  {cls}: ❌ Non trouvé")
        
        print(f"🎯 TOTAL: {total_images:,} images")
    elif config.data_dir.exists():
        print(f"\n✅ Dataset accessible: {config.data_dir}")
        print(f"💡 Utilisez setup_universal_environment(check_dataset=True) pour compter les images")
    else:
        print(f"\n❌ Dataset non accessible: {config.data_dir}")
    
    print(f"\n🎉 CONFIGURATION TERMINÉE!")
    print(f"💡 Prêt pour l'entraînement ML/DL")
    print("=" * 60)
    
    return config
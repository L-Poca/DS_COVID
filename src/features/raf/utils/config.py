# =================================
# CONFIGURATION UNIVERSELLE RAF
# =================================
"""
Configuration universelle qui intègre la logique Colab/WSL
Remplace la cellule de configuration du notebook par une approche modulaire
"""

import os
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv


def detect_environment() -> Tuple[bool, bool]:
    """
    Détecte l'environnement d'exécution
    
    Returns:
        tuple: (in_colab, in_wsl)
    """
    try:
        import google.colab
        return True, False  # Colab
    except ImportError:
        # Détecter WSL
        wsl_check = os.path.exists('/proc/version') and 'microsoft' in open('/proc/version').read().lower()
        return False, wsl_check #


def setup_colab_environment(project_root: Path) -> bool:
    """
    Configure automatiquement l'environnement Colab
    
    Args:
        project_root: Racine du projet
        
    Returns:
        bool: Succès de la configuration
    """
    try:
        print("🔄 === CONFIGURATION COLAB AUTOMATIQUE ===")
        
        # 1. Positionnement
        os.chdir('/content')
        
        # 2. Clone du repository si nécessaire
        if not os.path.exists('/content/DS_COVID'):
            print("📥 Clonage du repository...")
            result = subprocess.run(['git', 'clone', 'https://github.com/L-Poca/DS_COVID.git', '/content/DS_COVID'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Repository cloné")
            else:
                print(f"❌ Erreur clone: {result.stderr}")
                return False
        
        # 3. Positionnement dans le projet
        os.chdir('/content/DS_COVID')
        subprocess.run(['git', 'checkout', 'ReVamp'], capture_output=True)
        
        # 4. Installation packages si requirements-colab.txt existe
        if os.path.exists('./requirements-colab.txt'):
            print("📦 Installation requirements-colab.txt...")
            result = subprocess.run(['pip', 'install', '-r', './requirements-colab.txt', '--quiet'], 
                                  capture_output=True)
            if result.returncode == 0:
                print("✅ Requirements Colab installés")
        
        # 5. Installation du projet
        if os.path.exists('./setup.py') or os.path.exists('./pyproject.toml'):
            print("🔧 Installation du projet...")
            subprocess.run(['pip', 'install', '-e', '.', '--quiet'], capture_output=True)
        
        # 6. Montage Google Drive
        if not os.path.exists('/content/drive'):
            print("💾 Montage Google Drive...")
            from google.colab import drive
            drive.mount('/content/drive')
        
        # 7. Création/mise à jour du .env pour Colab
        create_colab_env_file()
        
        # 8. Extraction dataset si nécessaire
        setup_colab_dataset()
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur configuration Colab: {e}")
        return False


def create_colab_env_file():
    """Crée un fichier .env optimisé pour Colab à partir du template"""
    # Chemin vers le template
    template_path = Path(__file__).parent.parent / 'templates' / 'colab_env.template'
    
    try:
        # Lecture du template
        with open(template_path, 'r', encoding='utf-8') as f:
            colab_env_content = f.read()
        
        # Écriture du fichier .env
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(colab_env_content)
            
        print("✅ Fichier .env Colab créé depuis template")
        return True
        
    except FileNotFoundError:
        print(f"⚠️ Template Colab non trouvé: {template_path}")
        # Fallback vers l'ancienne méthode si template introuvable
        return _create_colab_env_fallback()
    except Exception as e:
        print(f"❌ Erreur création .env Colab: {e}")
        return False


def _create_colab_env_fallback():
    """Fallback: création .env Colab sans template"""
    colab_env_content = '''# CONFIGURATION COLAB AUTO-GÉNÉRÉE RAF (FALLBACK)
PROJECT_ROOT=.
DATA_DIR=./data/raw/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset
MODELS_DIR=./models
RESULTS_DIR=./results
IMG_WIDTH=256
IMG_HEIGHT=256
IMG_CHANNELS=3
BATCH_SIZE=32
EPOCHS=50
LEARNING_RATE=0.001
VALIDATION_SPLIT=0.2
TEST_SPLIT=0.2
RANDOM_SEED=42
ARCHIVE_PATH=/content/drive/MyDrive/archive_covid.zip
USE_GPU=true'''
    
    with open('.env', 'w') as f:
        f.write(colab_env_content)
    print("✅ Fichier .env Colab créé")


def setup_colab_dataset():
    """Configure le dataset pour Colab"""
    dataset_paths = [
        './data/raw/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset/',
        './data/raw/COVID-19_Radiography_Dataset/',
    ]
    
    # Vérifier si dataset déjà disponible
    for path in dataset_paths:
        if os.path.exists(path) and os.path.exists(f"{path}/COVID"):
            print(f"✅ Dataset déjà disponible: {path}")
            return
    
    # Extraction depuis Drive si nécessaire
    archive_path = '/content/drive/MyDrive/archive_covid.zip'
    if os.path.exists(archive_path):
        print("📦 Extraction dataset depuis Drive...")
        os.makedirs('./data/raw/', exist_ok=True)
        result = subprocess.run(['unzip', '-o', '-q', archive_path, '-d', './data/raw/'], 
                              capture_output=True)
        if result.returncode == 0:
            print("✅ Dataset extrait")
        else:
            print("❌ Erreur extraction dataset")
    else:
        print("⚠️ Archive dataset non trouvée dans Drive")


@dataclass
class Config:
    """Configuration centralisée du projet"""
    
    # Chemins
    project_root: Path
    data_dir: Path
    models_dir: Path
    results_dir: Path
    
    # Images
    img_width: int = 256
    img_height: int = 256
    img_channels: int = 3
    
    # Entraînement
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    validation_split: float = 0.2
    test_split: float = 0.2
    
    # Classes
    classes: List[str] = field(default_factory=lambda: ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia'])
    num_classes: int = 4
    
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
    
    # Validation croisée
    cv_folds: int = 3
    n_jobs: int = -1
    
    # Transfer Learning
    pretrained_weights: str = "imagenet"
    freeze_base_layers: bool = True
    fine_tune_layers: int = 10
    
    # Callbacks
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7
    
    # Système
    random_seed: int = 42
    verbose: int = 1
    log_level: str = "INFO"
    
    # Gestion mémoire
    max_images_per_class: int = 1000
    sample_size_analysis: int = 200
    
    # Visualisation
    plot_style: str = "seaborn-v0_8"
    color_palette: str = "husl"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    
    # Export
    model_save_format: str = "h5"
    results_format: str = "csv"
    export_predictions: bool = True
    save_plots: bool = True

    def __post_init__(self):
        """Post-traitement après initialisation"""
        if self.classes is None:
            self.classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
        self.num_classes = len(self.classes)
        self.img_size = (self.img_width, self.img_height)


def get_project_config(env_file: Optional[Path] = None, auto_setup: bool = True) -> Config:
    """
    Configuration universelle avec setup automatique Colab/WSL
    
    Args:
        env_file: Chemin optionnel vers le fichier .env
        auto_setup: Active la configuration automatique
        
    Returns:
        Instance de Config avec tous les paramètres
    """
    
    print("🔧 === CONFIGURATION UNIVERSELLE RAF ===")
    
    # 1. Détection de l'environnement
    in_colab, in_wsl = detect_environment()
    
    if in_colab:
        print("📍 Environnement: ☁️ Google Colab")
        if auto_setup:
            success = setup_colab_environment(Path('/content/DS_COVID'))
            if success:
                project_root = Path('/content/DS_COVID')
                env_file = project_root / '.env'
            else:
                print("⚠️ Configuration Colab échouée, mode dégradé")
                project_root = Path.cwd()
        else:
            project_root = Path.cwd()
    else:
        print("📍 Environnement: 💻 WSL/Linux Local")
        
        # Chercher la racine du projet
        if env_file:
            project_root = env_file.parent
        else:
            current = Path.cwd()
            project_root = None
            
            for parent in [current] + list(current.parents):
                if (parent / '.env').exists():
                    project_root = parent
                    env_file = parent / '.env'
                    break
            
            if not project_root:
                project_root = current
                print("⚠️ Aucun .env trouvé, utilisation répertoire courant")
        
        # Vérification environnement virtuel WSL
        venv_path = project_root / '.venv'
        if venv_path.exists():
            print("✅ Environnement virtuel .venv détecté")
        else:
            print("⚠️ Aucun .venv détecté")
    
    # 2. Chargement de la configuration .env
    if env_file and env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Configuration chargée: {env_file}")
    else:
        print("⚠️ Configuration par défaut utilisée")
    
    # Construire la configuration
    config = Config(
        # Chemins
        project_root=project_root,
        data_dir=project_root / 'data' / 'raw' / 'COVID-19_Radiography_Dataset' / 'COVID-19_Radiography_Dataset',
        models_dir=Path(os.getenv('MODELS_DIR', project_root / 'models')),
        results_dir=Path(os.getenv('RESULTS_DIR', project_root / 'results')),
        
        # Images
        img_width=int(os.getenv('IMG_WIDTH', '256')),
        img_height=int(os.getenv('IMG_HEIGHT', '256')),
        img_channels=int(os.getenv('IMG_CHANNELS', '3')),
        
        # Entraînement
        batch_size=int(os.getenv('BATCH_SIZE', '32')),
        epochs=int(os.getenv('EPOCHS', '50')),
        learning_rate=float(os.getenv('LEARNING_RATE', '0.001')),
        validation_split=float(os.getenv('VALIDATION_SPLIT', '0.2')),
        test_split=float(os.getenv('TEST_SPLIT', '0.2')),
        
        # Classes
        classes=os.getenv('CLASS_NAMES', 'COVID,Lung_Opacity,Normal,Viral Pneumonia').split(','),
        
        # Random Forest
        rf_n_estimators=int(os.getenv('RF_N_ESTIMATORS', '200')),
        rf_max_depth=int(os.getenv('RF_MAX_DEPTH', '15')),
        rf_min_samples_split=int(os.getenv('RF_MIN_SAMPLES_SPLIT', '5')),
        rf_min_samples_leaf=int(os.getenv('RF_MIN_SAMPLES_LEAF', '2')),
        
        # XGBoost
        xgb_n_estimators=int(os.getenv('XGB_N_ESTIMATORS', '100')),
        xgb_learning_rate=float(os.getenv('XGB_LEARNING_RATE', '0.1')),
        xgb_max_depth=int(os.getenv('XGB_MAX_DEPTH', '3')),
        xgb_min_child_weight=int(os.getenv('XGB_MIN_CHILD_WEIGHT', '1')),
        
        # Validation croisée
        cv_folds=int(os.getenv('CV_FOLDS', '3')),
        n_jobs=int(os.getenv('N_JOBS', '-1')),
        
        # Transfer Learning
        pretrained_weights=os.getenv('PRETRAINED_WEIGHTS', 'imagenet'),
        freeze_base_layers=os.getenv('FREEZE_BASE_LAYERS', 'True').lower() == 'true',
        fine_tune_layers=int(os.getenv('FINE_TUNE_LAYERS', '10')),
        
        # Callbacks
        early_stopping_patience=int(os.getenv('EARLY_STOPPING_PATIENCE', '10')),
        reduce_lr_patience=int(os.getenv('REDUCE_LR_PATIENCE', '5')),
        reduce_lr_factor=float(os.getenv('REDUCE_LR_FACTOR', '0.5')),
        min_lr=float(os.getenv('MIN_LR', '1e-7')),
        
        # Système
        random_seed=int(os.getenv('RANDOM_SEED', '42')),
        verbose=int(os.getenv('VERBOSE', '1')),
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        
        # Gestion mémoire
        max_images_per_class=int(os.getenv('MAX_IMAGES_PER_CLASS', '1000')),
        sample_size_analysis=int(os.getenv('SAMPLE_SIZE_ANALYSIS', '200')),
        
        # Visualisation
        plot_style=os.getenv('PLOT_STYLE', 'seaborn-v0_8'),
        color_palette=os.getenv('COLOR_PALETTE', 'husl'),
        figure_size=(int(os.getenv('FIGURE_SIZE_WIDTH', '12')), int(os.getenv('FIGURE_SIZE_HEIGHT', '8'))),
        dpi=int(os.getenv('DPI', '100')),
        
        # Export
        model_save_format=os.getenv('MODEL_SAVE_FORMAT', 'h5'),
        results_format=os.getenv('RESULTS_FORMAT', 'csv'),
        export_predictions=os.getenv('EXPORT_PREDICTIONS', 'True').lower() == 'true',
        save_plots=os.getenv('SAVE_PLOTS', 'True').lower() == 'true'
    )
    
    # Créer les répertoires s'ils n'existent pas
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    
    return config


# Configuration globale par défaut
_global_config = None

def get_config() -> Config:
    """Récupère la configuration globale (singleton)"""
    global _global_config
    if _global_config is None:
        _global_config = get_project_config()
    return _global_config

def set_config(config: Config):
    """Définit la configuration globale"""
    global _global_config
    _global_config = config

def setup_universal_environment() -> Config:
    """
    Configuration universelle complète - REMPLACE la cellule 1 du notebook
    
    Returns:
        Config: Configuration prête à l'emploi
    """
    print("=" * 60)
    print("🚀 CONFIGURATION UNIVERSELLE RAF")
    print("=" * 60)
    
    # Configuration avec setup automatique
    config = get_project_config(auto_setup=True)
    set_config(config)
    
    # Affichage récapitulatif
    in_colab, in_wsl = detect_environment()
    env_name = "Colab" if in_colab else "WSL/Linux"
    
    print(f"\n📊 === RÉCAPITULATIF ===")
    print(f"🌍 Environnement: {env_name}")
    print(f"📂 Projet: {config.project_root}")
    print(f"📊 Dataset: {config.data_dir}")
    print(f"🎛️ Images: {config.img_size}")
    print(f"🏷️ Classes: {config.classes}")
    print(f"🔧 Batch size: {config.batch_size}")
    print(f"🎯 Époques: {config.epochs}")
    
    # Vérification dataset
    if config.data_dir.exists():
        print(f"✅ Dataset accessible")
        
        # Comptage rapide des images
        total_images = 0
        for cls in config.classes:
            class_paths = [
                config.data_dir / cls / "images",
                config.data_dir / cls
            ]
            
            for class_path in class_paths:
                if class_path.exists():
                    images = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg"))
                    count = len(images)
                    print(f"  {cls}: {count:,} images")
                    total_images += count
                    break
            else:
                print(f"  {cls}: ❌ Non trouvé")
        
        print(f"🎯 TOTAL: {total_images:,} images")
    else:
        print(f"❌ Dataset non accessible: {config.data_dir}")
    
    print(f"\n🎉 CONFIGURATION UNIVERSELLE TERMINÉE!")
    print(f"💡 Prêt pour l'entraînement ML/DL")
    
    return config
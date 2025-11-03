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

from src.features.raf.utils.config import Config, deep_merge, flatten_dict, build_config

# @dataclass
# class Config:
#     """Configuration centralisée du projet"""
#     project_root: Path
#     data_dir: Path
#     models_dir: Path
#     results_dir: Path
    
#     img_width: int = 256
#     img_height: int = 256
#     img_channels: int = 3
#     batch_size: int = 32
#     epochs: int = 50
#     learning_rate: float = 0.001
#     validation_split: float = 0.2
#     test_split: float = 0.2
#     random_seed: int = 42
#     classes: list = field(default_factory=list)
    
#     # ML models
#     rf_n_estimators: int = 200
#     rf_max_depth: int = 15
#     rf_min_samples_split: int = 5
#     rf_min_samples_leaf: int = 2
#     xgb_n_estimators: int = 100
#     xgb_learning_rate: float = 0.1
#     xgb_max_depth: int = 3
#     xgb_min_child_weight: int = 1

#     # Transfer learning
#     pretrained_weights: str = "imagenet"
#     freeze_base_layers: bool = True
#     fine_tune_layers: int = 10
    
#     # Callbacks
#     early_stopping_patience: int = 10
#     reduce_lr_patience: int = 5
#     reduce_lr_factor: float = 0.5
#     min_lr: float = 1e-7

#     # Validation croisée
#     cv_folds: int = 3
#     n_jobs: int = -1
    
#     # Système
#     verbose: int = 1
#     log_level: str = "INFO"
    
#     # Gestion mémoire
#     max_images_per_class: int = 1000
#     sample_size_analysis: int = 200
    
#     # Visualization
#     plot_style: str = "seaborn-v0_8"
#     color_palette: str = "husl"
#     figure_width: int = 12
#     figure_height: int = 8
#     dpi: int = 100
    
#     # Export
#     model_save_format: str = "h5"
#     results_format: str = "csv"
#     export_predictions: bool = True
#     save_plots: bool = True

#     # Interpretability
#     gradcam_alpha: float = 0.4
#     gradcam_colormap: str = "jet"
#     gradcam_layer_auto: bool = True
#     shap_max_evals: int = 100
#     shap_background_size: int = 50
#     shap_model_type: str = "auto"
#     confidence_high_threshold: float = 0.8
#     confidence_medium_threshold: float = 0.6
    
#     # Interprétabilité - Rapports
#     reports_format: str = "csv"
#     reports_save_heatmaps: bool = True
#     reports_generate_dashboard: bool = True

    
#     def __post_init__(self):
#         self.num_classes = len(self.classes) if self.classes else 4
#         self.img_size = (self.img_width, self.img_height)
#         self.figure_size = (self.figure_width, self.figure_height)
#         self.models_dir.mkdir(parents=True, exist_ok=True)
#         self.results_dir.mkdir(parents=True, exist_ok=True)



# =============================================================================
# CHARGEMENT DE LA CONFIGURATION
# =============================================================================

config = build_config(project_root=project_root, environment=ENV)

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

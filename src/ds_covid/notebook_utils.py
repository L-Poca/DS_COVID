"""
Configuration pour notebooks Jupyter
====================================

Utilitaires spécifiques pour les notebooks du projet ds-covid
Approche simple avec imports directs depuis src/
"""

from pathlib import Path
import sys
import os

def setup_notebook_imports():
    """
    Configure les imports pour les notebooks
    Ajoute simplement src/ au Python path
    """
    # Détection du répertoire courant (normalement notebooks/)
    current_dir = Path.cwd()
    
    # Recherche de la racine du projet
    project_root = current_dir
    if current_dir.name == 'notebooks':
        project_root = current_dir.parent
    elif (current_dir / 'notebooks').exists():
        project_root = current_dir
    else:
        # Recherche récursive du pyproject.toml
        while project_root != project_root.parent:
            if (project_root / 'pyproject.toml').exists():
                break
            project_root = project_root.parent
    
    # Ajout de src/ au Python path
    src_path = project_root / 'src'
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"✅ Ajouté au path: {src_path}")
    else:
        print(f"❌ Dossier src/ non trouvé: {src_path}")
    
    return project_root, src_path

def simple_notebook_setup(verbose=True):
    """
    Setup simple pour notebooks avec imports directs
    
    Returns:
        project_root: Chemin racine du projet
    """
    project_root, src_path = setup_notebook_imports()
    
    if verbose:
        print(f"📁 Projet: {project_root}")
        print("💡 Vous pouvez maintenant faire:")
        print("   from ds_covid.config import Settings")
        print("   from ds_covid import configure_package")
    
    return project_root

def get_project_paths():
    """
    Retourne les chemins principaux du projet
    
    Returns:
        dict: Dictionnaire avec les chemins (data, models, etc.)
    """
    project_root, _ = setup_notebook_imports()
    
    paths = {
        'project_root': project_root,
        'data_dir': project_root / 'data',
        'raw_data': project_root / 'data' / 'raw',
        'covid_data': project_root / 'data' / 'raw' / 'COVID-19_Radiography_Dataset',
        'models_dir': project_root / 'models', 
        'notebooks_dir': project_root / 'notebooks',
        'results_dir': project_root / 'results',
        'reports_dir': project_root / 'reports'
    }
    
    # Vérification des chemins existants
    for name, path in paths.items():
        if path.exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (n'existe pas)")
    
    return paths

# Fonction de convenance
def nb_setup():
    """Alias court pour setup rapide"""
    return simple_notebook_setup()
"""
Utilitaires pour l'application Streamlit COVID-19
"""

import sys
from pathlib import Path


def setup_project_path():
    """
    Configure le chemin du projet pour les imports.
    
    Returns:
        Path: Le chemin racine du projet
    """
    current_dir = Path(__file__).parent
    
    # Trouver le répertoire racine du projet (contient le dossier src)
    project_root = current_dir
    while project_root.parent != project_root and not (project_root / "src").exists():
        project_root = project_root.parent
    
    # Si on n'a pas trouvé, essayer des méthodes alternatives
    if not (project_root / "src").exists():
        # Essayer la méthode traditionnelle depuis le dossier streamlit
        if "streamlit" in str(current_dir):
            project_root = current_dir.parent.parent
        else:
            # Dernier recours : remonter jusqu'à trouver src
            project_root = current_dir
            for _ in range(10):  # Limite pour éviter une boucle infinie
                if (project_root / "src").exists():
                    break
                project_root = project_root.parent
                if project_root.parent == project_root:
                    break
    
    # Ajouter au path Python
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    
    return project_root


def get_config_path(project_root):
    """
    Obtient le chemin vers le fichier de configuration Pipeline_Sklearn.
    
    Args:
        project_root (Path): Le chemin racine du projet
        
    Returns:
        Path: Le chemin vers le fichier de configuration
    """
    return project_root / "src" / "features" / "Pipelines" / "Pipeline_Sklearn_config.json"


def find_project_root_from_file(file_path):
    """
    Trouve le répertoire racine du projet à partir d'un fichier.
    
    Args:
        file_path (str or Path): Le chemin du fichier courant (__file__)
        
    Returns:
        Path: Le chemin racine du projet
    """
    current_dir = Path(file_path).parent
    
    # Remonter jusqu'à trouver le dossier src
    project_root = current_dir
    max_levels = 10  # Limite de sécurité
    
    for _ in range(max_levels):
        if (project_root / "src").exists():
            return project_root
        
        parent = project_root.parent
        if parent == project_root:  # On a atteint la racine du système
            break
        project_root = parent
    
    # Si on n'a pas trouvé, utiliser le chemin courant
    return current_dir
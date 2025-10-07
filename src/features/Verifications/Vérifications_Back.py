from pathlib import Path
import sys

# ====================================================================
# 🔧 BACKEND - LOGIQUE MÉTIER
# ====================================================================


def check_project_directory():
    """Vérifie l'existence du répertoire du projet."""
    project_dir = Path(__file__).parent.parent.parent.parent
    return project_dir if project_dir.exists() else None

def check_data_directory(project_dir):
    """Vérifie l'existence du dossier des données."""
    if not project_dir:
        return None
    data_dir = project_dir / "data/raw/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset"
    return data_dir if data_dir.exists() else None

def analyze_data_content(data_dir):
    """Analyse le contenu du dossier des données."""
    if not data_dir or not data_dir.exists():
        return [], []
    subdirs = [p for p in data_dir.glob("*") if p.is_dir()]
    files = [f for f in data_dir.glob("*") if f.is_file()]
    return subdirs, files

def validate_categories(subdirs):
    """Valide les catégories attendues."""
    expected = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]
    found = [d.name for d in subdirs]
    missing = [cat for cat in expected if cat not in found]
    return len(missing) == 0, found, missing

def validate_structure(subdirs):
    """Valide la structure des sous-dossiers."""
    results = []
    for subdir in subdirs:
        images_path = subdir / "images"
        masks_path = subdir / "masks"
        images_exists = images_path.exists()
        masks_exists = masks_path.exists()
        images_count = len(list(images_path.glob("*"))) if images_exists else 0
        masks_count = len(list(masks_path.glob("*"))) if masks_exists else 0
        
        results.append({
            "name": subdir.name,
            "images_ok": images_exists,
            "masks_ok": masks_exists,
            "images_count": images_count,
            "masks_count": masks_count,
            "complete": images_exists and masks_exists
        })
    
    all_complete = all(r["complete"] for r in results)
    return all_complete, results


def validate_metadata(data_dir):
    """Valide les fichiers de métadonnées."""
    if not data_dir:
        return False, []
    
    metadata_files = [
        "COVID.metadata.xlsx",
        "Normal.metadata.xlsx", 
        "Lung_Opacity.metadata.xlsx",
        "Viral Pneumonia.metadata.xlsx"
    ]
    
    results = []
    for filename in metadata_files:
        filepath = data_dir / filename
        exists = filepath.exists()
        size = filepath.stat().st_size / 1024 if exists else 0
        results.append({
            "name": filename,
            "exists": exists,
            "size_kb": size
        })
    
    all_present = all(r["exists"] for r in results)
    return all_present, results 

def validate_python():
    """Valide la version Python."""
    version = sys.version.split()[0]
    is_compatible = version.startswith("3.12")
    return is_compatible, version

def run_all_checks():
    """Exécute toutes les vérifications et retourne les résultats."""
    # Infrastructure
    project_dir = check_project_directory()
    data_dir = check_data_directory(project_dir)
    python_ok, python_version = validate_python()
    
    # Structure des données
    subdirs, files = analyze_data_content(data_dir)
    categories_ok, found_cats, missing_cats = validate_categories(subdirs)
    structure_ok, structure_results = validate_structure(subdirs)
    
    # Métadonnées
    metadata_ok, metadata_results = validate_metadata(data_dir)
    
    return {
        "project_dir": project_dir,
        "data_dir": data_dir,
        "subdirs": subdirs,
        "files": files,
        "python_ok": python_ok,
        "python_version": python_version,
        "categories_ok": categories_ok,
        "found_categories": found_cats,
        "missing_categories": missing_cats,
        "structure_ok": structure_ok,
        "structure_results": structure_results,
        "metadata_ok": metadata_ok,
        "metadata_results": metadata_results,
        "all_checks_passed": all([
            project_dir is not None,
            data_dir is not None,
            python_ok,
            categories_ok,
            structure_ok,
            metadata_ok
        ])
    }
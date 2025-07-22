"""
Backend pur pour l'inspection des features (sans dépendance Streamlit)
"""
from pathlib import Path
import inspect
import sys


def get_features_files():
    """
    Liste les fichiers Python du dossier features et fournit des informations détaillées sur chacun.

    Returns:
        tuple:
            sorted_files (list of Path): Liste triée des fichiers Python (hors __init__.py).
            features_dir (Path): Chemin du dossier features.
            file_info (dict): Dictionnaire contenant pour chaque fichier :
                - size_kb (float): Taille en kilo-octets.
                - modified (float): Timestamp de dernière modification.
                - lines (int): Nombre de lignes.
                - path (str): Chemin absolu.
                - exists (bool): Existence du fichier.
                - error (str, optionnel): Message d'erreur si problème d'accès.

    Exemple:
        files, features_dir, file_info = get_features_files()
    """
    try:
        # Chemin correct vers le dossier src/features
        current_file = Path(__file__)
        features_dir = current_file.parent
        
        # Vérifier que le dossier existe
        if not features_dir.exists():
            return [], features_dir, {}
        
        # Récupérer tous les fichiers Python sauf __init__.py et Features_*
        all_files = list(features_dir.glob("*.py"))
        sorted_files = sorted([f for f in all_files 
                             if f.name != "__init__.py"])
        
        # Collecter des informations détaillées sur chaque fichier
        file_info = {}
        for file_path in sorted_files:
            try:
                stat = file_path.stat()
                file_info[file_path.name] = {
                    'size_kb': round(stat.st_size / 1024, 2),
                    'modified': stat.st_mtime,
                    'lines': len(file_path.read_text(encoding='utf-8').splitlines()),
                    'path': str(file_path),
                    'exists': file_path.exists()
                }
            except Exception as e:
                file_info[file_path.name] = {
                    'error': str(e),
                    'exists': file_path.exists()
                }
        
        return sorted_files, features_dir, file_info
        
    except Exception as e:
        print(f"Erreur lors de l'analyse du dossier features: {e}")
        return [], Path(), {}


def analyze_module_functions(file_path):
    """
    Analyse dynamiquement un fichier Python et extrait les informations sur toutes ses fonctions définies.

    Args:
        file_path (Path): Chemin vers le fichier Python à analyser.

    Returns:
        list of dict: Liste de dictionnaires, un par fonction, contenant :
            - name (str): Nom de la fonction.
            - doc (str): Docstring de la fonction, ou "Pas de documentation".
            - signature (str): Signature complète de la fonction.
            - parameters (list): Liste des paramètres (nom, annotation, défaut, kind).
            - return_annotation (str|None): Annotation de retour si présente.
            - inputs (list): Inputs extraits de la docstring (si formatée).
            - outputs (list): Outputs extraits de la docstring (si formatée).
            - source (str): Code source de la fonction.
            - source_lines (int): Nombre de lignes de code source.
            - file (str): Nom du fichier d'origine.

    Exemple:
        functions = analyze_module_functions(Path('Features_Core.py'))
    """
    functions_info = []
    
    try:
        # Importer le module dynamiquement
        import importlib.util
        module_name = file_path.stem
        
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        
        # Ajouter le répertoire parent au sys.path temporairement
        original_path = sys.path.copy()
        if str(file_path.parent.parent) not in sys.path:
            sys.path.insert(0, str(file_path.parent.parent))
        
        try:
            spec.loader.exec_module(module)
        finally:
            # Restaurer le sys.path
            sys.path = original_path
        
        # Récupérer toutes les fonctions du module
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            # Ne prendre que les fonctions définies dans ce module
            if obj.__module__ == module_name:
                try:
                    # Récupérer la signature
                    sig = inspect.signature(obj)
                    
                    # Récupérer le code source
                    try:
                        source = inspect.getsource(obj)
                        source_lines = len(source.splitlines())
                    except Exception:
                        source = "Code source non disponible"
                        source_lines = 0
                    
                    # Analyser les paramètres
                    params_info = []
                    for param_name, param in sig.parameters.items():
                        param_info = {
                            'name': param_name,
                            'inputs': [],
                            'outputs': [],
                            'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                            'default': str(param.default) if param.default != inspect.Parameter.empty else None,
                            'kind': str(param.kind)
                        }
                        params_info.append(param_info)
                    
                    # Analyser le type de retour
                    return_annotation = str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else None
                    
                    # Extraction des inputs/outputs désactivée (fonction absente)
                    function_info = {
                        'name': name,
                        'doc': obj.__doc__ or "Pas de documentation",
                        'signature': str(sig),
                        'parameters': params_info,
                        'return_annotation': return_annotation,
                        'inputs': [],
                        'outputs': [],
                        'source': source,
                        'source_lines': source_lines,
                        'file': file_path.name
                    }
                    functions_info.append(function_info)
                except Exception as e:
                    # En cas d'erreur sur une fonction spécifique
                    functions_info.append({
                        'name': name,
                        'doc': f"Erreur lors de l'analyse: {e}",
                        'signature': "Non disponible",
                        'parameters': [],
                        'source': "Non disponible",
                        'source_lines': 0,
                        'file': file_path.name
                    })
                    
    except Exception as e:
        print(f"Erreur lors de l'analyse du module {file_path.name}: {e}")
    
    return functions_info


def get_all_functions_summary():
    """
    Récupère un résumé statistique de toutes les fonctions présentes dans les fichiers features.

    Returns:
        dict: Résumé global avec statistiques (nombre de fonctions, documentées, lignes, etc).

    Exemple:
        summary = get_all_functions_summary()
    """
    files, features_dir, file_info = get_features_files()
    all_functions = []
    for file_path in files:
        functions = analyze_module_functions(file_path)
        all_functions.extend(functions)
    # Calculer les statistiques
    summary = {
        'total_files': len(files),
        'total_functions': len(all_functions),
        'total_lines': sum(f['source_lines'] for f in all_functions),
        'documented_functions': sum(1 for f in all_functions if f['doc'] != "Pas de documentation"),
        'functions_by_file': {},
        'all_functions': all_functions
    }
    
    # Grouper par fichier
    for func in all_functions:
        file_name = func['file']
        if file_name not in summary['functions_by_file']:
            summary['functions_by_file'][file_name] = []
        summary['functions_by_file'][file_name].append(func)
    
    return summary

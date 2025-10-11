import streamlit as st
from pathlib import Path
import inspect
import sys


# Analyse de la structure du dossier Features
def get_features_files():
    """
    Retourne la liste des fichiers dans le dossier Features et ses sous-dossiers avec informations détaillées.
    
    Returns:
        tuple: (sorted_files, features_dir, file_info)
            - sorted_files: Liste des fichiers Python triés avec leurs chemins relatifs
            - features_dir: Chemin vers le dossier features
            - file_info: Dictionnaire avec infos détaillées sur chaque fichier
    """
    try:
        # Chemin correct vers le dossier src/features
        current_file = Path(__file__)
        features_dir = current_file.parent.parent  # Remonte au dossier features
        
        # Vérifier que le dossier existe
        if not features_dir.exists():
            st.warning(f"⚠️ Le dossier features n'existe pas: {features_dir}")
            return [], features_dir, {}
        
        # Récupérer tous les fichiers Python récursivement, sauf __init__.py et __pycache__
        all_files = []
        for py_file in features_dir.rglob("*.py"):
            # Exclure __init__.py et les fichiers dans __pycache__
            if py_file.name != "__init__.py" and "__pycache__" not in str(py_file):
                all_files.append(py_file)
        
        # Trier les fichiers par leur chemin relatif
        sorted_files = sorted(all_files, key=lambda x: str(x.relative_to(features_dir)))
        
        # Collecter des informations détaillées sur chaque fichier
        file_info = {}
        for file_path in sorted_files:
            try:
                stat = file_path.stat()
                # Calculer le chemin relatif pour l'affichage
                relative_path = file_path.relative_to(features_dir)
                display_name = str(relative_path).replace('\\', '/')  # Normaliser les séparateurs
                
                file_info[display_name] = {
                    'size_kb': round(stat.st_size / 1024, 2),
                    'modified': stat.st_mtime,
                    'lines': len(file_path.read_text(encoding='utf-8').splitlines()),
                    'path': str(file_path),
                    'exists': file_path.exists(),
                    'relative_path': display_name,
                    'folder': str(relative_path.parent) if relative_path.parent != Path('.') else 'racine'
                }
            except Exception as e:
                relative_path = file_path.relative_to(features_dir)
                display_name = str(relative_path).replace('\\', '/')
                file_info[display_name] = {
                    'error': str(e),
                    'exists': file_path.exists(),
                    'relative_path': display_name,
                    'folder': str(relative_path.parent) if relative_path.parent != Path('.') else 'racine'
                }
        
        return sorted_files, features_dir, file_info
        
    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse du dossier features: {e}")
        return [], Path(), {}

# Afficher la liste des fichiers dans le dossier Features
def show_features_files():
    """Affiche la liste des fichiers dans le dossier Features et ses sous-dossiers avec informations détaillées."""
    st.subheader("📁 Fichiers dans le dossier Features et sous-dossiers:")
    
    files, features_dir, file_info = get_features_files()
    
    # Afficher le chemin du dossier
    st.info(f"📂 Dossier analysé: `{features_dir}`")
    
    if files:
        st.success(f"✅ {len(files)} fichier(s) Python trouvé(s) (incluant les sous-dossiers)")
        
        # Créer un tableau avec les informations détaillées
        import pandas as pd
        from datetime import datetime
        
        table_data = []
        for file in files:
            # Utiliser le chemin relatif comme clé
            relative_path = file.relative_to(features_dir)
            display_name = str(relative_path).replace('\\', '/')
            info = file_info.get(display_name, {})
            
            if 'error' not in info:
                # Convertir le timestamp en date lisible
                try:
                    modified_date = datetime.fromtimestamp(info['modified']).strftime('%Y-%m-%d %H:%M:%S')
                except:
                    modified_date = "Inconnu"
                
                table_data.append({
                    'Fichier': display_name,
                    'Dossier': info.get('folder', 'racine'),
                    'Taille (KB)': info.get('size_kb', 'N/A'),
                    'Lignes': info.get('lines', 'N/A'),
                    'Modifié': modified_date,
                    'Statut': '✅ OK' if info.get('exists', False) else '❌ Erreur'
                })
            else:
                table_data.append({
                    'Fichier': display_name,
                    'Dossier': info.get('folder', 'racine'),
                    'Taille (KB)': 'Erreur',
                    'Lignes': 'Erreur',
                    'Modifié': 'Erreur',
                    'Statut': f"❌ {info['error']}"
                })
        
        # Afficher le tableau
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
        
        # Liste simple pour compatibilité, groupée par dossier
        with st.expander("📋 Liste simple des fichiers (groupés par dossier)"):
            # Grouper les fichiers par dossier
            files_by_folder = {}
            for file in files:
                relative_path = file.relative_to(features_dir)
                folder = str(relative_path.parent) if relative_path.parent != Path('.') else 'racine'
                if folder not in files_by_folder:
                    files_by_folder[folder] = []
                files_by_folder[folder].append(relative_path.name)
            
            # Afficher par dossier
            for folder, file_list in sorted(files_by_folder.items()):
                if folder == 'racine':
                    st.write(f"📁 **Dossier racine (features/)**")
                else:
                    st.write(f"📁 **{folder}/**")
                for filename in sorted(file_list):
                    st.write(f"  - 📄 {filename}")
                st.write("")  # Ligne vide pour séparer les dossiers
                
    else:
        st.warning(f"⚠️ Aucun fichier Python trouvé dans le dossier `{features_dir}` et ses sous-dossiers")
        
        # Suggestions de dépannage
        st.info("💡 **Suggestions:**")
        st.write("- Vérifiez que le dossier `src/features/` existe")
        st.write("- Assurez-vous qu'il contient des fichiers `.py`")
        st.write("- Vérifiez les permissions d'accès au dossier")
        st.write("- Vérifiez les sous-dossiers comme `Inspector/`, `Verifications/`, etc.")

# Fonction pour afficher les détails d'une fonction
def show_function_details(func):
    """Affiche les détails d'une fonction."""
    st.write(f"### {func.__name__}")
    st.write(f"**Documentation:** {func.__doc__ or 'Pas de documentation disponible'}")
    st.write(f"**Code:**")
    st.code(inspect.getsource(func))

def show_verification_backend():
    """Affiche la page de vérification du backend avec diagnostics avancés."""
    st.title("🔧 Vérifications Backend")
    
    # Section 1: Vérification de l'importation du module
    st.subheader("1️⃣ Test d'importation du module")
    
    try:
        # Tentative d'import (le PYTHONPATH est géré dans app.py)
        from src.features.Verifs_Env.Vérifications_Back import run_all_checks
        
        st.success("✅ Module Vérifications importé avec succès")
        
        # Test de la fonction
        result = run_all_checks()
        st.info(f"� Test de la fonction run_all_checks(): Toutes les vérifications passées: {result.get('all_checks_passed', 'N/A')}")
        
    except ImportError as e:
        st.error(f"❌ Erreur d'importation: {e}")
        
        # Diagnostic détaillé
        with st.expander("🔍 Diagnostic détaillé"):
            st.write("**Chemins testés:**")
            st.code(f"Répertoire actuel: {Path(__file__).parent}")
            st.code(f"Racine projet: {Path(__file__).parent.parent.parent.parent}")
            st.code(f"Dossier features: {Path(__file__).parent.parent.parent.parent / 'src' / 'features'}")
            
            
    except Exception as e:
        st.error(f"❌ Erreur générale: {e}")
    
    st.divider()
    
    # Section 2: Analyse du dossier Features
    st.subheader("2️⃣ Analyse du dossier Features")
    
    files, features_dir, file_info = get_features_files()
    
    # Afficher le chemin et le statut du dossier
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"📂 **Dossier analysé:** `{features_dir}`")
    
    with col2:
        if features_dir.exists():
            st.success("✅ Dossier trouvé")
        else:
            st.error("❌ Dossier non trouvé")
    
    # Afficher les fichiers avec le tableau amélioré
    if files:
        st.success(f"✅ **{len(files)} fichier(s) Python trouvé(s)**")
        
        # Tableau détaillé
        import pandas as pd
        from datetime import datetime
        
        table_data = []
        import pycodestyle
        for file in files:
            # Utiliser le chemin relatif comme clé
            relative_path = file.relative_to(features_dir)
            display_name = str(relative_path).replace('\\', '/')
            info = file_info.get(display_name, {})
            
            if 'error' not in info:
                try:
                    modified_date = datetime.fromtimestamp(info['modified']).strftime('%Y-%m-%d %H:%M:%S')
                except:
                    modified_date = "Inconnu"
                # Test d'import dynamique pour vérifier la présence d'erreurs de code
                import importlib.util
                has_code_error = False
                try:
                    spec = importlib.util.spec_from_file_location(file.stem, file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except Exception:
                    has_code_error = True
                # Test PEP8
                pep8_ok = True
                pep8_count = 0
                try:
                    style_guide = pycodestyle.StyleGuide(quiet=True)
                    report = style_guide.check_files([str(file)])
                    pep8_count = report.total_errors
                    pep8_ok = (pep8_count == 0)
                except Exception:
                    pep8_ok = False
                    pep8_count = -1
                    
                statut = '✅ OK' if info.get('exists', False) and not has_code_error else '❌ Erreur code' if has_code_error else '❌ Erreur'
                pep8_statut = '✅' if pep8_ok else f'❌ ({pep8_count})'
                table_data.append({
                    '📄 Fichier': display_name,
                    '📁 Dossier': info.get('folder', 'racine'),
                    '📏 Taille (KB)': info.get('size_kb', 'N/A'),
                    '📝 Lignes': info.get('lines', 'N/A'),
                    '🕒 Modifié': modified_date,
                    '✅ Statut': statut,
                    'PEP8': pep8_statut
                })
            else:
                table_data.append({
                    '📄 Fichier': display_name,
                    '📁 Dossier': info.get('folder', 'racine'),
                    '📏 Taille (KB)': 'Erreur',
                    '📝 Lignes': 'Erreur', 
                    '🕒 Modifié': 'Erreur',
                    '✅ Statut': f"❌ {info['error']}",
                    'PEP8': '❓'
                })
        
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
            
    else:
        st.warning(f"⚠️ **Aucun fichier Python trouvé**")


def analyze_module_functions(file_path):
    """
    Analyse un fichier Python pour extraire les informations sur ses fonctions.
    
    Args:
        file_path (Path): Chemin vers le fichier Python
        
    Returns:
        list: Liste des dictionnaires contenant les infos des fonctions
    """
    functions_info = []
    
    try:
        # Importer le module dynamiquement
        import importlib.util
        module_name = file_path.stem
        
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
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
                    except:
                        source = "Code source non disponible"
                        source_lines = 0
                    
                    # Analyser les paramètres
                    params_info = []
                    for param_name, param in sig.parameters.items():
                        param_info = {
                            'name': param_name,
                            'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                            'default': str(param.default) if param.default != inspect.Parameter.empty else None,
                            'kind': str(param.kind)
                        }
                        params_info.append(param_info)
                    
                    function_info = {
                        'name': name,
                        'doc': obj.__doc__ or "Pas de documentation",
                        'signature': str(sig),
                        'parameters': params_info,
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
        st.error(f"❌ Erreur lors de l'analyse du module {file_path.name}: {e}")
    
    return functions_info


def show_function_details(func_info):
    """
    Affiche les détails d'une fonction de manière organisée.
    
    Args:
        func_info (dict): Informations sur la fonction
    """
    with st.expander(f"🔧 {func_info['name']}", expanded=False):
        
        # Signature
        st.code(f"def {func_info['name']}{func_info['signature']}", language="python")
        
        # Documentation
        st.markdown("**📖 Documentation:**")
        if func_info['doc'] and func_info['doc'] != "Pas de documentation":
            st.info(func_info['doc'])
        else:
            st.warning("Pas de documentation disponible")
        
        # Paramètres
        if func_info['parameters']:
            st.markdown("**📝 Paramètres:**")
            for param in func_info['parameters']:
                col1, col2, col3 = st.columns([2, 2, 2])
                with col1:
                    st.write(f"**{param['name']}**")
                with col2:
                    if param['annotation']:
                        st.code(param['annotation'], language="python")
                    else:
                        st.write("_Type non spécifié_")
                with col3:
                    if param['default']:
                        st.write(f"Défaut: `{param['default']}`")
                    else:
                        st.write("_Requis_")
        
        # Métriques
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📏 Lignes de code", func_info['source_lines'])
        with col2:
            st.metric("📄 Fichier", func_info['file'])
        
        # Code source
        with st.expander("📋 Code source", expanded=False):
            st.code(func_info['source'], language="python")


def show_features_functions_analysis():
    """
    Affiche l'analyse complète des fonctions pour tous les fichiers features.
    """
    st.header("🔍 Analyse des Fonctions Features")
    
    files, features_dir, file_info = get_features_files()
    
    if not files:
        st.warning("⚠️ Aucun fichier à analyser")
        return
    
    # Créer une liste des options pour le sélecteur avec les chemins relatifs
    file_options = []
    for file in files:
        relative_path = file.relative_to(features_dir)
        display_name = str(relative_path).replace('\\', '/')
        file_options.append(display_name)
    
    # Sélecteur de fichier
    selected_file = st.selectbox(
        "📁 Choisir un fichier à analyser:",
        options=file_options,
        help="Sélectionnez un fichier pour voir ses fonctions"
    )
    
    if selected_file:
        # Trouver le fichier sélectionné
        file_path = None
        for f in files:
            relative_path = f.relative_to(features_dir)
            display_name = str(relative_path).replace('\\', '/')
            if display_name == selected_file:
                file_path = f
                break
        
        if file_path:
            st.markdown(f"### 📄 Analyse de `{selected_file}`")
            
            # Analyser les fonctions
            with st.spinner("🔍 Analyse en cours..."):
                functions_info = analyze_module_functions(file_path)
            
            if functions_info:
                # Statistiques générales
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔧 Fonctions", len(functions_info))
                with col2:
                    total_lines = sum(f['source_lines'] for f in functions_info)
                    st.metric("📏 Lignes totales", total_lines)
                with col3:
                    documented = sum(1 for f in functions_info if f['doc'] != "Pas de documentation")
                    st.metric("📖 Documentées", f"{documented}/{len(functions_info)}")
                
                st.markdown("---")
                
                # Afficher chaque fonction
                for func_info in functions_info:
                    show_function_details(func_info)
                    
            else:
                st.info("ℹ️ Aucune fonction trouvée dans ce fichier")
    
    # Option pour voir toutes les fonctions de tous les fichiers
    if st.checkbox("🌍 Afficher toutes les fonctions de tous les fichiers"):
        st.markdown("## 🌍 Vue d'ensemble de toutes les fonctions")
        
        all_functions = []
        for file_path in files:
            functions = analyze_module_functions(file_path)
            # Mettre à jour le nom du fichier avec le chemin relatif
            relative_path = file_path.relative_to(features_dir)
            display_name = str(relative_path).replace('\\', '/')
            for func in functions:
                func['file'] = display_name
            all_functions.extend(functions)
        
        if all_functions:
            # Tableau récapitulatif
            import pandas as pd
            
            summary_data = []
            for func in all_functions:
                summary_data.append({
                    '🔧 Fonction': func['name'],
                    '📄 Fichier': func['file'],
                    '📏 Lignes': func['source_lines'],
                    '📖 Documentée': '✅' if func['doc'] != "Pas de documentation" else '❌',
                    '📝 Paramètres': len(func['parameters'])
                })
            
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
            
            # Statistiques globales
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🔧 Total fonctions", len(all_functions))
            with col2:
                total_files = len(set(f['file'] for f in all_functions))
                st.metric("📄 Fichiers", total_files)
            with col3:
                total_lines = sum(f['source_lines'] for f in all_functions)
                st.metric("📏 Lignes totales", total_lines)
            with col4:
                documented = sum(1 for f in all_functions if f['doc'] != "Pas de documentation")
                st.metric("📖 Documentation", f"{documented}/{len(all_functions)}")

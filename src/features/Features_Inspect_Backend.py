import streamlit as st
from pathlib import Path
import inspect
import sys


# Analyse de la structure du dossier Features
def get_features_files():
    """
    Retourne la liste des fichiers dans le dossier Features avec informations détaillées.
    
    Returns:
        tuple: (sorted_files, features_dir, file_info)
            - sorted_files: Liste des fichiers Python triés
            - features_dir: Chemin vers le dossier features
            - file_info: Dictionnaire avec infos détaillées sur chaque fichier
    """
    try:
        # Chemin correct vers le dossier src/features
        current_file = Path(__file__)
        features_dir = current_file.parent
        
        # Vérifier que le dossier existe
        if not features_dir.exists():
            st.warning(f"⚠️ Le dossier features n'existe pas: {features_dir}")
            return [], features_dir, {}
        
        # Récupérer tous les fichiers Python sauf __init__.py
        all_files = list(features_dir.glob("*.py"))
        sorted_files = sorted([f for f in all_files if f.name != "__init__.py"])
        
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
        st.error(f"❌ Erreur lors de l'analyse du dossier features: {e}")
        return [], Path(), {}

# Afficher la liste des fichiers dans le dossier Features
def show_features_files():
    """Affiche la liste des fichiers dans le dossier Features avec informations détaillées."""
    st.subheader("📁 Fichiers dans le dossier Features:")
    
    files, features_dir, file_info = get_features_files()
    
    # Afficher le chemin du dossier
    st.info(f"📂 Dossier analysé: `{features_dir}`")
    
    if files:
        st.success(f"✅ {len(files)} fichier(s) Python trouvé(s)")
        
        # Créer un tableau avec les informations détaillées
        import pandas as pd
        from datetime import datetime
        
        table_data = []
        for file in files:
            info = file_info.get(file.name, {})
            
            if 'error' not in info:
                # Convertir le timestamp en date lisible
                try:
                    modified_date = datetime.fromtimestamp(info['modified']).strftime('%Y-%m-%d %H:%M:%S')
                except:
                    modified_date = "Inconnu"
                
                table_data.append({
                    'Fichier': file.name,
                    'Taille (KB)': info.get('size_kb', 'N/A'),
                    'Lignes': info.get('lines', 'N/A'),
                    'Modifié': modified_date,
                    'Statut': '✅ OK' if info.get('exists', False) else '❌ Erreur'
                })
            else:
                table_data.append({
                    'Fichier': file.name,
                    'Taille (KB)': 'Erreur',
                    'Lignes': 'Erreur',
                    'Modifié': 'Erreur',
                    'Statut': f"❌ {info['error']}"
                })
        
        # Afficher le tableau
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
        
        # Liste simple pour compatibilité
        with st.expander("📋 Liste simple des fichiers"):
            for file in files:
                st.write(f"- 📄 {file.name}")
                
    else:
        st.warning(f"⚠️ Aucun fichier Python trouvé dans le dossier `{features_dir}`")
        
        # Suggestions de dépannage
        st.info("💡 **Suggestions:**")
        st.write("- Vérifiez que le dossier `src/features/` existe")
        st.write("- Assurez-vous qu'il contient des fichiers `.py`")
        st.write("- Vérifiez les permissions d'accès au dossier")

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
        from src.features.Vérifications_Back import run_all_checks
        
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
        for file in files:
            info = file_info.get(file.name, {})
            
            if 'error' not in info:
                try:
                    modified_date = datetime.fromtimestamp(info['modified']).strftime('%Y-%m-%d %H:%M:%S')
                except:
                    modified_date = "Inconnu"
                
                table_data.append({
                    '📄 Fichier': file.name,
                    '📏 Taille (KB)': info.get('size_kb', 'N/A'),
                    '📝 Lignes': info.get('lines', 'N/A'),
                    '🕒 Modifié': modified_date,
                    '✅ Statut': '✅ OK' if info.get('exists', False) else '❌ Erreur'
                })
            else:
                table_data.append({
                    '📄 Fichier': file.name,
                    '📏 Taille (KB)': 'Erreur',
                    '📝 Lignes': 'Erreur', 
                    '🕒 Modifié': 'Erreur',
                    '✅ Statut': f"❌ {info['error']}"
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
    
    # Sélecteur de fichier
    selected_file = st.selectbox(
        "📁 Choisir un fichier à analyser:",
        options=[f.name for f in files],
        help="Sélectionnez un fichier pour voir ses fonctions"
    )
    
    if selected_file:
        # Trouver le fichier sélectionné
        file_path = None
        for f in files:
            if f.name == selected_file:
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

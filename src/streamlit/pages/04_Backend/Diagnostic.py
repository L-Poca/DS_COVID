"""
Page de diagnostic pour vérifier le statut de l'architecture Clean.
Aide au debugging et à la résolution des problèmes d'importation.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from typing import Dict, Any
import traceback
import importlib.util

# Ajout du chemin vers les modules
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

def main():
    """Point d'entrée principal de la page de diagnostic."""
    
    st.set_page_config(
        page_title="Diagnostic - COVID Detection",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔧 Diagnostic de l'Architecture Clean")
    st.markdown("---")
    
    st.info("Cette page vous aide à diagnostiquer et résoudre les problèmes de l'architecture Clean.")
    
    # Onglets de diagnostic
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏗️ Architecture", 
        "📦 Modules Legacy", 
        "🔍 Imports", 
        "⚙️ Configuration"
    ])
    
    with tab1:
        diagnose_architecture()
    
    with tab2:
        diagnose_legacy_modules()
    
    with tab3:
        diagnose_imports()
    
    with tab4:
        diagnose_configuration()


def diagnose_architecture():
    """Diagnostic de la structure de l'architecture Clean."""
    
    st.subheader("🏗️ Structure de l'Architecture Clean")
    
    # Vérification de la structure des dossiers
    core_path = project_root / 'src' / 'features' / 'Widget_Streamlit' / 'core'
    
    expected_structure = {
        'core': core_path,
        'entities': core_path / 'entities',
        'interfaces': core_path / 'interfaces', 
        'services': core_path / 'services',
        'adapters': core_path / 'adapters',
        'state': core_path / 'state',
        'components': core_path / 'components'
    }
    
    st.write("**Vérification de la structure des dossiers :**")
    
    all_good = True
    for name, path in expected_structure.items():
        if path.exists():
            st.success(f"✅ {name}/ - {path}")
        else:
            st.error(f"❌ {name}/ - MANQUANT: {path}")
            all_good = False
    
    if all_good:
        st.success("🎉 Structure de l'architecture complète !")
    else:
        st.error("⚠️ Structure incomplète - certains dossiers manquent")
    
    st.markdown("---")
    
    # Vérification des fichiers clés
    st.write("**Vérification des fichiers clés :**")
    
    key_files = {
        'DI Container': core_path / 'di_container.py',
        'Import Manager': core_path / 'adapters' / 'import_manager.py',
        'Error Handler': core_path / 'adapters' / 'error_handler.py',
        'State Manager': core_path / 'state' / 'streamlit_state_manager.py',
        'Components Forms': core_path / 'components' / 'forms.py'
    }
    
    for name, file_path in key_files.items():
        if file_path.exists():
            st.success(f"✅ {name} - {file_path.name}")
        else:
            st.error(f"❌ {name} - MANQUANT: {file_path}")


def diagnose_legacy_modules():
    """Diagnostic des modules legacy existants."""
    
    st.subheader("📦 Modules Legacy")
    
    # Vérification des modules legacy
    legacy_modules = {
        'Pipeline_Sklearn': project_root / 'src' / 'features' / 'Pipelines' / 'Pipeline_Sklearn.py',
        'Pipeline_TensorFlow': project_root / 'src' / 'features' / 'Pipelines' / 'Pipeline_TensorFlow.py',
        'covid_data_loader': project_root / 'src' / 'features' / 'Data_Loaders' / 'covid_data_loader.py'
    }
    
    st.write("**Vérification des fichiers legacy :**")
    
    for name, file_path in legacy_modules.items():
        if file_path.exists():
            st.success(f"✅ {name} - {file_path}")
            
            # Informations sur le fichier
            try:
                stat = file_path.stat()
                size_kb = stat.st_size / 1024
                st.caption(f"   Taille: {size_kb:.1f} KB")
            except:
                pass
        else:
            st.error(f"❌ {name} - MANQUANT: {file_path}")
    
    st.markdown("---")
    
    # Test des imports legacy
    st.write("**Test des imports legacy :**")
    
    try:
        from src.features.Widget_Streamlit.core.adapters.import_manager import import_manager
        
        # Test des imports via le gestionnaire
        sklearn_mgr = import_manager.get_sklearn_pipeline_manager()
        tf_mgr = import_manager.get_tensorflow_pipeline_manager()
        data_loader = import_manager.get_covid_data_loader()
        
        # Statut des imports
        status = {
            'sklearn_pipeline': sklearn_mgr is not None and not hasattr(sklearn_mgr, '_is_mock'),
            'tensorflow_pipeline': tf_mgr is not None and not hasattr(tf_mgr, '_is_mock'),
            'covid_data_loader': data_loader is not None and not hasattr(data_loader, '_is_mock')
        }
        
        for module, is_available in status.items():
            if is_available:
                st.success(f"✅ {module} - Importé avec succès")
            else:
                st.warning(f"⚠️ {module} - Mode dégradé (mock)")
        
        # Affichage du statut via l'error handler
        try:
            from src.features.Widget_Streamlit.core.adapters.error_handler import error_handler
            error_handler.show_dependency_status(status)
        except Exception as e:
            st.error(f"Erreur affichage statut: {e}")
            
    except Exception as e:
        st.error(f"❌ Erreur test imports: {e}")
        st.code(traceback.format_exc())


def diagnose_imports():
    """Diagnostic détaillé des imports."""
    
    st.subheader("🔍 Diagnostic des Imports")
    
    # Test des imports de base
    st.write("**Test des imports de base :**")
    
    basic_imports = [
        ('streamlit', 'st'),
        ('pandas', 'pd'), 
        ('numpy', 'np'),
        ('pathlib', 'Path')
    ]
    
    for module_name, alias in basic_imports:
        try:
            module = importlib.import_module(module_name)
            st.success(f"✅ {module_name} - Version: {getattr(module, '__version__', 'N/A')}")
        except ImportError as e:
            st.error(f"❌ {module_name} - ERREUR: {e}")
    
    st.markdown("---")
    
    # Test des imports ML
    st.write("**Test des imports ML :**")
    
    ml_imports = [
        'sklearn',
        'tensorflow',
        'cv2',
        'matplotlib',
        'seaborn'
    ]
    
    available_ml = []
    
    for module_name in ml_imports:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'N/A')
            st.success(f"✅ {module_name} - Version: {version}")
            available_ml.append(module_name)
        except ImportError as e:
            st.error(f"❌ {module_name} - ERREUR: {e}")
    
    # Recommandations d'installation
    missing_ml = set(ml_imports) - set(available_ml)
    if missing_ml:
        st.warning(f"⚠️ Modules ML manquants: {', '.join(missing_ml)}")
        
        with st.expander("📋 Commandes d'installation"):
            install_commands = {
                'sklearn': 'pip install scikit-learn',
                'tensorflow': 'pip install tensorflow', 
                'cv2': 'pip install opencv-python',
                'matplotlib': 'pip install matplotlib',
                'seaborn': 'pip install seaborn'
            }
            
            for missing in missing_ml:
                if missing in install_commands:
                    st.code(install_commands[missing], language="bash")
    
    st.markdown("---")
    
    # Test des imports architecture
    st.write("**Test des imports architecture Clean :**")
    
    arch_imports = [
        'src.features.Widget_Streamlit.core.di_container',
        'src.features.Widget_Streamlit.core.adapters.import_manager',
        'src.features.Widget_Streamlit.core.services.training_service',
        'src.features.Widget_Streamlit.core.entities.training_config'
    ]
    
    for import_path in arch_imports:
        try:
            module = importlib.import_module(import_path)
            st.success(f"✅ {import_path}")
        except ImportError as e:
            st.error(f"❌ {import_path} - ERREUR: {e}")


def diagnose_configuration():
    """Diagnostic de la configuration."""
    
    st.subheader("⚙️ Configuration du Système")
    
    # Informations système
    st.write("**Informations Système :**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        st.metric("Project Root", str(project_root))
    
    with col2:
        st.metric("Working Directory", os.getcwd())
        st.metric("Sys Path Entries", len(sys.path))
    
    # Chemins Python
    with st.expander("🔍 Chemins Python (sys.path)"):
        for i, path in enumerate(sys.path):
            st.text(f"{i}: {path}")
    
    st.markdown("---")
    
    # Variables d'environnement importantes
    st.write("**Variables d'Environnement :**")
    
    env_vars = ['PYTHONPATH', 'PATH', 'VIRTUAL_ENV']
    
    for var in env_vars:
        value = os.environ.get(var, 'Non définie')
        if value != 'Non définie':
            st.success(f"✅ {var}: {value[:100]}{'...' if len(value) > 100 else ''}")
        else:
            st.info(f"ℹ️ {var}: Non définie")
    
    st.markdown("---")
    
    # Test du DI Container
    st.write("**Test du DI Container :**")
    
    try:
        from src.features.Widget_Streamlit.core.di_container import AppContainerFactory
        
        with st.spinner("Création du container..."):
            container = AppContainerFactory.create_container()
        
        st.success("✅ DI Container créé avec succès!")
        
        # Test des services
        try:
            training_service = container.get_training_service()
            st.success("✅ TrainingService récupéré")
        except Exception as e:
            st.error(f"❌ TrainingService - {e}")
        
        try:
            state_manager = container.get_state_manager()
            st.success("✅ StateManager récupéré")
        except Exception as e:
            st.error(f"❌ StateManager - {e}")
            
    except Exception as e:
        st.error(f"❌ Erreur DI Container: {e}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
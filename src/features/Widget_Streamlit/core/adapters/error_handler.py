"""
Gestionnaire d'erreurs centralisé pour les cas de modules indisponibles.
Fournit des messages d'erreur clairs et des suggestions de résolution.
"""

import streamlit as st
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DependencyErrorHandler:
    """Gestionnaire centralisé des erreurs de dépendances manquantes."""
    
    @staticmethod
    def handle_sklearn_unavailable() -> Dict[str, Any]:
        """
        Gère le cas où sklearn pipeline n'est pas disponible.
        
        Returns:
            Dict avec message d'erreur et suggestions
        """
        error_info = {
            'success': False,
            'error': 'Module Sklearn Pipeline non disponible',
            'error_type': 'dependency_missing',
            'module': 'Pipeline_Sklearn',
            'suggestions': [
                'Vérifier que le module Pipeline_Sklearn.py existe dans src/features/Pipelines/',
                'Installer les dépendances sklearn: pip install scikit-learn',
                'Vérifier la configuration de l\'environnement Python'
            ]
        }
        
        st.error("❌ **Module Sklearn Pipeline Indisponible**")
        st.warning("Le module Pipeline_Sklearn n'a pas pu être importé.")
        
        with st.expander("🔧 Solutions Possibles"):
            st.write("**Vérifications à effectuer :**")
            for i, suggestion in enumerate(error_info['suggestions'], 1):
                st.write(f"{i}. {suggestion}")
            
            st.code("""
# Installation des dépendances sklearn
pip install scikit-learn pandas numpy matplotlib seaborn

# Vérification de l'environnement
python -c "import sklearn; print('sklearn OK')"
            """, language="bash")
        
        return error_info
    
    @staticmethod
    def handle_tensorflow_unavailable() -> Dict[str, Any]:
        """
        Gère le cas où tensorflow pipeline n'est pas disponible.
        
        Returns:
            Dict avec message d'erreur et suggestions
        """
        error_info = {
            'success': False,
            'error': 'Module TensorFlow Pipeline non disponible',
            'error_type': 'dependency_missing',
            'module': 'Pipeline_TensorFlow',
            'suggestions': [
                'Vérifier que le module Pipeline_TensorFlow.py existe dans src/features/Pipelines/',
                'Installer les dépendances tensorflow: pip install tensorflow',
                'Vérifier la configuration GPU si nécessaire',
                'Vérifier la compatibilité des versions tensorflow/keras'
            ]
        }
        
        st.error("❌ **Module TensorFlow Pipeline Indisponible**")
        st.warning("Le module Pipeline_TensorFlow n'a pas pu être importé.")
        
        with st.expander("🔧 Solutions Possibles"):
            st.write("**Vérifications à effectuer :**")
            for i, suggestion in enumerate(error_info['suggestions'], 1):
                st.write(f"{i}. {suggestion}")
            
            st.code("""
# Installation des dépendances tensorflow
pip install tensorflow matplotlib seaborn opencv-python

# Vérification de l'environnement
python -c "import tensorflow; print('TensorFlow OK')"
            """, language="bash")
        
        return error_info
    
    @staticmethod
    def handle_data_loader_unavailable() -> Dict[str, Any]:
        """
        Gère le cas où le data loader n'est pas disponible.
        
        Returns:
            Dict avec message d'erreur et suggestions
        """
        error_info = {
            'success': False,
            'error': 'Module COVID Data Loader non disponible',
            'error_type': 'dependency_missing',
            'module': 'covid_data_loader',
            'suggestions': [
                'Vérifier que le module covid_data_loader.py existe dans src/features/Data_Loaders/',
                'Installer les dépendances opencv: pip install opencv-python',
                'Vérifier que les données COVID sont disponibles',
                'Vérifier la configuration des chemins de données'
            ]
        }
        
        st.error("❌ **Module COVID Data Loader Indisponible**")
        st.warning("Le module covid_data_loader n'a pas pu être importé.")
        
        with st.expander("🔧 Solutions Possibles"):
            st.write("**Vérifications à effectuer :**")
            for i, suggestion in enumerate(error_info['suggestions'], 1):
                st.write(f"{i}. {suggestion}")
            
            st.code("""
# Installation des dépendances opencv
pip install opencv-python pillow numpy pandas

# Vérification de l'environnement
python -c "import cv2; print('OpenCV OK')"
            """, language="bash")
        
        return error_info
    
    @staticmethod
    def show_dependency_status(dependencies_status: Dict[str, bool]):
        """
        Affiche le statut des dépendances dans une interface claire.
        
        Args:
            dependencies_status: Dict avec le statut de chaque dépendance
        """
        st.subheader("🔍 Statut des Dépendances")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sklearn_status = dependencies_status.get('sklearn_pipeline', False)
            if sklearn_status:
                st.success("✅ **Sklearn Pipeline**")
                st.write("Module disponible")
            else:
                st.error("❌ **Sklearn Pipeline**")
                st.write("Module indisponible")
        
        with col2:
            tf_status = dependencies_status.get('tensorflow_pipeline', False)
            if tf_status:
                st.success("✅ **TensorFlow Pipeline**")
                st.write("Module disponible")
            else:
                st.error("❌ **TensorFlow Pipeline**")
                st.write("Module indisponible")
        
        with col3:
            data_status = dependencies_status.get('covid_data_loader', False)
            if data_status:
                st.success("✅ **COVID Data Loader**")
                st.write("Module disponible")
            else:
                st.error("❌ **COVID Data Loader**")
                st.write("Module indisponible")
        
        # Conseils globaux si des modules manquent
        missing_modules = [k for k, v in dependencies_status.items() if not v]
        if missing_modules:
            st.warning(f"⚠️ {len(missing_modules)} module(s) indisponible(s)")
            
            with st.expander("🚀 Installation Complète Recommandée"):
                st.code("""
# Installation de tous les packages nécessaires
pip install scikit-learn tensorflow opencv-python pillow
pip install pandas numpy matplotlib seaborn streamlit

# Vérification complète
python -c "
import sklearn, tensorflow, cv2, pandas, numpy
print('Toutes les dépendances sont installées !')
"
                """, language="bash")
    
    @staticmethod
    def create_fallback_message(module_name: str, operation: str) -> Dict[str, Any]:
        """
        Crée un message de fallback standardisé.
        
        Args:
            module_name: Nom du module manquant
            operation: Opération qui a échoué
            
        Returns:
            Dict avec les informations d'erreur
        """
        return {
            'success': False,
            'error': f'Impossible d\'exécuter {operation}',
            'error_type': 'module_unavailable',
            'module': module_name,
            'message': f'Le module {module_name} n\'est pas disponible. Fonctionnalité en mode dégradé.',
            'timestamp': str(st.session_state.get('current_timestamp', 'unknown'))
        }


# Instance globale du gestionnaire d'erreurs
error_handler = DependencyErrorHandler()
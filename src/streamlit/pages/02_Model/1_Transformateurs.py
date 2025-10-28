"""
Page Streamlit pour scanner et visualiser automatiquement les transformateurs disponibles.

Cette page scanne automatiquement le dossier Transformateurs pour découvrir
les transformateurs personnalisés et liste les transformateurs de base des librairies.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import inspect
import importlib.util
import sys
from typing import Dict, Any, List, Tuple, Optional
import os
import traceback
import warnings

# Supprimer les warnings pour une sortie plus propre
warnings.filterwarnings('ignore')

# Configuration des chemins
try:
    # Ajout du chemin pour importer les modules
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Chemin vers le dossier des transformateurs
    TRANSFORMERS_PATH = project_root / "features" / "Pipelines" / "Transformateurs"
    
except Exception as e:
    st.error(f"Erreur de configuration des chemins: {e}")
    TRANSFORMERS_PATH = None


@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def scan_custom_transformers() -> Dict[str, Dict]:
    """Scanne le dossier Transformateurs pour découvrir les transformateurs personnalisés."""
    custom_transformers = {}
    
    # Vérification de la disponibilité du chemin
    if TRANSFORMERS_PATH is None:
        st.error("Chemin des transformateurs non configuré")
        return custom_transformers
    
    if not TRANSFORMERS_PATH.exists():
        st.error(f"❌ Dossier Transformateurs non trouvé: {TRANSFORMERS_PATH}")
        return custom_transformers
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Lister tous les fichiers Python
        py_files = [f for f in TRANSFORMERS_PATH.glob("*.py") 
                   if not f.name.startswith("__") and f.name != "utilities.py"]
        
        if not py_files:
            st.warning("Aucun fichier Python trouvé dans le dossier Transformateurs")
            return custom_transformers
        
        # Scanner chaque fichier
        for i, py_file in enumerate(py_files):
            progress = (i + 1) / len(py_files)
            progress_bar.progress(progress)
            status_text.text(f"Scan en cours: {py_file.name}")
            
            try:
                # Créer un nom de module unique
                module_name = f"custom_transformer_{py_file.stem}_{i}"
                
                # Charger le module dynamiquement
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    
                    # Exécuter le module dans un environnement isolé
                    try:
                        spec.loader.exec_module(module)
                    except Exception as exec_error:
                        st.warning(f"⚠️ Erreur d'exécution pour {py_file.name}: {str(exec_error)[:100]}...")
                        continue
                    
                    # Analyser les classes
                    found_items = analyze_module_classes(module, py_file)
                    custom_transformers.update(found_items)
                    
                    # Analyser les fonctions
                    found_functions = analyze_module_functions(module, py_file)
                    custom_transformers.update(found_functions)
                        
            except Exception as e:
                st.warning(f"⚠️ Erreur lors du chargement de {py_file.name}: {str(e)[:100]}...")
                continue
        
        # Nettoyer l'affichage
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ Erreur générale lors du scan: {e}")
        progress_bar.empty()
        status_text.empty()
    
    return custom_transformers


def analyze_module_classes(module, py_file) -> Dict[str, Dict]:
    """Analyse les classes d'un module."""
    found_classes = {}
    
    try:
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (name.startswith("_") or 
                not hasattr(obj, '__module__') or 
                obj.__module__ != module.__name__):
                continue
            
            try:
                # Vérifier si c'est un transformateur/estimateur
                is_transformer = check_if_transformer(obj)
                
                if is_transformer:
                    found_classes[name] = {
                        'class': obj,
                        'module': py_file.stem,
                        'file': str(py_file.name),
                        'docstring': clean_docstring(inspect.getdoc(obj)),
                        'methods': get_public_methods(obj),
                        'parameters': get_class_parameters(obj),
                        'type': 'class'
                    }
            except Exception as class_error:
                # Ne pas afficher les erreurs mineures
                pass
                
    except Exception as e:
        pass
    
    return found_classes


def analyze_module_functions(module, py_file) -> Dict[str, Dict]:
    """Analyse les fonctions d'un module."""
    found_functions = {}
    
    try:
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if (not name.startswith("_") and 
                hasattr(obj, '__module__') and
                obj.__module__ == module.__name__ and
                is_relevant_function(name)):
                
                try:
                    found_functions[f"📋 {name}"] = {
                        'function': obj,
                        'module': py_file.stem,
                        'file': str(py_file.name),
                        'docstring': clean_docstring(inspect.getdoc(obj)),
                        'parameters': get_function_parameters(obj),
                        'type': 'function'
                    }
                except Exception:
                    pass
                        
    except Exception:
        pass
    
    return found_functions


def check_if_transformer(obj) -> bool:
    """Vérifie si un objet est un transformateur."""
    try:
        # Vérifier les méthodes essentielles
        has_methods = (
            hasattr(obj, 'fit') or 
            hasattr(obj, 'transform') or 
            hasattr(obj, 'fit_transform') or
            hasattr(obj, 'predict')
        )
        
        # Vérifier l'héritage
        base_names = [base.__name__ for base in obj.__mro__]
        has_inheritance = any(name in base_names for name in [
            'BaseEstimator', 'TransformerMixin', 'ClassifierMixin', 
            'RegressorMixin', 'ClusterMixin'
        ])
        
        return has_methods or has_inheritance
    except:
        return False


def is_relevant_function(name: str) -> bool:
    """Vérifie si une fonction est pertinente."""
    keywords = ['create', 'build', 'make', 'get_', 'load_', 'process_', 'transform_']
    return any(keyword in name.lower() for keyword in keywords)


def get_public_methods(obj) -> List[str]:
    """Obtient les méthodes publiques d'un objet."""
    try:
        methods = [method for method in dir(obj) 
                  if not method.startswith('_') and callable(getattr(obj, method, None))]
        return methods[:10]  # Limiter à 10 méthodes
    except:
        return []


def clean_docstring(docstring: str) -> str:
    """Nettoie et limite la taille d'une docstring."""
    if not docstring:
        return "Pas de documentation disponible"
    
    # Nettoyer et limiter
    cleaned = docstring.strip().replace('\n\n', '\n')
    if len(cleaned) > 200:
        cleaned = cleaned[:200] + "..."
    
    return cleaned


def get_sklearn_transformers() -> Dict[str, Dict]:
    """Liste les transformateurs de base de scikit-learn."""
    sklearn_transformers = {
        # Preprocessing
        "StandardScaler": {
            "module": "sklearn.preprocessing",
            "description": "Standardise les caractéristiques en supprimant la moyenne et en divisant par l'écart-type",
            "category": "Preprocessing"
        },
        "MinMaxScaler": {
            "module": "sklearn.preprocessing",
            "description": "Transforme les caractéristiques en les mettant à l'échelle dans une plage donnée",
            "category": "Preprocessing"
        },
        "RobustScaler": {
            "module": "sklearn.preprocessing",
            "description": "Met à l'échelle les caractéristiques en utilisant des statistiques robustes aux valeurs aberrantes",
            "category": "Preprocessing"
        },
        "Normalizer": {
            "module": "sklearn.preprocessing",
            "description": "Normalise les échantillons individuellement à la norme unitaire",
            "category": "Preprocessing"
        },
        "LabelEncoder": {
            "module": "sklearn.preprocessing",
            "description": "Encode les étiquettes cibles avec des valeurs entre 0 et n_classes-1",
            "category": "Preprocessing"
        },
        "OneHotEncoder": {
            "module": "sklearn.preprocessing",
            "description": "Encode les caractéristiques catégorielles sous forme de tableau one-hot numérique",
            "category": "Preprocessing"
        },
        
        # Feature Selection
        "SelectKBest": {
            "module": "sklearn.feature_selection",
            "description": "Sélectionne les k meilleures caractéristiques selon un test statistique",
            "category": "Feature Selection"
        },
        "SelectPercentile": {
            "module": "sklearn.feature_selection",
            "description": "Sélectionne les caractéristiques selon un percentile des scores les plus élevés",
            "category": "Feature Selection"
        },
        "RFE": {
            "module": "sklearn.feature_selection",
            "description": "Sélection de caractéristiques par élimination récursive",
            "category": "Feature Selection"
        },
        
        # Decomposition
        "PCA": {
            "module": "sklearn.decomposition",
            "description": "Analyse en composantes principales pour la réduction de dimensionnalité",
            "category": "Decomposition"
        },
        "TruncatedSVD": {
            "module": "sklearn.decomposition",
            "description": "Décomposition en valeurs singulières tronquée",
            "category": "Decomposition"
        },
        "FastICA": {
            "module": "sklearn.decomposition",
            "description": "Analyse en composantes indépendantes rapide",
            "category": "Decomposition"
        }
    }
    
    return sklearn_transformers


def get_tensorflow_keras_components() -> Dict[str, Dict]:
    """Liste les composants de base TensorFlow/Keras."""
    tf_components = {
        # Layers
        "Dense": {
            "module": "tensorflow.keras.layers",
            "description": "Couche dense (entièrement connectée)",
            "category": "Layers"
        },
        "Conv2D": {
            "module": "tensorflow.keras.layers",
            "description": "Couche de convolution 2D",
            "category": "Layers"
        },
        "MaxPooling2D": {
            "module": "tensorflow.keras.layers",
            "description": "Couche de max pooling 2D",
            "category": "Layers"
        },
        "Dropout": {
            "module": "tensorflow.keras.layers",
            "description": "Couche de dropout pour la régularisation",
            "category": "Layers"
        },
        "BatchNormalization": {
            "module": "tensorflow.keras.layers",
            "description": "Normalisation par lots",
            "category": "Layers"
        },
        "Flatten": {
            "module": "tensorflow.keras.layers",
            "description": "Aplatit l'entrée sans affecter la dimension du lot",
            "category": "Layers"
        },
        
        # Activations
        "ReLU": {
            "module": "tensorflow.keras.layers",
            "description": "Fonction d'activation ReLU",
            "category": "Activations"
        },
        "LeakyReLU": {
            "module": "tensorflow.keras.layers",
            "description": "Fonction d'activation Leaky ReLU",
            "category": "Activations"
        },
        
        # Optimizers
        "Adam": {
            "module": "tensorflow.keras.optimizers",
            "description": "Optimiseur Adam",
            "category": "Optimizers"
        },
        "SGD": {
            "module": "tensorflow.keras.optimizers",
            "description": "Optimiseur de descente de gradient stochastique",
            "category": "Optimizers"
        },
        "RMSprop": {
            "module": "tensorflow.keras.optimizers",
            "description": "Optimiseur RMSprop",
            "category": "Optimizers"
        },
        
        # Pre-trained Models
        "VGG16": {
            "module": "tensorflow.keras.applications",
            "description": "Modèle VGG16 pré-entraîné",
            "category": "Pre-trained Models"
        },
        "ResNet50": {
            "module": "tensorflow.keras.applications",
            "description": "Modèle ResNet50 pré-entraîné",
            "category": "Pre-trained Models"
        },
        "InceptionV3": {
            "module": "tensorflow.keras.applications",
            "description": "Modèle InceptionV3 pré-entraîné",
            "category": "Pre-trained Models"
        },
        "MobileNetV2": {
            "module": "tensorflow.keras.applications",
            "description": "Modèle MobileNetV2 pré-entraîné (léger)",
            "category": "Pre-trained Models"
        },
        "EfficientNetB0": {
            "module": "tensorflow.keras.applications",
            "description": "Modèle EfficientNetB0 pré-entraîné",
            "category": "Pre-trained Models"
        }
    }
    
    return tf_components


def get_class_parameters(cls) -> Dict[str, Any]:
    """Extrait les paramètres d'initialisation d'une classe."""
    try:
        if not hasattr(cls, '__init__'):
            return {}
            
        signature = inspect.signature(cls.__init__)
        params = {}
        
        for name, param in signature.parameters.items():
            if name != 'self':
                # Formater la valeur par défaut
                default_val = param.default
                if default_val == inspect.Parameter.empty:
                    default_str = 'Requis'
                elif default_val is None:
                    default_str = 'None'
                elif isinstance(default_val, str):
                    default_str = f"'{default_val}'"
                else:
                    default_str = str(default_val)
                
                # Formater l'annotation de type
                annotation = param.annotation
                if annotation == inspect.Parameter.empty:
                    type_str = 'Any'
                else:
                    type_str = str(annotation).replace('<class \'', '').replace('\'>', '')
                    if len(type_str) > 20:
                        type_str = type_str[:20] + '...'
                
                params[name] = {
                    'default': default_str,
                    'annotation': type_str
                }
        
        return params
    except Exception:
        return {}


def get_function_parameters(func) -> Dict[str, Any]:
    """Extrait les paramètres d'une fonction."""
    try:
        signature = inspect.signature(func)
        params = {}
        
        for name, param in signature.parameters.items():
            # Formater la valeur par défaut
            default_val = param.default
            if default_val == inspect.Parameter.empty:
                default_str = 'Requis'
            elif default_val is None:
                default_str = 'None'
            elif isinstance(default_val, str):
                default_str = f"'{default_val}'"
            else:
                default_str = str(default_val)
            
            # Formater l'annotation de type
            annotation = param.annotation
            if annotation == inspect.Parameter.empty:
                type_str = 'Any'
            else:
                type_str = str(annotation).replace('<class \'', '').replace('\'>', '')
                if len(type_str) > 20:
                    type_str = type_str[:20] + '...'
            
            params[name] = {
                'default': default_str,
                'annotation': type_str
            }
        
        return params
    except Exception:
        return {}


def show_custom_transformers(custom_transformers: Dict):
    """Affiche les transformateurs personnalisés découverts."""
    st.header("🔧 Transformateurs Personnalisés Découverts")
    
    if not custom_transformers:
        st.warning("🚫 Aucun transformateur personnalisé trouvé")
        if TRANSFORMERS_PATH:
            st.info(f"📁 Dossier scanné: `{TRANSFORMERS_PATH}`")
        
        # Suggestions d'aide
        with st.expander("💡 Suggestions", expanded=True):
            st.markdown("""
            **Vérifiez que:**
            - Le dossier `Transformateurs` existe
            - Il contient des fichiers `.py` avec des classes
            - Les classes héritent de `BaseEstimator` ou ont des méthodes `fit`/`transform`
            - Les fichiers ne contiennent pas d'erreurs de syntaxe
            """)
        return
    
    # Statistiques
    classes_count = len([t for t in custom_transformers.values() if t.get('type') == 'class'])
    functions_count = len([t for t in custom_transformers.values() if t.get('type') == 'function'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📦 Total", len(custom_transformers))
    with col2:
        st.metric("🏗️ Classes", classes_count)
    with col3:
        st.metric("⚙️ Fonctions", functions_count)
    
    # Grouper par module
    modules = {}
    for name, info in custom_transformers.items():
        module_name = info['module']
        if module_name not in modules:
            modules[module_name] = {'classes': [], 'functions': []}
        
        if info.get('type') == 'function':
            modules[module_name]['functions'].append((name, info))
        else:
            modules[module_name]['classes'].append((name, info))
    
    # Afficher par module
    for module_name, items in modules.items():
        total_items = len(items['classes']) + len(items['functions'])
        
        with st.expander(f"📁 **{module_name}** ({total_items} éléments)", expanded=False):
            # Afficher le nom du fichier
            if items['classes']:
                file_name = items['classes'][0][1]['file']
            elif items['functions']:
                file_name = items['functions'][0][1]['file']
            else:
                file_name = "Unknown"
            
            st.info(f"� Fichier: `{file_name}`")
            
            # Afficher les classes
            if items['classes']:
                st.subheader("🏗️ Classes")
                for name, info in items['classes']:
                    show_transformer_details(name, info)
            
            # Afficher les fonctions
            if items['functions']:
                st.subheader("⚙️ Fonctions utilitaires")
                for name, info in items['functions']:
                    show_transformer_details(name, info)


def show_transformer_details(name: str, info: Dict):
    """Affiche les détails d'un transformateur."""
    with st.container():
        # Nom et type
        icon = "🏗️" if info.get('type') == 'class' else "⚙️"
        st.markdown(f"### {icon} **{name}**")
        
        # Description
        if info['docstring'] and info['docstring'] != "Pas de documentation disponible":
            with st.expander("📖 Description", expanded=False):
                st.markdown(info['docstring'])
        
        # Paramètres
        if info['parameters']:
            with st.expander("⚙️ Paramètres", expanded=False):
                params_data = []
                for param_name, param_info in info['parameters'].items():
                    params_data.append({
                        'Paramètre': param_name,
                        'Type': param_info['annotation'],
                        'Défaut': param_info['default']
                    })
                
                if params_data:
                    params_df = pd.DataFrame(params_data)
                    st.dataframe(
                        params_df, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            'Paramètre': st.column_config.TextColumn('Paramètre', width='medium'),
                            'Type': st.column_config.TextColumn('Type', width='medium'),
                            'Défaut': st.column_config.TextColumn('Valeur par défaut', width='medium')
                        }
                    )
        
        # Méthodes (pour les classes)
        if info.get('methods') and info['type'] == 'class':
            with st.expander("🔧 Méthodes disponibles", expanded=False):
                methods = info['methods'][:8]  # Limiter à 8 méthodes
                cols = st.columns(min(4, len(methods)))
                for i, method in enumerate(methods):
                    with cols[i % 4]:
                        st.code(method, language='python')
        
        # Code d'exemple
        with st.expander("💻 Exemple d'utilisation", expanded=False):
            if info.get('type') == 'function':
                st.code(f"""
# Import de la fonction
from src.features.Pipelines.Transformateurs.{info['module']} import {name.replace('📋 ', '')}

# Utilisation
result = {name.replace('📋 ', '')}(...)
                """, language='python')
            else:
                st.code(f"""
# Import de la classe
from src.features.Pipelines.Transformateurs.{info['module']} import {name}

# Instanciation
transformer = {name}()

# Utilisation dans un pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('{name.lower()}', transformer),
    # ... autres étapes
])
                """, language='python')
        
        st.divider()


def show_sklearn_transformers(sklearn_transformers: Dict):
    """Affiche les transformateurs scikit-learn."""
    st.header("🧪 Transformateurs Scikit-Learn")
    
    # Grouper par catégorie
    categories = {}
    for name, info in sklearn_transformers.items():
        category = info['category']
        if category not in categories:
            categories[category] = []
        categories[category].append((name, info))
    
    # Afficher par catégorie
    for category, items in categories.items():
        with st.expander(f"📂 {category}", expanded=False):
            cols = st.columns(2)
            
            for i, (name, info) in enumerate(items):
                with cols[i % 2]:
                    st.subheader(f"🔸 {name}")
                    st.write(f"**Module:** `{info['module']}`")
                    st.write(f"**Description:** {info['description']}")
                    
                    # Code d'exemple
                    st.code(f"from {info['module']} import {name}", language='python')
                    st.write("")


def show_tensorflow_components(tf_components: Dict):
    """Affiche les composants TensorFlow/Keras."""
    st.header("🧠 Composants TensorFlow/Keras")
    
    # Grouper par catégorie
    categories = {}
    for name, info in tf_components.items():
        category = info['category']
        if category not in categories:
            categories[category] = []
        categories[category].append((name, info))
    
    # Afficher par catégorie
    for category, items in categories.items():
        with st.expander(f"🎯 {category}", expanded=False):
            cols = st.columns(2)
            
            for i, (name, info) in enumerate(items):
                with cols[i % 2]:
                    st.subheader(f"🔹 {name}")
                    st.write(f"**Module:** `{info['module']}`")
                    st.write(f"**Description:** {info['description']}")
                    
                    # Code d'exemple
                    st.code(f"from {info['module']} import {name}", language='python')
                    st.write("")


def show_scan_info():
    """Affiche les informations sur le scan."""
    st.header("🔍 Informations sur le Scan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Configuration des chemins")
        
        if TRANSFORMERS_PATH is None:
            st.error("❌ Chemin non configuré")
            return
            
        st.code(str(TRANSFORMERS_PATH), language='text')
        
        try:
            if TRANSFORMERS_PATH.exists():
                st.success("✅ Dossier accessible")
                
                # Lister les fichiers Python
                py_files = list(TRANSFORMERS_PATH.glob("*.py"))
                non_private_files = [f for f in py_files if not f.name.startswith("__")]
                
                st.metric("Fichiers Python", len(py_files))
                st.metric("Fichiers analysables", len(non_private_files))
                
                if non_private_files:
                    with st.expander("📄 Fichiers détectés", expanded=False):
                        for file in non_private_files:
                            # Vérifier la taille du fichier
                            try:
                                size = file.stat().st_size
                                size_str = f"({size} bytes)"
                            except:
                                size_str = ""
                            
                            st.write(f"- `{file.name}` {size_str}")
                
            else:
                st.error("❌ Dossier introuvable")
                
                # Suggestions de dépannage
                with st.expander("🔧 Dépannage", expanded=True):
                    st.markdown("""
                    **Vérifiez que:**
                    1. Le dossier `src/features/Pipelines/Transformateurs/` existe
                    2. Vous êtes dans le bon répertoire de travail
                    3. Les permissions d'accès sont correctes
                    """)
                    
                    # Vérifier le répertoire parent
                    parent = TRANSFORMERS_PATH.parent
                    if parent.exists():
                        st.info(f"✅ Dossier parent trouvé: `{parent}`")
                        subdirs = [d.name for d in parent.iterdir() if d.is_dir()]
                        st.write(f"Sous-dossiers disponibles: {subdirs}")
                    else:
                        st.error(f"❌ Dossier parent introuvable: `{parent}`")
                        
        except Exception as e:
            st.error(f"❌ Erreur lors de la vérification: {str(e)}")
            st.exception(e)
    
    with col2:
        st.subheader("⚙️ Critères de détection")
        
        st.markdown("""
        **Classes analysées:**
        - ✅ Avec méthodes `fit`, `transform`, `fit_transform`
        - ✅ Héritant de `BaseEstimator`, `TransformerMixin`
        - ✅ Héritant de `ClassifierMixin`, `RegressorMixin`
        - ✅ Avec méthode `predict`
        
        **Fonctions analysées:**
        - ✅ Contenant: `create`, `build`, `make`
        - ✅ Préfixées: `get_`, `load_`, `process_`
        - ✅ Non privées (ne commencent pas par `_`)
        
        **Fichiers exclus:**
        - ❌ Commençant par `__` (comme `__init__.py`)
        - ❌ `utilities.py` (traité séparément)
        """)
        
        # Informations sur le cache
        st.subheader("💾 Cache")
        st.info("Les résultats sont cachés pendant 5 minutes pour améliorer les performances")


def show_usage_guide():
    """Affiche un guide d'utilisation."""
    st.header("💡 Guide d'Utilisation")
    
    tab1, tab2, tab3 = st.tabs(["Transformateurs Personnalisés", "Sklearn", "TensorFlow/Keras"])
    
    with tab1:
        st.subheader("Utilisation des transformateurs personnalisés")
        st.code("""
# Import depuis le dossier Transformateurs
from src.features.Pipelines.Transformateurs import TensorFlowFeatureExtractor

# Utilisation dans un pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('feature_extractor', TensorFlowFeatureExtractor(model_name='VGG16')),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
        """, language='python')
    
    with tab2:
        st.subheader("Utilisation des transformateurs scikit-learn")
        st.code("""
# Import classique
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Pipeline avec transformateurs sklearn
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
        """, language='python')
    
    with tab3:
        st.subheader("Utilisation des composants TensorFlow/Keras")
        st.code("""
# Import TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

# Construction d'un modèle
base_model = VGG16(weights='imagenet', include_top=False)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        """, language='python')


def main():
    """Fonction principale de la page."""

    """
    st.set_page_config(
        page_title="Scanner Transformateurs - DS COVID",
        page_icon="🔍",
        layout="wide"
    )
    """
    # Header avec style
    st.title("🔍 Scanner de Transformateurs")
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;">
    <h4>🎯 Objectif</h4>
    Cette page scanne automatiquement le projet pour découvrir et cataloguer tous les transformateurs disponibles :
    <br>• <b>Transformateurs personnalisés</b> du projet COVID
    <br>• <b>Transformateurs Scikit-Learn</b> classiques  
    <br>• <b>Composants TensorFlow/Keras</b> pour le deep learning
    </div>
    """, unsafe_allow_html=True)
    
    # Contrôles
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🔄 Rafraîchir le scan", type="primary", help="Relance le scan des transformateurs"):
            st.cache_data.clear()  # Vider le cache
            st.rerun()
    
    with col2:
        show_details = st.checkbox("📋 Affichage détaillé", value=True)
    
    with col3:
        auto_expand = st.checkbox("📂 Auto-expansion", value=False)
    
    st.divider()
    
    # Tabs pour organiser le contenu
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔧 Transformateurs Personnalisés", 
        "🧪 Scikit-Learn", 
        "🧠 TensorFlow/Keras",
        "🔍 Infos Scan",
        "💡 Guide"
    ])
    
    # Tab 1: Transformateurs personnalisés
    with tab1:
        try:
            with st.spinner("🔍 Analyse du projet en cours..."):
                custom_transformers = scan_custom_transformers()
            
            show_custom_transformers(custom_transformers)
            
        except Exception as e:
            st.error(f"❌ Erreur lors du scan des transformateurs personnalisés: {e}")
            with st.expander("🔧 Détails de l'erreur", expanded=False):
                st.exception(e)
    
    # Tab 2: Scikit-Learn
    with tab2:
        try:
            sklearn_transformers = get_sklearn_transformers()
            show_sklearn_transformers(sklearn_transformers)
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des transformateurs Scikit-Learn: {e}")
    
    # Tab 3: TensorFlow/Keras
    with tab3:
        try:
            tf_components = get_tensorflow_keras_components()
            show_tensorflow_components(tf_components)
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des composants TensorFlow: {e}")
    
    # Tab 4: Informations sur le scan
    with tab4:
        show_scan_info()
    
    # Tab 5: Guide d'utilisation
    with tab5:
        show_usage_guide()
    
    # Footer avec statistiques
    st.divider()
    show_footer_stats()


def show_footer_stats():
    """Affiche les statistiques finales."""
    st.header("📊 Statistiques Globales")
    
    try:
        # Obtenir les données avec gestion d'erreur
        try:
            custom_transformers = scan_custom_transformers()
        except:
            custom_transformers = {}
        
        sklearn_transformers = get_sklearn_transformers()
        tf_components = get_tensorflow_keras_components()
        
        # Calculer les statistiques
        custom_classes = len([t for t in custom_transformers.values() if t.get('type') == 'class'])
        custom_functions = len([t for t in custom_transformers.values() if t.get('type') == 'function'])
        
        # Affichage en colonnes
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🔧 Transformateurs Personnalisés", 
                len(custom_transformers),
                help="Classes et fonctions découvertes dans le projet"
            )
            st.caption(f"📦 {custom_classes} classes • ⚙️ {custom_functions} fonctions")
        
        with col2:
            st.metric(
                "🧪 Scikit-Learn", 
                len(sklearn_transformers),
                help="Transformateurs de la librairie scikit-learn"
            )
        
        with col3:
            st.metric(
                "🧠 TensorFlow/Keras", 
                len(tf_components),
                help="Composants de deep learning"
            )
        
        with col4:
            total = len(custom_transformers) + len(sklearn_transformers) + len(tf_components)
            st.metric(
                "📈 Total Global", 
                total,
                help="Total de tous les composants disponibles"
            )
        
        # Graphique de répartition si suffisamment de données
        if total > 0:
            with st.expander("📊 Répartition détaillée", expanded=False):
                chart_data = pd.DataFrame({
                    'Catégorie': [
                        'Transformateurs\nPersonnalisés', 
                        'Scikit-Learn', 
                        'TensorFlow/Keras'
                    ],
                    'Nombre': [
                        len(custom_transformers), 
                        len(sklearn_transformers), 
                        len(tf_components)
                    ]
                })
                
                st.bar_chart(chart_data.set_index('Catégorie'))
    
    except Exception as e:
        st.error(f"❌ Erreur lors du calcul des statistiques: {e}")


main()

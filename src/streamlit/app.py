import streamlit as st
import sys
import os
import inspect
from pathlib import Path


# Ajout du répertoire racine du projet au PYTHONPATH pour tous les imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configuration de la page AVANT toute autre chose
st.set_page_config(
    page_title="Détection COVID-19",
    page_icon="🦠",
    layout="wide",
)

# Fonction pour découvrir automatiquement les pages
def discover_pages():
    """Découvre automatiquement toutes les pages dans la structure de dossiers"""
    pages_dir = Path(__file__).parent / "pages"
    pages_structure = {}
    
    for category_dir in sorted(pages_dir.iterdir()):
        if category_dir.is_dir() and not category_dir.name.startswith('__'):
            # Extraire le nom de la catégorie (enlever le préfixe numérique)
            category_name = category_dir.name.split('_', 1)[1] if '_' in category_dir.name else category_dir.name
            
            # Icônes par catégorie
            icons = {
                'Data': '📊',
                'Model': '🤖', 
                'Results': '🎯'
            }
            
            category_icon = icons.get(category_name, '📁')
            category_title = f"{category_icon} {category_name}"
            
            # Découvrir les pages dans cette catégorie
            category_pages = []
            for page_file in sorted(category_dir.glob("*.py")):
                if not page_file.name.startswith('__'):
                    # Extraire le titre de la page
                    page_name = page_file.stem.split('_', 1)[1] if '_' in page_file.stem else page_file.stem
                    
                    # Icônes par page
                    page_icons = {
                        'Vérification': '✅',
                        'Chargement': '📁',
                        'Exploration': '🔍',
                        'Training': '🏋️',
                        'Evaluation': '📈',
                        'Predictions': '🎯'
                    }
                    
                    # Utiliser l'icône définie ou une icône par défaut
                    page_icon = page_icons.get(page_name, '📄')
                    
                    category_pages.append({
                        'title': page_name,
                        'icon': page_icon,
                        'path': str(page_file.relative_to(Path(__file__).parent))
                    })
            
            if category_pages:
                pages_structure[category_title] = category_pages
    
    return pages_structure

# Découvrir et organiser les pages
pages_structure = discover_pages()

# Créer la navigation personnalisée avec sections pliables
if pages_structure:
    # Ajouter une page d'accueil simple
    def home_page():
        #st.title("🦠 Détection COVID-19")

        # afficher le contenu du README du projet
        readme_path = Path(__file__).parent.parent.parent / "README.md"
        if readme_path.exists():
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.read()
                st.markdown(readme_content)
        else:
            st.error(f"❌ Fichier README non trouvé : {readme_path}")
    # Navigation personnalisée avec sidebar
    with st.sidebar:
        st.title("🦠 Navigation")
        
        # Page d'accueil toujours visible
        if st.button("🏠 Accueil", use_container_width=True, key="home_nav"):
            st.session_state.current_page = "home"
        
        # Sections pliables pour chaque catégorie
        for category_title, category_pages in pages_structure.items():
            with st.expander(category_title, expanded=False):
                for page in category_pages:
                    page_key = f"nav_{page['title']}"
                    if st.button(f"{page['icon']} {page['title']}", use_container_width=True, key=page_key):
                        st.session_state.current_page = page['path']
    
    # Gestion de l'état de la page courante
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "home"
    
    # Affichage de la page courante
    if st.session_state.current_page == "home":
        home_page()
    else:
        # Exécuter la page sélectionnée
        page_path = st.session_state.current_page
        try:
            # Si c'est un chemin relatif, construire le chemin complet
            if isinstance(page_path, str):
                full_path = Path(__file__).parent / page_path
                if full_path.exists():
                    # Exécuter le fichier Python
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("page_module", full_path)
                    page_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(page_module)
                else:
                    st.error(f"Page non trouvée : {page_path}")
            else:
                # Si c'est une fonction de page, l'exécuter directement
                page_path()
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement de la page : {e}")
            st.info("🔧 Retour à l'accueil...")
            st.session_state.current_page = "home"
            st.rerun()
else:
    st.error("Aucune page n'a été trouvée dans la structure de dossiers.")
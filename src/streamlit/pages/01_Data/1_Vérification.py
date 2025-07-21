import streamlit as st
import os
import sys
from pathlib import Path


# ====================================================================
# 🔧 BACKEND - LOGIQUE MÉTIER
# ====================================================================

def check_project_directory():
    """Vérifie l'existence du répertoire du projet."""
    project_dir = Path(__file__).parent.parent.parent.parent.parent
    return project_dir if project_dir.exists() else None

def check_data_directory(project_dir):
    """Vérifie l'existence du dossier des données."""
    if not project_dir:
        return None
    data_dir = project_dir / "data/covid19-radiography-database/COVID-19_Radiography_Dataset"
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

# ====================================================================
# 🎨 FRONTEND - INTERFACE UTILISATEUR
# ====================================================================
# ====================================================================
# 🎨 FRONTEND - INTERFACE UTILISATEUR
# ====================================================================

def show_global_status(results):
    """Affiche l'indicateur de statut global en haut de page."""
    if results["all_checks_passed"]:
        st.success("🎉 **TOUTES LES VÉRIFICATIONS SONT AU VERT !** Vous pouvez procéder à l'étape suivante.", icon="✅")
    else:
        st.error("⚠️ **CERTAINES VÉRIFICATIONS ONT ÉCHOUÉ.** Veuillez corriger les erreurs avant de continuer.", icon="❌")

def show_infrastructure_section(results):
    """Affiche la section infrastructure."""
    with st.expander("🏗️ Infrastructure", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if results["project_dir"]:
                st.success("✅ Projet")
                st.code(str(results["project_dir"]), language="bash")
            else:
                st.error("❌ Projet non trouvé")
        
        with col2:
            if results["data_dir"]:
                st.success("✅ Données")
                st.code(str(results["data_dir"]), language="bash")
            else:
                st.error("❌ Dossier données manquant")
        
        with col3:
            if results["python_ok"]:
                st.success(f"✅ Python {results['python_version']}")
            else:
                st.error(f"❌ Python {results['python_version']}")
                st.info("💡 Utilisez Python 3.12")

def show_data_content_section(results):
    """Affiche la section contenu des données."""
    with st.expander("� Contenu des données", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 Dossiers", len(results["subdirs"]))
        with col2:
            st.metric("📄 Fichiers", len(results["files"]))
        with col3:
            st.metric("📊 Total", len(results["subdirs"]) + len(results["files"]))
        
        if results["subdirs"]:
            st.markdown("**Dossiers trouvés:**")
            for subdir in results["subdirs"]:
                st.markdown(f"• `{subdir.name}`")

def show_categories_section(results):
    """Affiche la section validation des catégories."""
    with st.expander("🏷️ Catégories", expanded=True):
        if results["categories_ok"]:
            st.success("✅ Toutes les catégories attendues sont présentes")
            for cat in results["found_categories"]:
                st.markdown(f"• ✓ `{cat}`")
        else:
            st.error("❌ Certaines catégories manquent")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Trouvées:**")
                for cat in results["found_categories"]:
                    st.markdown(f"• ✓ `{cat}`")
            with col2:
                st.markdown("**Manquantes:**")
                for cat in results["missing_categories"]:
                    st.markdown(f"• ❌ `{cat}`")

def show_structure_section(results):
    """Affiche la section validation de structure."""
    with st.expander("🗂️ Structure", expanded=True):
        if results["structure_ok"]:
            st.success("✅ Toutes les structures sont complètes")
        else:
            st.warning("⚠️ Certaines structures sont incomplètes")
        
        for result in results["structure_results"]:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.markdown(f"**📁 {result['name']}**")
            with col2:
                if result["images_ok"]:
                    st.success(f"Images: {result['images_count']}")
                else:
                    st.error("Images: ❌")
            with col3:
                if result["masks_ok"]:
                    st.success(f"Masks: {result['masks_count']}")
                else:
                    st.error("Masks: ❌")
            with col4:
                if result["complete"]:
                    st.success("Complet")
                else:
                    st.error("Incomplet")

def show_metadata_section(results):
    """Affiche la section validation des métadonnées."""
    with st.expander("📋 Métadonnées", expanded=True):
        if results["metadata_ok"]:
            st.success("✅ Tous les fichiers de métadonnées sont présents")
        else:
            st.error("❌ Certains fichiers de métadonnées manquent")
        
        for result in results["metadata_results"]:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**📄 {result['name']}**")
            with col2:
                if result["exists"]:
                    st.success("Présent")
                else:
                    st.error("Manquant")
            with col3:
                if result["exists"]:
                    st.write(f"{result['size_kb']:.1f} KB")
                else:
                    st.write("-")

def show_sidebar_help():
    """Affiche l'aide dans la sidebar."""
    with st.sidebar:
        st.markdown("## 📋 **Workflow**")
        st.markdown("""
        ### 🏗️ Infrastructure
        - Répertoire projet
        - Dossier données
        - Version Python
        
        ### � Validation
        - Contenu des données
        - Catégories attendues
        - Structure dossiers
        - Fichiers métadonnées
        """)
        
        st.markdown("---")
        st.markdown("### 💡 **Aide**")
        st.info("""
        **En cas d'erreur:**
        - Vérifiez les chemins
        - Téléchargez le dataset complet
        - Utilisez Python 3.12
        """)

def main():
    """Fonction principale."""
    st.markdown("# 🔍 Vérification des Données COVID-19")
    st.markdown("---")
    
    # Sidebar
    show_sidebar_help()
    
    # Exécution des vérifications
    with st.spinner("� Exécution des vérifications..."):
        results = run_all_checks()
    
    # Stockage des résultats dans session_state pour les autres pages
    st.session_state["verification_results"] = results
    
    # Affichage du statut global
    show_global_status(results)
    st.markdown("---")
    
    # Affichage des sections
    show_infrastructure_section(results)
    show_data_content_section(results)
    show_categories_section(results)
    show_structure_section(results)
    show_metadata_section(results)
    
    # Message final
    st.markdown("---")
    if results["all_checks_passed"]:
        st.info("💡 **Prochaine étape:** Vous pouvez maintenant procéder au chargement des données.")
    else:
        st.warning("⚠️ **Action requise:** Corrigez les erreurs avant de continuer.")

if __name__ == "__main__":
    main()
else:
    # Exécution automatique quand importé
    main()

import streamlit as st
from pathlib import Path

from src.features.Verifs_Env.Vérifications_Back import *
from src.features.Widget_Streamlit.W_Vérifications_Front import *




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



def main():
    """Fonction principale."""
    st.markdown("# 🔍 Vérification des Données COVID-19")
    st.markdown("---")
    
    
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

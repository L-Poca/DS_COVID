"""
Frontend Streamlit pour l'inspection des features
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from src.features.Inspector.Features_Core import get_features_files, analyze_module_functions, get_all_functions_summary


def show_features_files():
    """Affiche la liste des fichiers dans le dossier Features avec informations détaillées."""
    st.subheader("📁 Fichiers dans le dossier Features:")
    
    files, features_dir, file_info = get_features_files()
    
    # Afficher le chemin du dossier
    st.info(f"📂 Dossier analysé: `{features_dir}`")
    
    if files:
        st.success(f"✅ {len(files)} fichier(s) Python trouvé(s)")
        
        # Créer un tableau avec les informations détaillées
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


def show_function_details(func_info, use_expander=True):
    """
    Affiche les détails d'une fonction de manière organisée.
    
    Args:
        func_info (dict): Informations sur la fonction
        use_expander (bool): Utiliser un expander ou afficher directement
    """
    if use_expander:
        with st.expander(f"🔧 {func_info['name']}", expanded=False):
            _render_function_content(func_info)
    else:
        st.markdown(f"### 🔧 {func_info['name']}")
        _render_function_content(func_info)


def _render_function_content(func_info):
    """
    Rend le contenu des détails d'une fonction (utilisé par show_function_details).
    
    Args:
        func_info (dict): Informations sur la fonction
    """
    # Signature
    st.code(f"def {func_info['name']}{func_info['signature']}", language="python")
    
    # Documentation centrée
    st.markdown("**📖 Documentation:**")
    if func_info['doc'] and func_info['doc'] != "Pas de documentation":
        # Centrer la documentation avec du CSS
        st.markdown(
            f"""
            <div style="
                text-align: center; 
                padding: 15px; 
                background-color: #f0f2f6; 
                border-radius: 8px; 
                margin: 10px 0;
                border-left: 4px solid #1f77b4;
            ">
                <em>{func_info['doc']}</em>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="
                text-align: center; 
                padding: 15px; 
                background-color: #ffe6e6; 
                border-radius: 8px; 
                margin: 10px 0;
                border-left: 4px solid #ff6b6b;
            ">
                <em>⚠️ Pas de documentation disponible</em>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
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
    
    # Code source - utiliser un container plutôt qu'un expander imbriqué
    if st.button(f"📋 Voir le code source de {func_info['name']}", key=f"source_{func_info['name']}_{func_info['file']}"):
        st.code(func_info['source'], language="python")


def show_features_functions_analysis():
    """
    Affiche l'analyse complète des fonctions pour tous les fichiers features.
    """
    st.header("🔍 Analyse des Fonctions Features")
    
    try:
        files, features_dir, file_info = get_features_files()
        
        if not files:
            st.warning("⚠️ Aucun fichier à analyser")
            st.info("💡 Assurez-vous que le dossier `src/features/` contient des fichiers Python")
            return
        
        st.success(f"✅ {len(files)} fichier(s) trouvé(s) à analyser")
        
        # Sélecteur de fichier
        selected_file = st.selectbox(
            "📁 Choisir un fichier à analyser:",
            options=[f.name for f in files],
            help="Sélectionnez un fichier pour voir ses fonctions en détail"
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
                
                # Analyser les fonctions avec gestion d'erreur
                try:
                    with st.spinner("🔍 Analyse en cours..."):
                        functions_info = analyze_module_functions(file_path)
                    
                    if functions_info:
                        # Statistiques générales
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🔧 Fonctions", len(functions_info))
                        with col2:
                            total_lines = sum(f.get('source_lines', 0) for f in functions_info)
                            st.metric("📏 Lignes totales", total_lines)
                        with col3:
                            documented = sum(1 for f in functions_info if f.get('doc', '') != "Pas de documentation")
                            st.metric("📖 Documentées", f"{documented}/{len(functions_info)}")
                        
                        st.markdown("---")
                        
                        # Filtrage optionnel
                        filter_option = st.radio(
                            "🔍 Filtrer les fonctions:",
                            ["Toutes", "Documentées uniquement", "Non documentées uniquement"],
                            horizontal=True
                        )
                        
                        # Appliquer le filtre
                        filtered_functions = functions_info
                        if filter_option == "Documentées uniquement":
                            filtered_functions = [f for f in functions_info if f.get('doc', '') != "Pas de documentation"]
                        elif filter_option == "Non documentées uniquement":
                            filtered_functions = [f for f in functions_info if f.get('doc', '') == "Pas de documentation"]
                        
                        if filtered_functions:
                            st.info(f"📊 Affichage de {len(filtered_functions)}/{len(functions_info)} fonction(s)")
                            
                            # Choix du mode d'affichage
                            display_mode = st.radio(
                                "📋 Mode d'affichage:",
                                ["Liste compacte", "Détails complets"],
                                horizontal=True
                            )
                            
                            if display_mode == "Liste compacte":
                                # Affichage compact avec sélection
                                for i, func_info in enumerate(filtered_functions):
                                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                                    with col1:
                                        st.write(f"**🔧 {func_info['name']}**")
                                    with col2:
                                        st.write(f"📏 {func_info.get('source_lines', 0)} lignes")
                                    with col3:
                                        doc_status = "✅" if func_info.get('doc', '') != "Pas de documentation" else "❌"
                                        st.write(f"📖 {doc_status}")
                                    with col4:
                                        if st.button("👁️ Détails", key=f"details_{i}_{func_info['name']}"):
                                            st.session_state[f"show_details_{func_info['name']}"] = True
                                    
                                    # Afficher les détails si demandé
                                    if st.session_state.get(f"show_details_{func_info['name']}", False):
                                        with st.container():
                                            show_function_details(func_info, use_expander=False)
                                        if st.button("🔼 Masquer", key=f"hide_{i}_{func_info['name']}"):
                                            st.session_state[f"show_details_{func_info['name']}"] = False
                                            st.rerun()
                                        st.markdown("---")
                            else:
                                # Affichage détaillé avec expanders
                                for func_info in filtered_functions:
                                    show_function_details(func_info, use_expander=True)
                        else:
                            st.warning(f"⚠️ Aucune fonction ne correspond au filtre '{filter_option}'")
                            
                    else:
                        st.info("ℹ️ Aucune fonction trouvée dans ce fichier")
                        st.write("Cela peut arriver si:")
                        st.write("- Le fichier ne contient que des imports")
                        st.write("- Le fichier contient des erreurs de syntaxe")
                        st.write("- Les fonctions sont importées d'autres modules")
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse de {selected_file}: {str(e)}")
                    with st.expander("🔍 Détails de l'erreur"):
                        st.code(str(e))
        
        # Séparateur
        st.markdown("---")
        
        # Option pour voir toutes les fonctions de tous les fichiers
        if st.checkbox("🌍 Afficher toutes les fonctions de tous les fichiers"):
            st.markdown("## 🌍 Vue d'ensemble de toutes les fonctions")
            
            try:
                with st.spinner("🔍 Analyse de tous les fichiers..."):
                    summary = get_all_functions_summary()
                
                if summary and summary.get('all_functions'):
                    # Tableau récapitulatif
                    summary_data = []
                    for func in summary['all_functions']:
                        summary_data.append({
                            '🔧 Fonction': func.get('name', 'N/A'),
                            '📄 Fichier': func.get('file', 'N/A'),
                            '📏 Lignes': func.get('source_lines', 0),
                            '📖 Documentée': '✅' if func.get('doc', '') != "Pas de documentation" else '❌',
                            '📝 Paramètres': len(func.get('parameters', []))
                        })
                    
                    if summary_data:
                        df = pd.DataFrame(summary_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Statistiques globales
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🔧 Total fonctions", summary.get('total_functions', 0))
                        with col2:
                            st.metric("📄 Fichiers", summary.get('total_files', 0))
                        with col3:
                            st.metric("📏 Lignes totales", summary.get('total_lines', 0))
                        with col4:
                            st.metric("📖 Documentation", f"{summary.get('documented_functions', 0)}/{summary.get('total_functions', 0)}")
                        
                        # Graphique de répartition par fichier
                        if len(summary.get('functions_by_file', {})) > 1:
                            st.subheader("📊 Répartition des fonctions par fichier")
                            
                            chart_data = []
                            for file_name, functions in summary['functions_by_file'].items():
                                chart_data.append({
                                    'Fichier': file_name,
                                    'Nombre de fonctions': len(functions)
                                })
                            
                            if chart_data:
                                chart_df = pd.DataFrame(chart_data)
                                st.bar_chart(chart_df.set_index('Fichier'))
                    else:
                        st.warning("⚠️ Aucune donnée à afficher")
                else:
                    st.info("ℹ️ Aucune fonction trouvée dans l'ensemble des fichiers")
                    
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse globale: {str(e)}")
                with st.expander("🔍 Détails de l'erreur"):
                    st.code(str(e))
                    
    except Exception as e:
        st.error(f"❌ Erreur lors de l'initialisation: {str(e)}")
        st.info("💡 Vérifiez que le module Features_Core est correctement configuré")

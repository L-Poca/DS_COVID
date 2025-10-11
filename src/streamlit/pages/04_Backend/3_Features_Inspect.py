import streamlit as st
import pandas as pd
from src.features.Inspector.Features_Core import *
from src.features.Verifs_Env.Vérifications_Back import *
from src.features.Widget_Streamlit.W_Vérifications_Front import *

show_global_status(run_all_checks())

#st.set_page_config(page_title="Inspection des Features", layout="wide")
st.title("🔬 Inspection des Fonctions Features")

# Chargement des fichiers et fonctions
@st.cache_data
def load_features():
    files, features_dir, file_info = get_features_files()
    all_functions = []
    for file_path in files:
        all_functions.extend(analyze_module_functions(file_path))
    return files, features_dir, file_info, all_functions

files, features_dir, file_info, all_functions = load_features()

# Sidebar - Filtres
with st.sidebar:
    st.header("Filtres")
    show_docs_only = st.checkbox("Afficher uniquement les fonctions documentées", value=False)
    min_lines = st.number_input("Nombre minimal de lignes de code", min_value=0, value=0)
    selected_file = st.selectbox("Fichier", options=["Tous"] + [f.name for f in files])
    search = st.text_input("Recherche (nom ou docstring)")

# Application des filtres
filtered = all_functions
if show_docs_only:
    filtered = [f for f in filtered if f['doc'] and f['doc'] != "Pas de documentation"]
if min_lines > 0:
    filtered = [f for f in filtered if f['source_lines'] >= min_lines]
if selected_file != "Tous":
    filtered = [f for f in filtered if f['file'] == selected_file]
if search:
    filtered = [f for f in filtered if search.lower() in f['name'].lower() or search.lower() in (f['doc'] or "").lower()]

st.markdown(f"**{len(filtered)} fonction(s) trouvée(s) / {len(all_functions)}**")

# Présentation sous forme de tableau
if filtered:
    df = pd.DataFrame([
        {
            "Fonction": f["name"],
            "Dossier": f["folder"],
            "Fichier": f["file"],
            "Lignes": f["source_lines"],
            "Paramètres": len(f["parameters"]),
            "Retour": ", ".join(f.get("return_values", ["Non détecté"])) if f.get("return_values") else "Non détecté",
            "Doc": "✅" if f["doc"] and f["doc"] != "Pas de documentation" else "❌",
            "Description": (f["doc"][:80] + "...") if f["doc"] and len(f["doc"]) > 80 else (f["doc"] or "Pas de documentation")
        }
        for f in filtered
    ])
    st.dataframe(df, use_container_width=True)

    # Sélection d'une fonction pour détails
    idx = st.number_input("Index de la fonction à détailler", min_value=0, max_value=len(filtered)-1, value=0, step=1)
    func = filtered[idx]
    st.markdown("---")
    st.subheader(f"Détails de la fonction : {func['name']}")
    st.code(f"def {func['name']}{func['signature']}", language="python")
    st.markdown(f"**Fichier :** `{func['file']}`  ")
    st.markdown(f"**Lignes de code :** {func['source_lines']}")
    st.markdown(f"**Paramètres :** {len(func['parameters'])}")
    if func.get('return_values'):
        st.markdown(f"**Valeurs de retour :** {', '.join(func['return_values'])}")
    else:
        st.markdown(f"**Valeurs de retour :** Non détecté")
    if func['doc'] and func['doc'] != "Pas de documentation":
        st.info(func['doc'])
    else:
        st.warning("Pas de documentation disponible")
    if func['parameters']:
        st.markdown("**Paramètres détaillés :**")
        for p in func['parameters']:
            st.write(f"- `{p['name']}` : {p.get('annotation', 'Type non spécifié')} (Défaut : {p.get('default', 'Requis')})")
    #if st.toggle("Afficher le code source"):
    st.code(func['source'], language="python")
else:
    st.warning("Aucune fonction trouvée avec ces critères.")

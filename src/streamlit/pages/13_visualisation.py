import os
import streamlit as st
from PIL import Image
from collections import Counter
import pandas as pd
import plotly.graph_objects as go

# ------------------ 📁 Dossier cible à analyser
dossier_images = os.path.abspath(
    os.path.join(__file__, "..", "..", "..", "..", "data", "raw",
                 "COVID-19_Radiography_Dataset",
                 "COVID-19_Radiography_Dataset",
                 "Viral Pneumonia", "images")
)


extensions = (".png", ".jpg", ".jpeg")

# ------------------ 🧠 UI de contexte
st.title("🔍 Analyse des modes d’image")
st.markdown("Cette page détecte si certaines radiographies sont mal encodées en `RGB` au lieu de `L` (niveaux de gris).")
# st.write("📂 Dossier analysé :", dossier_images)

# ------------------ 🔍 Analyse des images
modes = Counter()
rgb_files = []

for root, _, files in os.walk(dossier_images):
    for file in files:
        if file.lower().endswith(extensions):
            try:
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    modes[img.mode] += 1
                    if img.mode == "RGB":
                        rgb_files.append(img_path)
            except Exception as e:
                st.warning(f"❌ Erreur sur {file} : {e}")

# ------------------ 📊 DataFrame pour Plotly
df_modes = pd.DataFrame(modes.items(), columns=["Mode", "Nombre"])

# Débogage
# st.write("📂 Chemin utilisé :", os.path.abspath(dossier_images))
# st.write("📸 Exemples de fichiers :")

# Scan rapide
# extraits = []
# for root, _, files in os.walk(dossier_images):
# for file in files:
# # if file.lower().endswith(('.png', '.jpg', '.jpeg')):
# extraits.append(file)
# if extraits:
# break

# st.write(extraits[:5] or "Aucun fichier trouvé")


if df_modes.empty:
    st.error("⚠️ Aucune image détectée dans le dossier. Vérifie le chemin ou les extensions.")
else:
    nb_l = modes.get("L", 0)
    nb_rgb = modes.get("RGB", 0)
    total = nb_l + nb_rgb

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=nb_l,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Images en niveaux de gris (L)"},
        gauge={
            'axis': {'range': [0, total]},
            'bar': {'color': "limegreen"},
            'steps': [
                {'range': [0, total * 0.5], 'color': "lightgray"},
                {'range': [total * 0.5, total], 'color': "gray"}
            ],
        }
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig)

    st.markdown(f"⚠️ **{nb_rgb} images RGB détectées (hors norme)** sur un total de {total} images.")

    # ------------------ 🧠 Analyse métier
    st.markdown("### 🧠 Analyse métier")
    st.markdown("""
- Les radiographies doivent être en **niveau de gris** (`L`), car les informations pertinentes ne sont pas dans les couleurs.
- La présence de plusieurs images en `RGB` est donc **anormale**.
- Cela peut indiquer des erreurs de traitement ou d'export depuis un outil d'annotation.
""")

    # ------------------ 🛠️ Recommandations
    st.markdown("### 🛠️ Recommandations techniques")
    st.markdown("""
- Identifier les images en `RGB` et les **vérifier visuellement**.
- Si ce sont bien des radios, les **convertir en `L`** via `img.convert("L")` pour :
    - homogénéiser les données
    - réduire la taille mémoire
- Ces anomalies sont visibles dans le **rapport template d’exploration automatique** généré en amont.
""")

    # ------------------ 📁 Liste des fichiers RGB
@st.cache_resource
def load_images(filepaths, max_images=3):
    images = []
    for path in filepaths[:max_images]:
        img = Image.open(path)
        images.append(img)
    return images

if rgb_files:
    st.markdown("### 🖼️ Aperçu des fichiers RGB détectés (extrait)")

    images = load_images(rgb_files, max_images=3)

    cols = st.columns(len(images))
    for col, img in zip(cols, images):
        col.image(img, width=200)

    total_images = sum(modes.values())
    st.info(f"{len(rgb_files)} images RGB détectées sur {total_images} "
            f"({len(rgb_files)/total_images*100:.2f}%)")
else:
    st.success("✅ Aucune image en RGB détectée !")

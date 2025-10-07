import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io

st.header("📁 T_06 - Gestion des Fichiers (Version Simple)")

st.markdown("**📋 Objectif :** Apprendre à gérer les fichiers - upload, download, images, CSV - les essentiels pour échanger des données.")

st.markdown("---")

# ================================
# 1. UPLOADER UN FICHIER
# ================================
st.subheader("1️⃣ Uploader un fichier")

st.markdown("""
**📖 Explication simple :**
Permettre à l'utilisateur d'envoyer un fichier depuis son ordinateur.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Upload simple")
    st.code("""
# Zone d'upload pour n'importe quel fichier
fichier = st.file_uploader("Choisissez un fichier")

if fichier is not None:
    # Afficher les informations du fichier
    st.write("**Nom du fichier:**", fichier.name)
    st.write("**Taille:**", len(fichier.getvalue()), "bytes")
    st.write("**Type:**", fichier.type)
    
    st.success("Fichier uploadé avec succès !")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Zone d'upload
    fichier = st.file_uploader("Choisissez un fichier", key="demo_upload_1")
    
    if fichier is not None:
        # Afficher les informations du fichier
        st.write("**Nom du fichier:**", fichier.name)
        st.write("**Taille:**", len(fichier.getvalue()), "bytes")
        st.write("**Type:**", fichier.type)
        
        st.success("Fichier uploadé avec succès !")

st.divider()

# ================================
# 2. UPLOADER UNE IMAGE
# ================================
st.subheader("2️⃣ Uploader et afficher une image")

st.markdown("""
**📖 Explication simple :**
Spécialement pour les images - on peut les afficher directement.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Upload image")
    st.code("""
# Upload spécifique pour images
image_file = st.file_uploader(
    "Choisissez une image", 
    type=['png', 'jpg', 'jpeg']
)

if image_file is not None:
    # Ouvrir et afficher l'image
    from PIL import Image
    image = Image.open(image_file)
    
    st.image(image, caption="Votre image uploadée")
    
    # Informations sur l'image
    st.write(f"Dimensions: {image.size}")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Upload d'image
    image_file = st.file_uploader(
        "Choisissez une image", 
        type=['png', 'jpg', 'jpeg'],
        key="demo_upload_image"
    )
    
    if image_file is not None:
        # Ouvrir et afficher l'image
        image = Image.open(image_file)
        
        st.image(image, caption="Votre image uploadée", width=300)
        
        # Informations sur l'image
        st.write(f"Dimensions: {image.size}")

st.divider()

# ================================
# 3. UPLOADER UN CSV
# ================================
st.subheader("3️⃣ Uploader et lire un fichier CSV")

st.markdown("""
**📖 Explication simple :**
CSV (tableur) est très courant pour les données. On peut le lire et l'afficher.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Upload CSV")
    st.code("""
# Upload spécifique pour CSV
csv_file = st.file_uploader(
    "Choisissez un fichier CSV", 
    type=['csv']
)

if csv_file is not None:
    # Lire le CSV
    df = pd.read_csv(csv_file)
    
    # Afficher les informations
    st.write(f"Nombre de lignes: {len(df)}")
    st.write(f"Nombre de colonnes: {len(df.columns)}")
    st.write("Colonnes:", list(df.columns))
    
    # Afficher le tableau
    st.dataframe(df)
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Upload de CSV
    csv_file = st.file_uploader(
        "Choisissez un fichier CSV", 
        type=['csv'],
        key="demo_upload_csv"
    )
    
    if csv_file is not None:
        try:
            # Lire le CSV
            df = pd.read_csv(csv_file)
            
            # Afficher les informations
            st.write(f"Nombre de lignes: {len(df)}")
            st.write(f"Nombre de colonnes: {len(df.columns)}")
            st.write("Colonnes:", list(df.columns))
            
            # Afficher le tableau
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors de la lecture: {e}")

st.divider()

# ================================
# 4. TÉLÉCHARGER UN FICHIER
# ================================
st.subheader("4️⃣ Télécharger un fichier")

st.markdown("""
**📖 Explication simple :**
Permettre à l'utilisateur de télécharger des données générées par l'application.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Download simple")
    st.code("""
# Créer des données à télécharger
data = {
    'Nom': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Ville': ['Paris', 'Lyon', 'Marseille']
}
df = pd.DataFrame(data)

# Convertir en CSV
csv = df.to_csv(index=False)

# Bouton de téléchargement
st.download_button(
    label="📥 Télécharger les données CSV",
    data=csv,
    file_name='mes_donnees.csv',
    mime='text/csv'
)
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Créer des données à télécharger
    data_download = {
        'Nom': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Ville': ['Paris', 'Lyon', 'Marseille']
    }
    df_download = pd.DataFrame(data_download)
    
    # Afficher les données
    st.dataframe(df_download)
    
    # Convertir en CSV
    csv = df_download.to_csv(index=False)
    
    # Bouton de téléchargement
    st.download_button(
        label="📥 Télécharger les données CSV",
        data=csv,
        file_name='mes_donnees.csv',
        mime='text/csv',
        key="demo_download_csv"
    )

st.divider()

# ================================
# 5. TÉLÉCHARGER DU TEXTE
# ================================
st.subheader("5️⃣ Télécharger du texte")

st.markdown("""
**📖 Explication simple :**
On peut aussi faire télécharger du texte simple, des rapports, etc.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Download texte")
    st.code("""
# Zone de texte pour l'utilisateur
texte_utilisateur = st.text_area(
    "Écrivez votre texte:",
    "Bonjour, ceci est un exemple de texte."
)

# Bouton pour télécharger le texte
st.download_button(
    label="📄 Télécharger le texte",
    data=texte_utilisateur,
    file_name='mon_texte.txt',
    mime='text/plain'
)
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Zone de texte pour l'utilisateur
    texte_utilisateur = st.text_area(
        "Écrivez votre texte:",
        "Bonjour, ceci est un exemple de texte.",
        key="demo_text_area"
    )
    
    # Bouton pour télécharger le texte
    st.download_button(
        label="📄 Télécharger le texte",
        data=texte_utilisateur,
        file_name='mon_texte.txt',
        mime='text/plain',
        key="demo_download_text"
    )

st.divider()

# ================================
# 6. AFFICHER DES IMAGES STATIQUES
# ================================
st.subheader("6️⃣ Afficher des images fixes")

st.markdown("""
**📖 Explication simple :**
Afficher des images qui font partie de votre application (logos, exemples, etc.).
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Image depuis URL")
    st.code("""
# Image depuis une URL
st.image(
    "https://via.placeholder.com/300x200/blue/white?text=Exemple",
    caption="Image d'exemple depuis internet"
)

# Ou depuis un fichier local (si vous en avez)
# st.image("chemin/vers/image.jpg", caption="Mon image")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Image depuis une URL
    st.image(
        "https://via.placeholder.com/300x200/blue/white?text=Exemple",
        caption="Image d'exemple depuis internet",
        width=300
    )

st.divider()

# ================================
# 7. EXERCICE PRATIQUE
# ================================
st.subheader("7️⃣ Exercice pratique")

st.markdown("""
**🎯 À vous de jouer !**
Créez un mini-outil de traitement de fichiers.
""")

with st.expander("📝 Exercice : Outil de fichiers"):
    st.markdown("""
    **Mission :**
    
    1. Upload d'un CSV
    2. Affichage du contenu avec statistiques
    3. Possibilité de filtrer les données
    4. Download du fichier filtré
    
    **Code de départ :**
    """)
    
    st.code("""
st.subheader("🔧 Outil de traitement CSV")

# 1. Upload
uploaded_file = st.file_uploader("Choisissez votre CSV", type=['csv'])

if uploaded_file:
    # 2. Lecture et affichage
    df = pd.read_csv(uploaded_file)
    st.write(f"📊 {len(df)} lignes, {len(df.columns)} colonnes")
    
    # 3. Filtre simple (exemple sur une colonne numérique)
    if df.select_dtypes(include=[np.number]).columns.any():
        col_num = df.select_dtypes(include=[np.number]).columns[0]
        min_val = st.slider(f"Valeur min pour {col_num}:", 
                           float(df[col_num].min()), 
                           float(df[col_num].max()))
        df_filtered = df[df[col_num] >= min_val]
    else:
        df_filtered = df
    
    # Affichage
    st.dataframe(df_filtered)
    
    # 4. Download
    csv_filtered = df_filtered.to_csv(index=False)
    st.download_button("📥 Télécharger filtré", csv_filtered, "filtered.csv")
""")

# Zone de test
st.markdown("**💻 Exemple fonctionnel :**")

st.markdown("### 🔧 Outil de traitement CSV")

# 1. Upload
uploaded_file_exercise = st.file_uploader("Choisissez votre CSV", type=['csv'], key="exercise_upload")

if uploaded_file_exercise:
    try:
        # 2. Lecture et affichage
        df_exercise = pd.read_csv(uploaded_file_exercise)
        st.write(f"📊 {len(df_exercise)} lignes, {len(df_exercise.columns)} colonnes")
        st.write("**Aperçu des données:**")
        st.dataframe(df_exercise.head(), use_container_width=True)
        
        # 3. Filtre simple 
        if len(df_exercise.select_dtypes(include=[np.number]).columns) > 0:
            col_num = df_exercise.select_dtypes(include=[np.number]).columns[0]
            min_val = st.slider(f"Valeur minimum pour {col_num}:", 
                               float(df_exercise[col_num].min()), 
                               float(df_exercise[col_num].max()),
                               key="exercise_slider")
            df_filtered = df_exercise[df_exercise[col_num] >= min_val]
            
            st.write(f"**Données filtrées:** {len(df_filtered)} lignes")
        else:
            df_filtered = df_exercise
            st.info("Aucune colonne numérique trouvée, pas de filtre appliqué")
        
        # Affichage des données filtrées
        st.dataframe(df_filtered, use_container_width=True)
        
        # 4. Download
        csv_filtered = df_filtered.to_csv(index=False)
        st.download_button(
            "📥 Télécharger les données filtrées", 
            csv_filtered, 
            "donnees_filtrees.csv",
            key="exercise_download"
        )
        
    except Exception as e:
        st.error(f"Erreur lors du traitement: {e}")

else:
    # Créer des données d'exemple pour tester
    st.info("💡 Pas de fichier ? Testez avec des données d'exemple !")
    
    if st.button("📊 Créer des données d'exemple", key="create_example_data"):
        # Générer des données d'exemple
        np.random.seed(42)
        exemple_data = pd.DataFrame({
            'Nom': [f'Personne_{i}' for i in range(1, 21)],
            'Age': np.random.randint(20, 60, 20),
            'Salaire': np.random.randint(30000, 80000, 20),
            'Ville': np.random.choice(['Paris', 'Lyon', 'Marseille', 'Toulouse'], 20)
        })
        
        # Sauvegarder en CSV temporaire
        csv_exemple = exemple_data.to_csv(index=False)
        
        st.success("✅ Données d'exemple créées!")
        st.dataframe(exemple_data)
        
        st.download_button(
            "📥 Télécharger les données d'exemple",
            csv_exemple,
            "donnees_exemple.csv",
            key="download_example"
        )

st.divider()

# ================================
# 8. RÉCAPITULATIF
# ================================
st.subheader("8️⃣ Récapitulatif")

st.markdown("""
**🎓 Ce que vous avez appris :**

✅ **Upload général :** `st.file_uploader()` pour tous types de fichiers  
✅ **Upload images :** Avec `type=['png', 'jpg']` + `st.image()`  
✅ **Upload CSV :** Avec `pd.read_csv()` pour lire les données  
✅ **Download :** `st.download_button()` pour faire télécharger  
✅ **Images fixes :** `st.image()` pour afficher des images  
✅ **Types MIME :** `text/csv`, `text/plain` pour les downloads  

**💡 Conseils pratiques :**
- Toujours gérer les erreurs avec `try/except`
- Vérifier que le fichier existe avant de le traiter
- Limiter les types de fichiers acceptés pour la sécurité
- Donner des noms explicites aux fichiers téléchargés

**🚀 Prochaine étape :** T_07 - Fonctionnalités avancées
""")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⬅️ Module précédent (T_05)", key="nav_prev_t6"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_05 - Layout")

with col2:
    if st.button("📚 Retour au sommaire", key="nav_home_t6"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_00 - Sommaire")

with col3:
    if st.button("➡️ Module suivant (T_07)", key="nav_next_t6"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_07 - Avancé")

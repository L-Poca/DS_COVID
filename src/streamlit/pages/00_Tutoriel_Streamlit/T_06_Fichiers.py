import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
import json
import zipfile
from datetime import datetime
import tempfile
import os

st.header("📁 T_06 - Gestion Avancée des Fichiers & Médias")

st.markdown("**📋 Objectif :** Maîtriser l'upload, le traitement, la validation et le téléchargement de fichiers pour créer des applications robustes et polyvalentes capables de gérer tous types de données et médias.")

st.markdown("---")

# ================================
# 1. UPLOAD DE FICHIERS AVEC VALIDATION AVANCÉE
# ================================
st.subheader("1️⃣ Upload de fichiers avec validation avancée")

st.markdown("""
**📖 Description :**
L'upload de fichiers est la porte d'entrée des données dans votre application. Au-delà du simple chargement,
une validation robuste garantit la sécurité, la performance et la qualité des données traitées.
Streamlit offre un contrôle granulaire sur les types, tailles et formats acceptés.

**🎯 Validations essentielles :**
- **Types de fichiers** : Filtrage par extensions (.csv, .xlsx, .png, etc.)
- **Taille maximum** : Éviter les fichiers trop volumineux (par défaut 200MB)
- **Validation du contenu** : Vérifier la structure des données
- **Sécurité** : Prévenir l'injection de code malveillant
- **Feedback utilisateur** : Messages d'erreur clairs et informatifs

**💡 Bonnes pratiques :**
- Toujours valider côté serveur, jamais uniquement côté client
- Définir des limites de taille appropriées selon le use case
- Fournir des exemples de formats acceptés
- Gérer les erreurs avec des messages explicites
- Logger les tentatives d'upload pour audit et debug
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code('''
# Configuration des limitations
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = ['csv', 'xlsx', 'txt', 'json']

def validate_file(uploaded_file):
    """Validation complète d'un fichier uploadé"""
    if uploaded_file is None:
        return False, "Aucun fichier sélectionné"
    
    # Vérifier la taille
    if uploaded_file.size > MAX_FILE_SIZE:
        size_mb = uploaded_file.size / (1024 * 1024)
        return False, f"Fichier trop volumineux: {size_mb:.1f}MB (max: 10MB)"
    
    # Vérifier l'extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in ALLOWED_TYPES:
        return False, f"Type non supporté: .{file_extension}"
    
    return True, "Fichier valide"

# Interface d'upload avec validation
uploaded_file = st.file_uploader(
    "Sélectionnez votre fichier",
    type=ALLOWED_TYPES,
    help="Types acceptés: CSV, Excel, TXT, JSON - Max 10MB"
)

if uploaded_file is not None:
    # Validation
    is_valid, message = validate_file(uploaded_file)
    
    if is_valid:
        st.success(f"✅ {message}")
        
        # Informations détaillées
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Nom", uploaded_file.name)
        with col2:
            size_mb = uploaded_file.size / (1024 * 1024)
            st.metric("Taille", f"{size_mb:.2f} MB")
        with col3:
            file_type = uploaded_file.name.split('.')[-1].upper()
            st.metric("Type", file_type)
    
    else:
        st.error(f"❌ {message}")
''', language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Configuration pour la démo
    demo_max_size = 5 * 1024 * 1024  # 5 MB pour la démo
    demo_allowed_types = ['csv', 'xlsx', 'txt', 'json']
    
    def demo_validate_file(uploaded_file):
        if uploaded_file is None:
            return False, "Aucun fichier"
        
        if uploaded_file.size > demo_max_size:
            size_mb = uploaded_file.size / (1024 * 1024)
            return False, f"Trop volumineux: {size_mb:.1f}MB (max: 5MB)"
        
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in demo_allowed_types:
            return False, f"Type non supporté: .{file_extension}"
        
        return True, "Fichier valide"
    
    st.markdown("**📤 Test d'Upload**")
    demo_file = st.file_uploader(
        "Testez la validation",
        type=demo_allowed_types,
        help="CSV, Excel, TXT, JSON - Max 5MB",
        key="demo_upload_validation"
    )
    
    if demo_file is not None:
        is_valid, message = demo_validate_file(demo_file)
        
        if is_valid:
            st.success(f"✅ {message}")
            
            # Métriques
            demo_col1, demo_col2 = st.columns(2)
            with demo_col1:
                st.metric("Fichier", demo_file.name)
            with demo_col2:
                size_kb = demo_file.size / 1024
                st.metric("Taille", f"{size_kb:.1f} KB")
        
        else:
            st.error(f"❌ {message}")

st.divider()

# ================================
# 2. TÉLÉCHARGEMENT ET EXPORT DE DONNÉES
# ================================
st.subheader("2️⃣ Téléchargement et export de données")

st.markdown("""
**📖 Description :**
Le téléchargement permet aux utilisateurs d'exporter les résultats de leur travail dans différents formats.
Une stratégie d'export bien conçue améliore l'utilisabilité et permet l'intégration avec d'autres outils.

**🎯 Formats d'export courants :**
- **CSV** : Données tabulaires, compatible Excel
- **Excel** : Feuilles multiples, formatage avancé
- **JSON** : Structures de données, APIs
- **PDF** : Rapports finalisés, présentation
- **ZIP** : Archives multiples, backup

**💡 Bonnes pratiques d'export :**
- Noms de fichiers explicites avec timestamp
- Encodage UTF-8 pour caractères spéciaux
- Métadonnées incluses (date création, source)
- Validation des données avant export
- Options de formatage selon l'usage
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code('''
# Fonction utilitaire pour créer des données d'exemple
def create_sample_data():
    """Génère des données d'exemple pour démonstration"""
    np.random.seed(42)
    data = {
        'ID': range(1, 101),
        'Nom': [f'Utilisateur_{i}' for i in range(1, 101)],
        'Age': np.random.randint(18, 65, 100),
        'Salaire': np.random.randint(30000, 80000, 100),
        'Département': np.random.choice(['IT', 'RH', 'Finance', 'Marketing'], 100),
        'Performance': np.random.uniform(1, 5, 100).round(2)
    }
    return pd.DataFrame(data)

# Génération des données d'exemple
sample_df = create_sample_data()

st.markdown("**📊 Données d'exemple à exporter:**")
st.dataframe(sample_df.head(), use_container_width=True)

# Export CSV
st.markdown("**📄 Export CSV**")
csv_data = sample_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Télécharger CSV",
    data=csv_data,
    file_name=f'donnees_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
    mime='text/csv',
    help="Fichier CSV compatible Excel"
)

# Export Excel avec formatage
st.markdown("**📊 Export Excel Avancé**")

def create_excel_download():
    """Crée un fichier Excel avec formatage"""
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Feuille principale
        sample_df.to_excel(writer, sheet_name='Données', index=False)
        
        # Feuille statistiques
        stats = sample_df.describe()
        stats.to_excel(writer, sheet_name='Statistiques')
        
        # Feuille par département
        for dept in sample_df['Département'].unique():
            dept_data = sample_df[sample_df['Département'] == dept]
            dept_data.to_excel(writer, sheet_name=f'Dept_{dept}', index=False)
    
    return buffer.getvalue()

excel_data = create_excel_download()

st.download_button(
    label="📥 Télécharger Excel (Multi-feuilles)",
    data=excel_data,
    file_name=f'rapport_complet_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

# Export JSON structuré
st.markdown("**🔧 Export JSON**")

# Créer une structure JSON riche
json_export = {
    'metadata': {
        'export_date': datetime.now().isoformat(),
        'total_records': len(sample_df),
        'data_source': 'Application Streamlit Demo'
    },
    'summary': {
        'departments': sample_df['Département'].unique().tolist(),
        'avg_age': float(sample_df['Age'].mean().round(1)),
        'avg_salary': float(sample_df['Salaire'].mean().round(0))
    },
    'data': sample_df.to_dict('records')
}

json_data = json.dumps(json_export, indent=2, ensure_ascii=False).encode('utf-8')

st.download_button(
    label="📥 Télécharger JSON",
    data=json_data,
    file_name=f'donnees_structurees_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
    mime='application/json'
)
''', language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Données d'exemple pour la démo
    def create_demo_data():
        np.random.seed(42)
        data = {
            'ID': range(1, 21),
            'Produit': [f'Produit_{i}' for i in range(1, 21)],
            'Prix': np.random.randint(10, 100, 20),
            'Stock': np.random.randint(0, 50, 20),
            'Catégorie': np.random.choice(['Électronique', 'Vêtements', 'Maison'], 20)
        }
        return pd.DataFrame(data)
    
    demo_df = create_demo_data()
    
    st.markdown("**📊 Données Demo:**")
    st.dataframe(demo_df.head(5), use_container_width=True)
    
    # Export CSV démo
    demo_csv = demo_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 CSV Demo",
        data=demo_csv,
        file_name=f'demo_data_{datetime.now().strftime("%H%M%S")}.csv',
        mime='text/csv',
        key="demo_csv_download"
    )
    
    # Export JSON démo
    demo_json_export = {
        'export_info': {
            'date': datetime.now().isoformat(),
            'records': len(demo_df)
        },
        'data': demo_df.to_dict('records')[:5]  # Limiter pour la démo
    }
    
    demo_json_data = json.dumps(demo_json_export, indent=2).encode('utf-8')
    
    st.download_button(
        label="📥 JSON Demo",
        data=demo_json_data,
        file_name=f'demo_struct_{datetime.now().strftime("%H%M%S")}.json',
        mime='application/json',
        key="demo_json_download"
    )
    
    st.info("💡 Fichiers générés avec timestamp unique")

st.divider()

# ================================
# 3. GESTION D'IMAGES SIMPLIFIÉE
# ================================
st.subheader("3️⃣ Gestion d'images et analyse visuelle")

st.markdown("""
**📖 Description :**
La gestion d'images permet d'enrichir les applications avec du contenu visuel.
Au-delà de l'affichage, elle inclut l'analyse des propriétés, la validation des formats,
et des fonctionnalités de traitement basique pour optimiser l'expérience utilisateur.

**🎯 Fonctionnalités clés :**
- **Validation de formats** : PNG, JPG, JPEG, GIF
- **Analyse des propriétés** : Dimensions, taille, format
- **Redimensionnement** : Création de miniatures
- **Affichage optimisé** : Contrôle de la largeur et du cache

**💡 Bonnes pratiques :**
- Limiter la taille des fichiers pour les performances
- Créer des miniatures pour les galeries
- Validation côté serveur des formats d'image
- Gestion d'erreurs avec messages explicites
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code('''
# Fonction d'analyse d'image
def analyze_image_simple(image_file):
    """Analyse basique d'une image"""
    try:
        image = Image.open(image_file)
        
        analysis = {
            'width': image.size[0],
            'height': image.size[1],
            'format': image.format,
            'mode': image.mode,
            'size_kb': round(image_file.size / 1024, 1)
        }
        
        return True, analysis, image
    except Exception as e:
        return False, str(e), None

# Interface d'upload d'images
st.markdown("**🖼️ Upload d'Images**")
uploaded_images = st.file_uploader(
    "Sélectionnez des images",
    type=['png', 'jpg', 'jpeg', 'gif'],
    accept_multiple_files=True,
    help="Formats: PNG, JPG, JPEG, GIF"
)

if uploaded_images:
    st.success(f"📸 {len(uploaded_images)} image(s) chargée(s)")
    
    for idx, img_file in enumerate(uploaded_images):
        st.markdown(f"**📷 Image {idx + 1}: {img_file.name}**")
        
        # Analyse
        success, analysis, pil_image = analyze_image_simple(img_file)
        
        if success:
            # Affichage en colonnes
            img_col1, img_col2 = st.columns([1, 2])
            
            with img_col1:
                st.image(pil_image, caption=img_file.name, width=200)
            
            with img_col2:
                st.write(f"**Dimensions:** {analysis['width']} x {analysis['height']}")
                st.write(f"**Format:** {analysis['format']}")
                st.write(f"**Mode:** {analysis['mode']}")
                st.write(f"**Taille:** {analysis['size_kb']} KB")
                
                # Calcul des mégapixels
                megapixels = (analysis['width'] * analysis['height']) / 1000000
                st.write(f"**Mégapixels:** {megapixels:.1f} MP")
        
        else:
            st.error(f"❌ Erreur: {analysis}")
        
        st.divider()
''', language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Démo pour images
    st.markdown("**🖼️ Test Images**")
    demo_images = st.file_uploader(
        "Uploadez des images",
        type=['png', 'jpg', 'jpeg', 'gif'],
        accept_multiple_files=True,
        help="Formats acceptés: PNG, JPG, JPEG, GIF",
        key="demo_images_upload"
    )
    
    if demo_images:
        st.success(f"📸 {len(demo_images)} image(s) chargée(s)")
        
        for idx, img_file in enumerate(demo_images[:2]):  # Limiter à 2
            try:
                image = Image.open(img_file)
                width, height = image.size
                file_size_kb = round(img_file.size / 1024, 1)
                
                with st.expander(f"🖼️ {img_file.name}"):
                    demo_img_col1, demo_img_col2 = st.columns(2)
                    
                    with demo_img_col1:
                        st.image(image, caption=img_file.name, width=150)
                    
                    with demo_img_col2:
                        st.write(f"**Dimensions:** {width}x{height}")
                        st.write(f"**Format:** {image.format}")
                        st.write(f"**Taille:** {file_size_kb} KB")
                        
                        megapixels = round((width * height) / 1000000, 1)
                        st.write(f"**Mégapixels:** {megapixels} MP")
            
            except Exception as e:
                st.error(f"Erreur avec {img_file.name}: {str(e)}")
        
        if len(demo_images) > 2:
            st.info(f"... et {len(demo_images) - 2} autre(s) image(s)")

st.markdown("---")

st.success("🎉 **Félicitations !** Vous maîtrisez maintenant la gestion complète des fichiers et médias avec Streamlit !")

st.markdown("""
**🚀 Points clés à retenir :**

**📤 Upload et Validation :**
- Validez toujours les fichiers côté serveur
- Définissez des limites de taille appropriées
- Fournissez des messages d'erreur clairs
- Supportez les formats métier essentiels

**📥 Export et Téléchargement :**
- Proposez plusieurs formats selon l'usage
- Ajoutez des timestamps aux noms de fichiers
- Incluez des métadonnées dans les exports
- Optimisez pour l'intégration avec d'autres outils

**🖼️ Gestion d'Images :**
- Analysez les propriétés pour validation
- Créez des miniatures pour les performances
- Gérez les erreurs de format gracieusement
- Optimisez l'affichage selon le contexte

**🔗 Prochaine étape :** Explorez T_08_Performance pour optimiser vos applications Streamlit !
""")

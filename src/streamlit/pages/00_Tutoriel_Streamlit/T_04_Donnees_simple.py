import streamlit as st
import pandas as pd
import numpy as np

st.header("📊 T_04 - Gestion des Données (Version Simple)")

st.markdown("**📋 Objectif :** Apprendre à afficher et manipuler des données - tableaux, CSV, filtres simples.")

st.markdown("---")

# ================================
# 1. AFFICHER UN TABLEAU SIMPLE
# ================================
st.subheader("1️⃣ Afficher un tableau")

st.markdown("""
**📖 Explication simple :**
Un tableau permet d'organiser et afficher des données de façon claire.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
import pandas as pd

# Créer des données simples
data = {
    'Nom': ['Alice', 'Bob', 'Charlie'],
    'Âge': [25, 30, 35],
    'Ville': ['Paris', 'Lyon', 'Marseille']
}

# Créer un DataFrame
df = pd.DataFrame(data)

# Afficher le tableau
st.dataframe(df)
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Créer des données simples
    data = {
        'Nom': ['Alice', 'Bob', 'Charlie'],
        'Âge': [25, 30, 35],
        'Ville': ['Paris', 'Lyon', 'Marseille']
    }
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

st.divider()

# ================================
# 2. TABLEAU AVEC PLUS DE DONNÉES
# ================================
st.subheader("2️⃣ Tableau avec plus de données")

st.markdown("""
**📖 Explication simple :**
Quand on a beaucoup de données, Streamlit permet de naviguer facilement.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Créer plus de données
noms = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 
        'Frank', 'Grace', 'Henry', 'Iris', 'Jack']
ages = [25, 30, 35, 28, 32, 45, 29, 38, 26, 41]
villes = ['Paris', 'Lyon', 'Marseille', 'Toulouse', 
          'Nice', 'Nantes', 'Strasbourg', 'Montpellier', 
          'Bordeaux', 'Lille']
salaires = [35000, 42000, 38000, 36000, 44000, 
            55000, 39000, 48000, 37000, 52000]

# Créer le DataFrame
df = pd.DataFrame({
    'Nom': noms,
    'Âge': ages,
    'Ville': villes,
    'Salaire': salaires
})

# Afficher
st.dataframe(df, height=300)
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Créer plus de données
    noms = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 
            'Frank', 'Grace', 'Henry', 'Iris', 'Jack']
    ages = [25, 30, 35, 28, 32, 45, 29, 38, 26, 41]
    villes = ['Paris', 'Lyon', 'Marseille', 'Toulouse', 
              'Nice', 'Nantes', 'Strasbourg', 'Montpellier', 
              'Bordeaux', 'Lille']
    salaires = [35000, 42000, 38000, 36000, 44000, 
                55000, 39000, 48000, 37000, 52000]
    
    df_large = pd.DataFrame({
        'Nom': noms,
        'Âge': ages,
        'Ville': villes,
        'Salaire': salaires
    })
    
    st.dataframe(df_large, height=300, use_container_width=True)

st.divider()

# ================================
# 3. MÉTRIQUES SIMPLES
# ================================
st.subheader("3️⃣ Afficher des métriques")

st.markdown("""
**📖 Explication simple :**
Les métriques permettent de montrer les chiffres importants en grand.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Calculer des statistiques
nombre_employes = len(df)
age_moyen = df['Âge'].mean()
salaire_max = df['Salaire'].max()

# Afficher les métriques
st.metric("Nombre d'employés", nombre_employes)
st.metric("Âge moyen", f"{age_moyen:.1f} ans")
st.metric("Salaire maximum", f"{salaire_max:,} €")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Calculer des statistiques
    nombre_employes = len(df_large)
    age_moyen = df_large['Âge'].mean()
    salaire_max = df_large['Salaire'].max()
    
    # Afficher les métriques
    st.metric("Nombre d'employés", nombre_employes)
    st.metric("Âge moyen", f"{age_moyen:.1f} ans")
    st.metric("Salaire maximum", f"{salaire_max:,} €")

st.divider()

# ================================
# 4. FILTRER LES DONNÉES
# ================================
st.subheader("4️⃣ Filtrer les données")

st.markdown("""
**📖 Explication simple :**
Les filtres permettent de ne montrer que certaines données.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Widget pour filtrer par âge
age_min = st.slider("Âge minimum:", 20, 50, 25)

# Filtrer les données
df_filtre = df[df['Âge'] >= age_min]

# Afficher le résultat
st.write(f"Personnes de {age_min} ans ou plus:")
st.dataframe(df_filtre)
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Widget pour filtrer par âge
    age_min = st.slider("Âge minimum:", 20, 50, 25, key="demo_age_filter")
    
    # Filtrer les données
    df_filtre = df_large[df_large['Âge'] >= age_min]
    
    # Afficher le résultat
    st.write(f"Personnes de {age_min} ans ou plus:")
    st.dataframe(df_filtre, use_container_width=True)

st.divider()

# ================================
# 5. FILTRER PAR CATÉGORIE
# ================================
st.subheader("5️⃣ Filtrer par catégorie")

st.markdown("""
**📖 Explication simple :**
On peut aussi filtrer par ville, nom, ou toute autre catégorie.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Widget pour choisir les villes
villes_selectionnees = st.multiselect(
    "Choisissez les villes:",
    df['Ville'].unique(),
    default=df['Ville'].unique()[:3]
)

# Filtrer par villes
if villes_selectionnees:
    df_villes = df[df['Ville'].isin(villes_selectionnees)]
    st.dataframe(df_villes)
else:
    st.write("Sélectionnez au moins une ville")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Widget pour choisir les villes
    villes_selectionnees = st.multiselect(
        "Choisissez les villes:",
        df_large['Ville'].unique(),
        default=list(df_large['Ville'].unique()[:3]),
        key="demo_ville_filter"
    )
    
    # Filtrer par villes
    if villes_selectionnees:
        df_villes = df_large[df_large['Ville'].isin(villes_selectionnees)]
        st.dataframe(df_villes, use_container_width=True)
    else:
        st.write("Sélectionnez au moins une ville")

st.divider()

# ================================
# 6. RECHERCHE DANS LES DONNÉES
# ================================
st.subheader("6️⃣ Recherche simple")

st.markdown("""
**📖 Explication simple :**
Permettre à l'utilisateur de rechercher dans les données.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Zone de recherche
recherche = st.text_input("Rechercher un nom:")

# Filtrer par nom
if recherche:
    df_recherche = df[df['Nom'].str.contains(
        recherche, case=False, na=False
    )]
    
    if not df_recherche.empty:
        st.dataframe(df_recherche)
    else:
        st.write("Aucun résultat trouvé")
else:
    st.write("Tapez un nom pour rechercher")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Zone de recherche
    recherche = st.text_input("Rechercher un nom:", key="demo_search")
    
    # Filtrer par nom
    if recherche:
        df_recherche = df_large[df_large['Nom'].str.contains(
            recherche, case=False, na=False
        )]
        
        if not df_recherche.empty:
            st.dataframe(df_recherche, use_container_width=True)
        else:
            st.write("Aucun résultat trouvé")
    else:
        st.write("Tapez un nom pour rechercher")

st.divider()

# ================================
# 7. CRÉER DES DONNÉES À LA VOLÉE
# ================================
st.subheader("7️⃣ Créer des données aléatoirement")

st.markdown("""
**📖 Explication simple :**
Parfois utile pour tester avec des données générées automatiquement.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
import numpy as np

# Widget pour choisir le nombre de lignes
nb_lignes = st.number_input(
    "Nombre de lignes:", 
    min_value=1, max_value=100, value=10
)

# Générer des données aléatoirement
np.random.seed(42)  # Pour reproduire les mêmes données

df_aleatoire = pd.DataFrame({
    'ID': range(1, nb_lignes + 1),
    'Valeur A': np.random.randint(1, 100, nb_lignes),
    'Valeur B': np.random.randint(50, 200, nb_lignes),
    'Catégorie': np.random.choice(['X', 'Y', 'Z'], nb_lignes)
})

st.dataframe(df_aleatoire)
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Widget pour choisir le nombre de lignes
    nb_lignes = st.number_input(
        "Nombre de lignes:", 
        min_value=1, max_value=100, value=10,
        key="demo_nb_lignes"
    )
    
    # Générer des données aléatoirement
    np.random.seed(42)  # Pour reproduire les mêmes données
    
    df_aleatoire = pd.DataFrame({
        'ID': range(1, nb_lignes + 1),
        'Valeur A': np.random.randint(1, 100, nb_lignes),
        'Valeur B': np.random.randint(50, 200, nb_lignes),
        'Catégorie': np.random.choice(['X', 'Y', 'Z'], nb_lignes)
    })
    
    st.dataframe(df_aleatoire, use_container_width=True)

st.divider()

# ================================
# 8. EXERCICE PRATIQUE
# ================================
st.subheader("8️⃣ Exercice pratique")

st.markdown("""
**🎯 À vous de jouer !**
Créez votre propre tableau de données avec filtres.
""")

with st.expander("📝 Exercice : Mon tableau avec filtres"):
    st.markdown("""
    **Mission :**
    
    1. Créez un tableau avec 5 colonnes de votre choix
    2. Ajoutez un filtre par nombre
    3. Ajoutez un filtre par catégorie
    4. Affichez des métriques sur vos données
    
    **Exemple - Notes d'étudiants :**
    """)
    
    st.code("""
# Données d'étudiants
etudiants = {
    'Nom': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Classe': ['A', 'B', 'A', 'C', 'B'],
    'Math': [15, 12, 18, 14, 16],
    'Français': [16, 14, 15, 17, 13],
    'Sciences': [17, 13, 19, 15, 14]
}

df_notes = pd.DataFrame(etudiants)

# Filtre par note minimum
note_min = st.slider("Note minimum en Math:", 0, 20, 10)
df_filtre = df_notes[df_notes['Math'] >= note_min]

# Filtre par classe
classes = st.multiselect("Classes:", ['A', 'B', 'C'], default=['A', 'B'])
if classes:
    df_filtre = df_filtre[df_filtre['Classe'].isin(classes)]

# Affichage
st.dataframe(df_filtre)

# Métriques
st.metric("Moyenne Math", f"{df_filtre['Math'].mean():.1f}")
""")

# Zone de test
st.markdown("**💻 Exemple fonctionnel :**")

# Données d'étudiants
etudiants = {
    'Nom': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace'],
    'Classe': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
    'Math': [15, 12, 18, 14, 16, 17, 13],
    'Français': [16, 14, 15, 17, 13, 18, 15],
    'Sciences': [17, 13, 19, 15, 14, 16, 16]
}

df_notes = pd.DataFrame(etudiants)

# Filtres
note_min = st.slider("Note minimum en Math:", 0, 20, 10, key="demo_note_min")
classes = st.multiselect("Classes:", ['A', 'B', 'C'], default=['A', 'B'], key="demo_classes")

# Application des filtres
df_filtre_notes = df_notes[df_notes['Math'] >= note_min]
if classes:
    df_filtre_notes = df_filtre_notes[df_filtre_notes['Classe'].isin(classes)]

# Affichage
st.dataframe(df_filtre_notes, use_container_width=True)

# Métriques
if not df_filtre_notes.empty:
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Moyenne Math", f"{df_filtre_notes['Math'].mean():.1f}")
    with col_m2:
        st.metric("Moyenne Français", f"{df_filtre_notes['Français'].mean():.1f}")
    with col_m3:
        st.metric("Nombre d'étudiants", len(df_filtre_notes))

st.divider()

# ================================
# 9. RÉCAPITULATIF
# ================================
st.subheader("9️⃣ Récapitulatif")

st.markdown("""
**🎓 Ce que vous avez appris :**

✅ **Tableaux :** `st.dataframe()` pour afficher des données  
✅ **Métriques :** `st.metric()` pour les chiffres importants  
✅ **Filtres numériques :** Utiliser `df[df['colonne'] >= valeur]`  
✅ **Filtres catégories :** Utiliser `df['colonne'].isin(liste)`  
✅ **Recherche :** Utiliser `df['colonne'].str.contains()`  
✅ **Statistiques :** `.mean()`, `.max()`, `.min()`, `len()`  
✅ **Données aléatoaires :** `np.random` pour tester  

**💡 Conseils :**
- Toujours vérifier si le DataFrame filtré n'est pas vide
- Utiliser `use_container_width=True` pour l'affichage
- Combiner plusieurs filtres pour plus de flexibilité

**🚀 Prochaine étape :** T_05 - Layout (organiser l'interface)
""")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⬅️ Module précédent (T_03)", key="nav_prev_t4"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_03 - Graphiques")

with col2:
    if st.button("📚 Retour au sommaire", key="nav_home_t4"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_00 - Sommaire")

with col3:
    if st.button("➡️ Module suivant (T_05)", key="nav_next_t4"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_05 - Layout")

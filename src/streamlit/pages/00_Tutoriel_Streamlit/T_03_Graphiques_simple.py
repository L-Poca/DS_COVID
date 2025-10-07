import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

st.header("📊 T_03 - Graphiques et Visualisations (Version Simple)")

st.markdown("**📋 Objectif :** Apprendre à créer des graphiques simples - un type de graphique à la fois, avec des exemples isolés et faciles.")

st.markdown("---")

# ================================
# 1. GRAPHIQUE EN LIGNE SIMPLE
# ================================
st.subheader("1️⃣ Graphique en ligne")

st.markdown("""
**📖 Explication simple :**
Un graphique en ligne montre l'évolution d'une valeur dans le temps.
Parfait pour voir les tendances.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Créer des données simples
import pandas as pd
import numpy as np

# Données d'exemple : température sur 7 jours
jours = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
temperatures = [18, 20, 22, 19, 21, 25, 23]

# Créer un DataFrame
data = pd.DataFrame({
    'Jour': jours,
    'Température': temperatures
})

# Afficher le graphique
st.line_chart(data.set_index('Jour'))
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Créer les données
    jours = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    temperatures = [18, 20, 22, 19, 21, 25, 23]
    
    data_line = pd.DataFrame({
        'Jour': jours,
        'Température': temperatures
    })
    
    # Afficher le graphique
    st.line_chart(data_line.set_index('Jour'))

st.divider()

# ================================
# 2. GRAPHIQUE EN BARRES SIMPLE
# ================================
st.subheader("2️⃣ Graphique en barres")

st.markdown("""
**📖 Explication simple :**
Un graphique en barres compare différentes catégories.
Parfait pour voir qui a le plus, qui a le moins.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Données d'exemple : ventes par magasin
magasins = ["Paris", "Lyon", "Marseille", "Toulouse"]
ventes = [150, 120, 180, 90]

# Créer un DataFrame
data = pd.DataFrame({
    'Magasin': magasins,
    'Ventes': ventes
})

# Afficher le graphique
st.bar_chart(data.set_index('Magasin'))
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Créer les données
    magasins = ["Paris", "Lyon", "Marseille", "Toulouse"]
    ventes = [150, 120, 180, 90]
    
    data_bar = pd.DataFrame({
        'Magasin': magasins,
        'Ventes': ventes
    })
    
    # Afficher le graphique
    st.bar_chart(data_bar.set_index('Magasin'))

st.divider()

# ================================
# 3. GRAPHIQUE EN SECTEURS (CAMEMBERT)
# ================================
st.subheader("3️⃣ Graphique en secteurs (camembert)")

st.markdown("""
**📖 Explication simple :**
Un camembert montre les parts de chaque catégorie dans un total.
Parfait pour voir les proportions.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
import plotly.express as px

# Données d'exemple : budget familial
categories = ["Logement", "Nourriture", "Transport", "Loisirs"]
montants = [1200, 400, 300, 200]

# Créer un DataFrame
data = pd.DataFrame({
    'Catégorie': categories,
    'Montant': montants
})

# Créer le camembert
fig = px.pie(data, values='Montant', names='Catégorie', 
             title="Budget familial")

# Afficher le graphique
st.plotly_chart(fig)
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Créer les données
    categories = ["Logement", "Nourriture", "Transport", "Loisirs"]
    montants = [1200, 400, 300, 200]
    
    data_pie = pd.DataFrame({
        'Catégorie': categories,
        'Montant': montants
    })
    
    # Créer le camembert
    fig = px.pie(data_pie, values='Montant', names='Catégorie', 
                 title="Budget familial")
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ================================
# 4. NUAGE DE POINTS
# ================================
st.subheader("4️⃣ Nuage de points")

st.markdown("""
**📖 Explication simple :**
Un nuage de points montre la relation entre deux valeurs.
Parfait pour voir si deux choses sont liées.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
import plotly.express as px

# Données d'exemple : taille vs poids
tailles = [160, 165, 170, 175, 180, 185]
poids = [55, 60, 68, 72, 78, 85]

# Créer un DataFrame
data = pd.DataFrame({
    'Taille (cm)': tailles,
    'Poids (kg)': poids
})

# Créer le nuage de points
fig = px.scatter(data, x='Taille (cm)', y='Poids (kg)',
                title="Relation Taille/Poids")

# Afficher le graphique
st.plotly_chart(fig)
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Créer les données
    tailles = [160, 165, 170, 175, 180, 185]
    poids = [55, 60, 68, 72, 78, 85]
    
    data_scatter = pd.DataFrame({
        'Taille (cm)': tailles,
        'Poids (kg)': poids
    })
    
    # Créer le nuage de points
    fig = px.scatter(data_scatter, x='Taille (cm)', y='Poids (kg)',
                     title="Relation Taille/Poids")
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ================================
# 5. HISTOGRAMME
# ================================
st.subheader("5️⃣ Histogramme")

st.markdown("""
**📖 Explication simple :**
Un histogramme montre la répartition d'une valeur.
Parfait pour voir combien de personnes ont tel âge, telle note, etc.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
import plotly.express as px
import numpy as np

# Données d'exemple : notes d'examen (sur 20)
notes = [8, 10, 12, 14, 15, 16, 18, 12, 13, 11, 
         15, 17, 14, 16, 13, 19, 9, 11, 15, 14]

# Créer un DataFrame
data = pd.DataFrame({'Notes': notes})

# Créer l'histogramme
fig = px.histogram(data, x='Notes', nbins=10,
                  title="Répartition des notes")

# Afficher le graphique
st.plotly_chart(fig)
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Créer les données
    notes = [8, 10, 12, 14, 15, 16, 18, 12, 13, 11, 
             15, 17, 14, 16, 13, 19, 9, 11, 15, 14]
    
    data_hist = pd.DataFrame({'Notes': notes})
    
    # Créer l'histogramme
    fig = px.histogram(data_hist, x='Notes', nbins=10,
                       title="Répartition des notes")
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ================================
# 6. GRAPHIQUE AVEC WIDGETS
# ================================
st.subheader("6️⃣ Graphique interactif")

st.markdown("""
**📖 Explication simple :**
Combinons graphiques et widgets pour rendre les visualisations interactives.
""")

st.markdown("#### 💻 Exemple complet")
st.code("""
# Widget pour choisir le type de graphique
type_graphique = st.selectbox(
    "Choisissez le type de graphique:",
    ["Ligne", "Barres", "Secteurs"]
)

# Données communes
mois = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun"]
ventes = [100, 120, 80, 150, 200, 180]

data = pd.DataFrame({'Mois': mois, 'Ventes': ventes})

# Afficher selon le choix
if type_graphique == "Ligne":
    st.line_chart(data.set_index('Mois'))
elif type_graphique == "Barres":
    st.bar_chart(data.set_index('Mois'))
else:  # Secteurs
    fig = px.pie(data, values='Ventes', names='Mois')
    st.plotly_chart(fig)
""")

st.markdown("#### 🎯 Résultat")

# Widget pour choisir le type de graphique
type_graphique = st.selectbox(
    "Choisissez le type de graphique:",
    ["Ligne", "Barres", "Secteurs"],
    key="demo_graph_type"
)

# Données communes
mois = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun"]
ventes = [100, 120, 80, 150, 200, 180]

data_interactive = pd.DataFrame({'Mois': mois, 'Ventes': ventes})

# Afficher selon le choix
if type_graphique == "Ligne":
    st.line_chart(data_interactive.set_index('Mois'))
elif type_graphique == "Barres":
    st.bar_chart(data_interactive.set_index('Mois'))
else:  # Secteurs
    fig = px.pie(data_interactive, values='Ventes', names='Mois')
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ================================
# 7. EXERCICE PRATIQUE
# ================================
st.subheader("7️⃣ Exercice pratique")

st.markdown("""
**🎯 À vous de jouer !**
Créez vos propres graphiques avec des données personnalisées.
""")

with st.expander("📝 Exercice : Créer ses graphiques"):
    st.markdown("""
    **Mission :**
    
    1. Créez un graphique en barres des populations de 5 villes
    2. Ajoutez un widget pour changer les données
    3. Affichez le graphique choisi
    
    **Code de départ :**
    """)
    
    st.code("""
# Vos données
villes = ["Paris", "Lyon", "Marseille", "Toulouse", "Nice"]
populations = [2200000, 516000, 870000, 479000, 340000]

# Votre DataFrame
data = pd.DataFrame({
    'Ville': villes,
    'Population': populations
})

# Votre graphique
st.bar_chart(data.set_index('Ville'))
""")

# Zone de test
st.markdown("**💻 Testez ici :**")

# Exemple fonctionnel
st.markdown("**Exemple avec widget :**")

# Widget pour modifier les données
multiplicateur = st.slider("Multiplicateur de population:", 0.5, 2.0, 1.0, 0.1, key="demo_mult")

# Données de base
villes = ["Paris", "Lyon", "Marseille", "Toulouse", "Nice"]
populations_base = [2200000, 516000, 870000, 479000, 340000]

# Appliquer le multiplicateur
populations_modifiees = [int(p * multiplicateur) for p in populations_base]

data_exercice = pd.DataFrame({
    'Ville': villes,
    'Population': populations_modifiees
})

# Graphique résultant
st.bar_chart(data_exercice.set_index('Ville'))

st.divider()

# ================================
# 8. RÉCAPITULATIF
# ================================
st.subheader("8️⃣ Récapitulatif")

st.markdown("""
**🎓 Ce que vous avez appris :**

✅ **Ligne :** `st.line_chart()` pour les évolutions dans le temps  
✅ **Barres :** `st.bar_chart()` pour comparer des catégories  
✅ **Camembert :** `px.pie()` pour montrer les proportions  
✅ **Nuage de points :** `px.scatter()` pour les relations entre variables  
✅ **Histogramme :** `px.histogram()` pour les répartitions  
✅ **Interactivité :** Combiner graphiques et widgets  

**🎨 Conseils de choix :**
- **Évolution** → Ligne
- **Comparaison** → Barres  
- **Proportions** → Camembert
- **Relations** → Nuage de points
- **Répartitions** → Histogramme

**🚀 Prochaine étape :** T_04 - Données (tableaux, CSV, etc.)
""")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⬅️ Module précédent (T_02)", key="nav_prev_t3"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_02 - Widgets")

with col2:
    if st.button("📚 Retour au sommaire", key="nav_home_t3"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_00 - Sommaire")

with col3:
    if st.button("➡️ Module suivant (T_04)", key="nav_next_t3"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_04 - Données")

import streamlit as st
import pandas as pd
import plotly.express as px

st.header("🎨 T_05 - Layout et Organisation (Version Simple)")

st.markdown("**📋 Objectif :** Apprendre à organiser votre page - colonnes, barres latérales, conteneurs pour une belle présentation.")

st.markdown("---")

# ================================
# 1. COLONNES SIMPLES
# ================================
st.subheader("1️⃣ Créer des colonnes")

st.markdown("""
**📖 Explication simple :**
Les colonnes permettent d'organiser le contenu côte à côte.
Comme des journaux avec plusieurs colonnes.
""")

st.markdown("#### 💻 Code - 2 colonnes égales")
st.code("""
# Créer 2 colonnes de même taille
col1, col2 = st.columns(2)

# Contenu de la première colonne
with col1:
    st.write("Contenu de gauche")
    st.button("Bouton gauche")

# Contenu de la deuxième colonne  
with col2:
    st.write("Contenu de droite")
    st.button("Bouton droite")
""")

st.markdown("#### 🎯 Résultat")

# Démonstration
col1, col2 = st.columns(2)

with col1:
    st.write("**Colonne de gauche**")
    st.write("Contenu de gauche")
    if st.button("Bouton gauche", key="demo_btn_left"):
        st.success("Bouton gauche cliqué!")

with col2:
    st.write("**Colonne de droite**")
    st.write("Contenu de droite")
    if st.button("Bouton droite", key="demo_btn_right"):
        st.success("Bouton droite cliqué!")

st.divider()

# ================================
# 2. COLONNES DE TAILLES DIFFÉRENTES
# ================================
st.subheader("2️⃣ Colonnes de tailles différentes")

st.markdown("""
**📖 Explication simple :**
On peut faire des colonnes plus larges ou plus étroites selon les besoins.
""")

st.markdown("#### 💻 Code - Colonnes 70% / 30%")
st.code("""
# Colonne large (70%) et colonne étroite (30%)
col_large, col_etroite = st.columns([7, 3])

with col_large:
    st.write("Grande colonne pour le contenu principal")
    
    # Exemple: graphique dans la grande colonne
    data = {'x': [1, 2, 3, 4], 'y': [10, 11, 12, 13]}
    df = pd.DataFrame(data)
    st.line_chart(df.set_index('x'))

with col_etroite:
    st.write("Petite colonne pour les options")
    
    # Exemple: widgets dans la petite colonne
    couleur = st.selectbox("Couleur:", ["Rouge", "Bleu", "Vert"])
    taille = st.slider("Taille:", 1, 10, 5)
""")

st.markdown("#### 🎯 Résultat")

# Démonstration
col_large, col_etroite = st.columns([7, 3])

with col_large:
    st.write("**Grande colonne pour le contenu principal**")
    
    # Exemple: graphique dans la grande colonne
    data = {'x': [1, 2, 3, 4], 'y': [10, 11, 12, 13]}
    df = pd.DataFrame(data)
    st.line_chart(df.set_index('x'))

with col_etroite:
    st.write("**Petite colonne pour les options**")
    
    # Exemple: widgets dans la petite colonne
    couleur = st.selectbox("Couleur:", ["Rouge", "Bleu", "Vert"], key="demo_couleur")
    taille = st.slider("Taille:", 1, 10, 5, key="demo_taille")
    
    st.write(f"Couleur: {couleur}")
    st.write(f"Taille: {taille}")

st.divider()

# ================================
# 3. TROIS COLONNES
# ================================
st.subheader("3️⃣ Trois colonnes")

st.markdown("""
**📖 Explication simple :**
Parfait pour afficher 3 métriques ou 3 graphiques côte à côte.
""")

st.markdown("#### 💻 Code - 3 colonnes pour métriques")
st.code("""
# Créer 3 colonnes égales
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Ventes", "1,234", "+12%")

with col2:
    st.metric("Clients", "567", "+5%")

with col3:
    st.metric("Revenus", "€89,012", "+8%")
""")

st.markdown("#### 🎯 Résultat")

# Démonstration
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Ventes", "1,234", "+12%")

with col2:
    st.metric("Clients", "567", "+5%")

with col3:
    st.metric("Revenus", "€89,012", "+8%")

st.divider()

# ================================
# 4. BARRE LATÉRALE
# ================================
st.subheader("4️⃣ Barre latérale")

st.markdown("""
**📖 Explication simple :**
La barre latérale (sidebar) est parfaite pour les paramètres et options.
Elle ne prend pas de place sur le contenu principal.
""")

st.markdown("#### 💻 Code - Widgets dans la sidebar")
st.code("""
# Ajouter des widgets dans la barre latérale
st.sidebar.title("⚙️ Paramètres")

nom_utilisateur = st.sidebar.text_input("Votre nom:")
niveau = st.sidebar.selectbox("Niveau:", ["Débutant", "Intermédiaire", "Expert"])
notifications = st.sidebar.checkbox("Recevoir des notifications")

# Utiliser ces paramètres dans le contenu principal
if nom_utilisateur:
    st.write(f"Bonjour {nom_utilisateur}!")
    st.write(f"Niveau sélectionné: {niveau}")
    
    if notifications:
        st.info("✅ Notifications activées")
""")

st.markdown("#### 🎯 Résultat")

# Ajouter des widgets dans la barre latérale
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Paramètres T_05")

nom_utilisateur = st.sidebar.text_input("Votre nom:", key="demo_sidebar_nom")
niveau = st.sidebar.selectbox("Niveau:", ["Débutant", "Intermédiaire", "Expert"], key="demo_sidebar_niveau")
notifications = st.sidebar.checkbox("Recevoir des notifications", key="demo_sidebar_notif")

# Utiliser ces paramètres dans le contenu principal
if nom_utilisateur:
    st.write(f"Bonjour {nom_utilisateur}!")
    st.write(f"Niveau sélectionné: {niveau}")
    
    if notifications:
        st.info("✅ Notifications activées")
else:
    st.info("👈 Entrez votre nom dans la barre latérale")

st.divider()

# ================================
# 5. CONTENEURS ET EXPANDERS
# ================================
st.subheader("5️⃣ Conteneurs et sections pliables")

st.markdown("""
**📖 Explication simple :**
Les conteneurs groupent le contenu, les expanders permettent de cacher/montrer des sections.
""")

st.markdown("#### 💻 Code - Conteneur simple")
st.code("""
# Conteneur pour grouper des éléments
with st.container():
    st.write("Contenu dans un conteneur")
    st.button("Bouton dans le conteneur")
    st.write("Tout ce qui est ici est groupé ensemble")
""")

st.markdown("#### 🎯 Résultat - Conteneur")

# Démonstration conteneur
with st.container():
    st.write("**Contenu dans un conteneur**")
    if st.button("Bouton dans le conteneur", key="demo_container_btn"):
        st.success("Bouton du conteneur cliqué!")
    st.write("Tout ce qui est ici est groupé ensemble")

st.markdown("#### 💻 Code - Section pliable")
st.code("""
# Section pliable (fermée par défaut)
with st.expander("📊 Voir les détails"):
    st.write("Contenu caché par défaut")
    st.write("L'utilisateur peut cliquer pour voir")
    
    # On peut mettre n'importe quoi dedans
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    df = pd.DataFrame(data)
    st.dataframe(df)
""")

st.markdown("#### 🎯 Résultat - Section pliable")

# Démonstration expander
with st.expander("📊 Voir les détails"):
    st.write("**Contenu caché par défaut**")
    st.write("L'utilisateur peut cliquer pour voir")
    
    # On peut mettre n'importe quoi dedans
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    df_expander = pd.DataFrame(data)
    st.dataframe(df_expander)

st.divider()

# ================================
# 6. ONGLETS
# ================================
st.subheader("6️⃣ Onglets")

st.markdown("""
**📖 Explication simple :**
Les onglets permettent d'organiser le contenu en plusieurs pages dans la même section.
""")

st.markdown("#### 💻 Code - Onglets")
st.code("""
# Créer des onglets
tab1, tab2, tab3 = st.tabs(["📊 Graphiques", "📋 Données", "⚙️ Paramètres"])

with tab1:
    st.write("Contenu de l'onglet Graphiques")
    # Exemple de graphique
    st.line_chart([1, 3, 2, 4, 5])

with tab2:
    st.write("Contenu de l'onglet Données")
    # Exemple de tableau
    data = {'Col1': [1, 2, 3], 'Col2': ['A', 'B', 'C']}
    st.dataframe(pd.DataFrame(data))

with tab3:
    st.write("Contenu de l'onglet Paramètres")
    # Exemple de paramètres
    st.selectbox("Option:", ["Option 1", "Option 2"])
""")

st.markdown("#### 🎯 Résultat")

# Démonstration onglets
tab1, tab2, tab3 = st.tabs(["📊 Graphiques", "📋 Données", "⚙️ Paramètres"])

with tab1:
    st.write("**Contenu de l'onglet Graphiques**")
    # Exemple de graphique
    st.line_chart([1, 3, 2, 4, 5])

with tab2:
    st.write("**Contenu de l'onglet Données**")
    # Exemple de tableau
    data = {'Col1': [1, 2, 3], 'Col2': ['A', 'B', 'C']}
    st.dataframe(pd.DataFrame(data))

with tab3:
    st.write("**Contenu de l'onglet Paramètres**")
    # Exemple de paramètres
    option_tab = st.selectbox("Option:", ["Option 1", "Option 2"], key="demo_tab_option")
    st.write(f"Vous avez choisi: {option_tab}")

st.divider()

# ================================
# 7. EXERCICE PRATIQUE
# ================================
st.subheader("7️⃣ Exercice pratique")

st.markdown("""
**🎯 À vous de jouer !**
Créez un dashboard avec différents layouts.
""")

with st.expander("📝 Exercice : Créer un dashboard"):
    st.markdown("""
    **Mission :**
    
    1. Créez 3 métriques en haut (colonnes)
    2. Ajoutez un graphique et des paramètres (2 colonnes)
    3. Utilisez la sidebar pour les filtres
    4. Ajoutez des onglets pour différentes vues
    
    **Code de départ :**
    """)
    
    st.code("""
# 1. Métriques en haut
st.subheader("📊 Tableau de bord")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Visiteurs", "1,234", "+5%")
with col2:
    st.metric("Ventes", "567", "+12%")
with col3:
    st.metric("Revenus", "€8,901", "+8%")

# 2. Contenu principal avec sidebar
filtre = st.sidebar.selectbox("Période:", ["Semaine", "Mois", "Année"])

# 3. Graphique et options
col_graph, col_options = st.columns([7, 3])

with col_graph:
    # Votre graphique
    data = [10, 20, 30, 25, 35]
    st.line_chart(data)

with col_options:
    st.write("Options")
    type_vue = st.radio("Vue:", ["Ligne", "Barres"])
    
# 4. Onglets pour plus de contenu
tab1, tab2 = st.tabs(["Analyse", "Rapport"])

with tab1:
    st.write("Contenu analyse")
    
with tab2:
    st.write("Contenu rapport")
""")

# Zone de test
st.markdown("**💻 Exemple fonctionnel :**")

# Exemple de dashboard
st.markdown("### 📊 Tableau de bord exemple")

# 1. Métriques en haut
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric("Visiteurs", "1,234", "+5%")
with metric_col2:
    st.metric("Ventes", "567", "+12%")
with metric_col3:
    st.metric("Revenus", "€8,901", "+8%")

# 2. Sidebar pour filtres
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Filtres Dashboard")
filtre = st.sidebar.selectbox("Période:", ["Semaine", "Mois", "Année"], key="demo_dashboard_filtre")
st.sidebar.write(f"Période sélectionnée: {filtre}")

# 3. Contenu principal
col_graph, col_options = st.columns([7, 3])

with col_graph:
    st.write("**Évolution des ventes**")
    # Données qui changent selon le filtre
    if filtre == "Semaine":
        data_graph = [10, 20, 30, 25, 35, 40, 45]
    elif filtre == "Mois":
        data_graph = [100, 120, 150, 180, 200]
    else:  # Année
        data_graph = [1000, 1200, 1100, 1400]
    
    st.line_chart(data_graph)

with col_options:
    st.write("**Options d'affichage**")
    type_vue = st.radio("Type de vue:", ["Ligne", "Barres"], key="demo_dashboard_vue")
    couleur_theme = st.selectbox("Thème:", ["Bleu", "Rouge", "Vert"], key="demo_dashboard_theme")
    
    st.write(f"Vue: {type_vue}")
    st.write(f"Thème: {couleur_theme}")

# 4. Onglets
tab_analyse, tab_rapport = st.tabs(["📈 Analyse", "📄 Rapport"])

with tab_analyse:
    st.write("**Analyse détaillée**")
    st.write(f"Analyse pour la période: {filtre}")
    st.info("Ici vous pourriez mettre des graphiques détaillés, des statistiques avancées, etc.")

with tab_rapport:
    st.write("**Rapport de synthèse**")
    st.write(f"Rapport généré pour: {filtre}")
    st.success("Ici vous pourriez mettre un résumé, des conclusions, des recommandations, etc.")

st.divider()

# ================================
# 8. RÉCAPITULATIF
# ================================
st.subheader("8️⃣ Récapitulatif")

st.markdown("""
**🎓 Ce que vous avez appris :**

✅ **Colonnes :** `st.columns()` pour organiser côte à côte  
✅ **Sidebar :** `st.sidebar` pour les paramètres  
✅ **Conteneurs :** `st.container()` pour grouper  
✅ **Expanders :** `st.expander()` pour cacher/montrer  
✅ **Onglets :** `st.tabs()` pour organiser en pages  
✅ **Proportions :** `[7, 3]` pour des tailles différentes  

**💡 Conseils de design :**
- Utilisez 2-3 colonnes maximum pour la lisibilité
- Mettez les options importantes dans la sidebar
- Groupez le contenu logiquement avec des conteneurs
- Utilisez les onglets pour éviter de trop faire défiler

**🚀 Prochaine étape :** T_06 - Fichiers (upload, download, images)
""")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⬅️ Module précédent (T_04)", key="nav_prev_t5"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_04 - Données")

with col2:
    if st.button("📚 Retour au sommaire", key="nav_home_t5"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_00 - Sommaire")

with col3:
    if st.button("➡️ Module suivant (T_06)", key="nav_next_t5"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_06 - Fichiers")

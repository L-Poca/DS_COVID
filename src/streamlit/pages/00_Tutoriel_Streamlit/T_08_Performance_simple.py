import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px

st.header("⚡ T_08 - Performance (Version Simple)")

st.markdown("**📋 Objectif :** Apprendre à rendre votre application Streamlit plus rapide et plus fluide.")

st.markdown("---")

# ================================
# 1. POURQUOI L'OPTIMISATION ?
# ================================
st.subheader("1️⃣ Pourquoi optimiser ?")

st.markdown("""
**🐌 Problèmes courants :**
- Application qui rame
- Données qui se rechargent sans arrêt
- Interface qui se bloque
- Utilisateurs qui s'impatientent

**⚡ Solutions simples :**
- Cache pour éviter les recalculs
- Lazy loading (charger au besoin)
- Optimiser les données
- Widgets réactifs
""")

# ================================
# 2. CACHE BASIQUE
# ================================
st.subheader("2️⃣ Cache - La base de l'optimisation")

st.markdown("""
**📖 Principe simple :**
Au lieu de refaire le même calcul 100 fois, on le fait une fois et on garde le résultat en mémoire.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Sans cache (lent)")
    st.code("""
def calcul_lent_sans_cache(n):
    # Simulation d'un calcul compliqué
    time.sleep(2)  # 2 secondes d'attente
    return [i*i for i in range(n)]

# À chaque clic, le calcul recommence
if st.button("🐌 Calcul sans cache"):
    with st.spinner("Calcul en cours..."):
        resultat = calcul_lent_sans_cache(1000)
        st.write(f"Calculé {len(resultat)} éléments")
""")

with col2:
    st.markdown("#### 💻 Code - Avec cache (rapide)")
    st.code("""
@st.cache_data  # ⚡ Cache magique !
def calcul_rapide_avec_cache(n):
    # Même calcul mais avec cache
    time.sleep(2)  # Lent la première fois seulement
    return [i*i for i in range(n)]

# Rapide après la première fois
if st.button("⚡ Calcul avec cache"):
    with st.spinner("Calcul en cours..."):
        resultat = calcul_rapide_avec_cache(1000)
        st.write(f"Calculé {len(resultat)} éléments")
        st.info("💡 Cliquez encore : instantané !")
""")

# Démonstration
st.markdown("#### 🎯 Test en direct")

demo_col1, demo_col2 = st.columns(2)

with demo_col1:
    def calcul_lent_demo(n):
        time.sleep(1)  # 1 seconde pour la démo
        return [i*i for i in range(n)]
    
    if st.button("🐌 Sans cache", key="demo_no_cache"):
        start_time = time.time()
        with st.spinner("Calcul en cours..."):
            resultat = calcul_lent_demo(1000)
            end_time = time.time()
            st.write(f"✅ {len(resultat)} éléments")
            st.write(f"⏱️ Temps: {end_time - start_time:.1f}s")

with demo_col2:
    @st.cache_data
    def calcul_rapide_demo(n):
        time.sleep(1)  # Lent la première fois seulement
        return [i*i for i in range(n)]
    
    if st.button("⚡ Avec cache", key="demo_with_cache"):
        start_time = time.time()
        with st.spinner("Calcul en cours..."):
            resultat = calcul_rapide_demo(1000)
            end_time = time.time()
            st.write(f"✅ {len(resultat)} éléments")
            st.write(f"⏱️ Temps: {end_time - start_time:.1f}s")
            if end_time - start_time < 0.1:
                st.success("🚀 Instantané grâce au cache !")

st.divider()

# ================================
# 3. CACHE POUR DONNÉES
# ================================
st.subheader("3️⃣ Cache pour charger des données")

st.markdown("""
**📖 Cas pratique :**
Éviter de recharger un gros fichier CSV à chaque interaction.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Chargement optimisé")
    st.code("""
@st.cache_data  # Ne charge qu'une fois
def charger_donnees():
    # Simulation : générer des données
    data = pd.DataFrame({
        'nom': [f'Personne_{i}' for i in range(1000)],
        'age': np.random.randint(18, 80, 1000),
        'salaire': np.random.randint(25000, 80000, 1000)
    })
    return data

# Cette fonction ne s'exécute qu'une fois
df = charger_donnees()

# Interface rapide car les données sont en cache
filtre_age = st.slider("Âge minimum:", 18, 80, 25)
df_filtre = df[df['age'] >= filtre_age]

st.write(f"Personnes de {filtre_age} ans et +: {len(df_filtre)}")
st.dataframe(df_filtre.head())
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    @st.cache_data
    def charger_donnees_demo():
        # Simulation d'un chargement
        time.sleep(0.5)  # Simulation
        data = pd.DataFrame({
            'nom': [f'Personne_{i}' for i in range(100)],
            'age': np.random.randint(18, 80, 100),
            'salaire': np.random.randint(25000, 80000, 100)
        })
        return data
    
    # Cette fonction ne s'exécute qu'une fois
    df_demo = charger_donnees_demo()
    
    # Interface rapide car les données sont en cache
    filtre_age = st.slider("Âge minimum:", 18, 80, 25, key="demo_age_filter")
    df_filtre_demo = df_demo[df_demo['age'] >= filtre_age]
    
    st.write(f"**Personnes de {filtre_age} ans et +:** {len(df_filtre_demo)}")
    if len(df_filtre_demo) > 0:
        st.dataframe(df_filtre_demo.head(), use_container_width=True)
    else:
        st.write("Aucune personne trouvée avec ce critère")

st.divider()

# ================================
# 4. LAZY LOADING
# ================================
st.subheader("4️⃣ Lazy Loading - Charger au besoin")

st.markdown("""
**📖 Principe :**
Ne charger les données que quand l'utilisateur en a besoin.
Évite de surcharger l'application au démarrage.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Chargement à la demande")
    st.code("""
# Tabs pour organiser les données lourdes
tab1, tab2, tab3 = st.tabs(["📊 Graphiques", "📈 Stats", "📋 Raw Data"])

with tab1:
    if st.button("📊 Générer graphique"):
        @st.cache_data
        def creer_graphique():
            # Générer des données seulement quand demandé
            x = np.random.randn(1000)
            y = np.random.randn(1000)
            return px.scatter(x=x, y=y, title="Données aléatoires")
        
        fig = creer_graphique()
        st.plotly_chart(fig)

with tab2:
    if st.button("📈 Calculer statistiques"):
        @st.cache_data
        def calculer_stats():
            data = np.random.randn(10000)
            return {
                'moyenne': np.mean(data),
                'mediane': np.median(data),
                'ecart_type': np.std(data)
            }
        
        stats = calculer_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Moyenne", f"{stats['moyenne']:.2f}")
        col2.metric("Médiane", f"{stats['mediane']:.2f}")
        col3.metric("Écart-type", f"{stats['ecart_type']:.2f}")

with tab3:
    if st.button("📋 Charger données brutes"):
        @st.cache_data
        def charger_donnees_brutes():
            return pd.DataFrame(np.random.randn(1000, 5), 
                              columns=['A', 'B', 'C', 'D', 'E'])
        
        df = charger_donnees_brutes()
        st.dataframe(df)
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Tabs pour organiser les données lourdes
    tab1, tab2, tab3 = st.tabs(["📊 Graphiques", "📈 Stats", "📋 Raw Data"])
    
    with tab1:
        if st.button("📊 Générer graphique", key="demo_lazy_graph"):
            @st.cache_data
            def creer_graphique_demo():
                x = np.random.randn(200)
                y = np.random.randn(200)
                return px.scatter(x=x, y=y, title="Données aléatoires")
            
            with st.spinner("Génération du graphique..."):
                fig = creer_graphique_demo()
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if st.button("📈 Calculer statistiques", key="demo_lazy_stats"):
            @st.cache_data
            def calculer_stats_demo():
                data = np.random.randn(1000)
                return {
                    'moyenne': np.mean(data),
                    'mediane': np.median(data),
                    'ecart_type': np.std(data)
                }
            
            with st.spinner("Calcul des statistiques..."):
                stats = calculer_stats_demo()
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                stat_col1.metric("Moyenne", f"{stats['moyenne']:.2f}")
                stat_col2.metric("Médiane", f"{stats['mediane']:.2f}")
                stat_col3.metric("Écart-type", f"{stats['ecart_type']:.2f}")
    
    with tab3:
        if st.button("📋 Charger données brutes", key="demo_lazy_data"):
            @st.cache_data
            def charger_donnees_brutes_demo():
                return pd.DataFrame(np.random.randn(50, 5), 
                                  columns=['A', 'B', 'C', 'D', 'E'])
            
            with st.spinner("Chargement des données..."):
                df_brut = charger_donnees_brutes_demo()
                st.dataframe(df_brut, use_container_width=True)

st.divider()

# ================================
# 5. OPTIMISER LES WIDGETS
# ================================
st.subheader("5️⃣ Widgets optimisés")

st.markdown("""
**📖 Astuce :**
Certains widgets peuvent être optimisés pour de meilleures performances.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Widgets optimisés")
    st.code("""
# Fragment : évite de recharger toute la page
@st.fragment
def section_interactive():
    # Cette section se met à jour indépendamment
    valeur = st.slider("Valeur interactive:", 0, 100, 50)
    
    # Calcul local rapide
    resultat = valeur * 2
    st.write(f"Résultat: {resultat}")
    
    # Graphique qui ne se recalcule que si nécessaire
    if valeur > 0:
        x = range(valeur)
        y = [i*2 for i in x]
        st.line_chart(pd.DataFrame({'x': x, 'y': y}).set_index('x'))

# Utiliser le fragment
section_interactive()

# Le reste de l'app n'est pas affecté
st.write("Cette partie ne bouge pas quand vous changez le slider !")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Fragment pour éviter de recharger toute la page
    @st.fragment
    def section_interactive_demo():
        valeur = st.slider("Valeur interactive:", 0, 100, 50, key="demo_fragment_slider")
        
        # Calcul local rapide
        resultat = valeur * 2
        st.write(f"**Résultat:** {resultat}")
        
        # Graphique simple
        if valeur > 0:
            x = list(range(min(valeur, 20)))  # Limiter pour la performance
            y = [i*2 for i in x]
            if len(x) > 1:
                chart_data = pd.DataFrame({'Valeurs': y}, index=x)
                st.line_chart(chart_data)
    
    # Utiliser le fragment
    section_interactive_demo()
    
    st.info("💡 Cette section se met à jour indépendamment du reste !")

st.divider()

# ================================
# 6. GESTION MÉMOIRE
# ================================
st.subheader("6️⃣ Gestion de la mémoire")

st.markdown("""
**📖 Bonnes pratiques :**
Éviter d'accumuler trop de données en mémoire.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Nettoyage du cache")
    st.code("""
# Afficher l'état du cache
if st.button("📊 État du cache"):
    # Vérifier ce qui est en cache
    st.write("Cache Streamlit actif")
    st.info("Les données sont mises en cache automatiquement")

# Nettoyer le cache si nécessaire
if st.button("🧹 Vider le cache"):
    st.cache_data.clear()
    st.success("Cache vidé !")
    st.info("Les prochains calculs seront plus lents mais la mémoire est libérée")

# Limite de taille pour les gros datasets
@st.cache_data(max_entries=3)  # Garde max 3 versions
def donnees_limitees(taille):
    return pd.DataFrame(np.random.randn(taille, 3))

# Test
taille = st.selectbox("Taille des données:", [100, 1000, 5000])
df = donnees_limitees(taille)
st.write(f"Dataset de {len(df)} lignes créé")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Afficher l'état du cache
    if st.button("📊 État du cache", key="demo_cache_status"):
        st.write("**Cache Streamlit actif**")
        st.info("Les données sont mises en cache automatiquement")
    
    # Nettoyer le cache si nécessaire
    if st.button("🧹 Vider le cache", key="demo_clear_cache"):
        st.cache_data.clear()
        st.success("Cache vidé !")
        st.info("Les prochains calculs seront plus lents mais la mémoire est libérée")
    
    # Limite de taille pour les gros datasets
    @st.cache_data(max_entries=3)
    def donnees_limitees_demo(taille):
        time.sleep(0.2)  # Simulation
        return pd.DataFrame(np.random.randn(taille, 3), columns=['A', 'B', 'C'])
    
    # Test
    taille = st.selectbox("Taille des données:", [50, 100, 200], key="demo_data_size")
    df_limite = donnees_limitees_demo(taille)
    st.write(f"**Dataset de {len(df_limite)} lignes créé**")
    
    # Aperçu des données
    if st.checkbox("Voir aperçu", key="demo_preview_data"):
        st.dataframe(df_limite.head(), use_container_width=True)

st.divider()

# ================================
# 7. CONSEILS PERFORMANCE
# ================================
st.subheader("7️⃣ Conseils pour la performance")

performance_tips = {
    "🚀 Cache": [
        "Utilisez @st.cache_data pour les calculs longs",
        "Cachés les chargements de fichiers",
        "Limitez la taille du cache avec max_entries"
    ],
    "📊 Données": [
        "Chargez les données progressivement",
        "Filtrez avant d'afficher",
        "Utilisez des échantillons pour les gros datasets"
    ],
    "🎛️ Interface": [
        "Utilisez st.fragment pour les sections indépendantes",
        "Groupez les widgets dans des formulaires",
        "Évitez trop de widgets qui se mettent à jour"
    ],
    "💾 Mémoire": [
        "Videz le cache régulièrement",
        "Évitez de stocker des objets lourds en session_state",
        "Libérez les variables non utilisées"
    ]
}

for category, tips in performance_tips.items():
    with st.expander(f"{category} - Astuces"):
        for tip in tips:
            st.write(f"✅ {tip}")

# ================================
# 8. EXERCICE PRATIQUE
# ================================
st.subheader("8️⃣ Exercice : Application optimisée")

st.markdown("""
**🎯 Mission :**
Créer une app qui affiche des données avec plusieurs niveaux d'optimisation.
""")

with st.expander("💻 Code de l'exercice"):
    st.code("""
import streamlit as st
import pandas as pd
import numpy as np
import time

# 1. Cache pour les données
@st.cache_data
def generer_donnees(nb_lignes):
    time.sleep(1)  # Simulation chargement
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=nb_lignes),
        'ventes': np.random.randint(100, 1000, nb_lignes),
        'produit': np.random.choice(['A', 'B', 'C'], nb_lignes),
        'region': np.random.choice(['Nord', 'Sud', 'Est', 'Ouest'], nb_lignes)
    })

# 2. Interface avec cache
st.title("📊 Analyse des ventes - Version optimisée")

# Sidebar pour les paramètres
with st.sidebar:
    nb_lignes = st.selectbox("Taille des données:", [100, 500, 1000])
    
    # Bouton pour recharger
    if st.button("🔄 Recharger données"):
        st.cache_data.clear()

# 3. Chargement avec cache
with st.spinner("Chargement des données..."):
    df = generer_donnees(nb_lignes)

st.success(f"✅ {len(df)} lignes chargées")

# 4. Fragment pour les filtres
@st.fragment
def section_filtres():
    col1, col2 = st.columns(2)
    
    with col1:
        produits = st.multiselect("Produits:", df['produit'].unique(), 
                                default=df['produit'].unique())
    
    with col2:
        regions = st.multiselect("Régions:", df['region'].unique(),
                               default=df['region'].unique())
    
    # Filtrage optimisé
    df_filtre = df[
        (df['produit'].isin(produits)) & 
        (df['region'].isin(regions))
    ]
    
    # Affichage
    if len(df_filtre) > 0:
        st.metric("Ventes totales", f"{df_filtre['ventes'].sum():,}")
        st.dataframe(df_filtre.head(10))
        
        # Graphique simple
        ventes_par_produit = df_filtre.groupby('produit')['ventes'].sum()
        st.bar_chart(ventes_par_produit)
    else:
        st.warning("Aucune donnée avec ces filtres")

# Utiliser le fragment
section_filtres()

# Stats globales (ne change pas avec les filtres)
st.subheader("📈 Statistiques globales")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total général", f"{df['ventes'].sum():,}")
with col2:
    st.metric("Moyenne", f"{df['ventes'].mean():.0f}")
with col3:
    st.metric("Nombre de produits", len(df['produit'].unique()))
""")

# Implémentation de l'exercice
st.markdown("**🎯 Version fonctionnelle :**")

# 1. Cache pour les données
@st.cache_data
def generer_donnees_exercice(nb_lignes):
    time.sleep(0.5)  # Simulation plus rapide pour la démo
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=nb_lignes),
        'ventes': np.random.randint(100, 1000, nb_lignes),
        'produit': np.random.choice(['A', 'B', 'C'], nb_lignes),
        'region': np.random.choice(['Nord', 'Sud', 'Est', 'Ouest'], nb_lignes)
    })

# 2. Interface avec cache
st.markdown("#### 📊 Analyse des ventes - Version optimisée")

# Paramètres
param_col1, param_col2 = st.columns(2)
with param_col1:
    nb_lignes_ex = st.selectbox("Taille des données:", [50, 100, 200], key="ex_data_size")

with param_col2:
    if st.button("🔄 Recharger données", key="ex_reload"):
        st.cache_data.clear()
        st.success("Cache vidé !")

# 3. Chargement avec cache
with st.spinner("Chargement des données..."):
    df_ex = generer_donnees_exercice(nb_lignes_ex)

st.success(f"✅ {len(df_ex)} lignes chargées")

# 4. Fragment pour les filtres
@st.fragment
def section_filtres_exercice():
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        produits_ex = st.multiselect("Produits:", df_ex['produit'].unique(), 
                            default=df_ex['produit'].unique(),
                            key="ex_produits_filter")
    
    with filter_col2:
        regions_ex = st.multiselect("Régions:", df_ex['region'].unique(),
                           default=df_ex['region'].unique(),
                           key="ex_regions_filter")
    
    # Filtrage optimisé
    df_filtre_ex = df_ex[
        (df_ex['produit'].isin(produits_ex)) & 
        (df_ex['region'].isin(regions_ex))
    ]
    
    # Affichage
    if len(df_filtre_ex) > 0:
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Ventes filtrées", f"{df_filtre_ex['ventes'].sum():,}")
        with metric_col2:
            st.metric("Lignes filtrées", len(df_filtre_ex))
        
        # Tableau
        st.dataframe(df_filtre_ex.head(5), use_container_width=True)
        
        # Graphique simple
        if len(df_filtre_ex) > 0:
            ventes_par_produit_ex = df_filtre_ex.groupby('produit')['ventes'].sum()
            st.bar_chart(ventes_par_produit_ex)
    else:
        st.warning("Aucune donnée avec ces filtres")

# Utiliser le fragment
section_filtres_exercice()

# Stats globales
st.markdown("**📈 Statistiques globales (non filtrées)**")
global_col1, global_col2, global_col3 = st.columns(3)

with global_col1:
    st.metric("Total général", f"{df_ex['ventes'].sum():,}")
with global_col2:
    st.metric("Moyenne", f"{df_ex['ventes'].mean():.0f}")
with global_col3:
    st.metric("Nombre de produits", len(df_ex['produit'].unique()))

st.divider()

# ================================
# 9. RÉCAPITULATIF
# ================================
st.subheader("9️⃣ Récapitulatif")

st.markdown("""
**🎓 Ce que vous avez appris :**

✅ **Cache de base :** `@st.cache_data` pour éviter les recalculs  
✅ **Cache données :** Optimiser le chargement de fichiers  
✅ **Lazy loading :** Charger seulement quand nécessaire  
✅ **Fragments :** `@st.fragment` pour les sections indépendantes  
✅ **Gestion mémoire :** `st.cache_data.clear()` et limites  
✅ **Bonnes pratiques :** Conseils pour applications fluides  

**💡 Règles d'or :**
1. **Cache tout ce qui est lent** (calculs, chargements)
2. **Chargez progressivement** (tabs, boutons)
3. **Filtrez avant d'afficher** (gros datasets)
4. **Utilisez des fragments** (sections indépendantes)
5. **Nettoyez régulièrement** (mémoire)

**🚀 Prochaine étape :** T_09 - Astuces et bonnes pratiques
""")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⬅️ Module précédent (T_07)", key="nav_prev_t8"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_07 - Avancé")

with col2:
    if st.button("📚 Retour au sommaire", key="nav_home_t8"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_00 - Sommaire")

with col3:
    if st.button("➡️ Module suivant (T_09)", key="nav_next_t8"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_09 - Astuces")

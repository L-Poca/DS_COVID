import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

st.header("📐 T_05 - Layout & Organisation Avancée")

st.markdown("**📋 Objectif :** Maîtriser l'organisation et la structuration professionnelle d'interfaces utilisateur avec colonnes, conteneurs, onglets, et techniques de mise en page avancées pour créer des applications ergonomiques et visuellement attractives.")

st.markdown("---")

# ================================
# 1. COLONNES AVANCÉES ET GRILLES RESPONSIVES
# ================================
st.subheader("1️⃣ Colonnes avancées et grilles responsives")

st.markdown("""
**📖 Description :**
Le système de colonnes Streamlit offre une flexibilité remarquable pour créer des interfaces professionnelles.
Au-delà de la simple division horizontale, vous pouvez contrôler les proportions, créer des grilles complexes,
et adapter le layout selon le contenu pour une expérience utilisateur optimale.

**🎯 Techniques avancées :**
- **Proportions personnalisées** : `st.columns([3, 1, 2])` pour un contrôle précis
- **Colonnes imbriquées** : Grilles dans grilles pour layouts complexes
- **Colonnes conditionnelles** : Affichage adaptatif selon le contexte
- **Séparateurs visuels** : Espacements et délimiteurs pour clarté
- **Alignement et centrage** : Contrôle de la position du contenu

**💡 Bonnes pratiques :**
- Utilisez les proportions pour hiérarchiser l'information (3:1 pour principal:secondaire)
- Limitez à 4 colonnes maximum pour éviter l'encombrement
- Privilégiez l'asymétrie équilibrée (2:3:1) plutôt que l'uniformité
- Testez sur différentes tailles d'écran pour la responsivité
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# 1. Colonnes proportionnelles sophistiquées
main_col, sidebar_col, info_col = st.columns([5, 2, 3])

with main_col:
    st.markdown("**📊 Contenu Principal**")
    # Graphique ou données principales
    chart_data = pd.DataFrame({
        'Ventes': [100, 120, 140, 110, 160],
        'Mois': ['Jan', 'Fév', 'Mar', 'Avr', 'Mai']
    })
    st.bar_chart(chart_data.set_index('Mois'))

with sidebar_col:
    st.markdown("**⚙️ Contrôles**")
    metric_choice = st.selectbox("Métrique", ["Ventes", "Profit"])
    time_period = st.select_slider("Période", 
        options=['1M', '3M', '6M', '1Y'], value='3M')

with info_col:
    st.markdown("**📈 KPIs**")
    st.metric("Total", "2,540€", "+15%")
    st.metric("Moyenne", "508€", "+8%")

# 2. Grilles imbriquées pour dashboards
st.markdown("**🔧 Layout Complexe**")
header_section = st.columns(1)[0]
header_section.markdown("### Dashboard Executive")

# Ligne de métriques
metrics_row = st.columns(4)
metrics_data = [
    ("Revenus", "125K€", "+12%", "🔹"),
    ("Clients", "1,234", "+5%", "👥"), 
    ("Conversion", "3.4%", "+0.8%", "📈"),
    ("Satisfaction", "4.8/5", "+0.2", "⭐")
]

for idx, (label, value, delta, icon) in enumerate(metrics_data):
    with metrics_row[idx]:
        st.metric(f"{icon} {label}", value, delta)

# Section graphiques avec colonnes asymétriques
chart_left, chart_right = st.columns([7, 3])

with chart_left:
    st.markdown("**📊 Évolution Temporelle**")
    # Simulation données temporelles
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    ts_data = pd.DataFrame({
        'Date': dates,
        'Valeur': np.cumsum(np.random.randn(30)) + 100
    })
    st.line_chart(ts_data.set_index('Date'))

with chart_right:
    st.markdown("**🎯 Répartition**")
    pie_data = pd.DataFrame({
        'Catégorie': ['A', 'B', 'C', 'D'],
        'Valeur': [30, 25, 25, 20]
    })
    fig = px.pie(pie_data, values='Valeur', names='Catégorie')
    st.plotly_chart(fig, use_container_width=True)

# 3. Colonnes conditionnelles et responsives
display_mode = st.radio("Mode d'affichage:", 
    ["Compact", "Standard", "Détaillé"], horizontal=True)

if display_mode == "Compact":
    single_col = st.columns(1)[0]
    with single_col:
        st.info("Mode compact : une seule colonne")
elif display_mode == "Standard": 
    left, right = st.columns(2)
    left.success("Mode standard : deux colonnes")
    right.success("Contenu équilibré")
else:  # Détaillé
    c1, c2, c3, c4 = st.columns(4)
    for i, col in enumerate([c1, c2, c3, c4]):
        col.info(f"Détail {i+1}")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # 1. Colonnes proportionnelles
    demo_main, demo_sidebar, demo_info = st.columns([5, 2, 3])
    
    with demo_main:
        st.markdown("**📊 Zone Principale**")
        demo_chart_data = pd.DataFrame({
            'Ventes': [100, 120, 140, 110, 160],
            'Mois': ['Jan', 'Fév', 'Mar', 'Avr', 'Mai']
        })
        st.bar_chart(demo_chart_data.set_index('Mois'), height=200)
    
    with demo_sidebar:
        st.markdown("**⚙️ Paramètres**")
        demo_metric = st.selectbox("Métrique", ["Ventes", "Profit"], key="demo_metric")
        demo_period = st.select_slider("Période", 
            options=['1M', '3M', '6M', '1Y'], value='3M', key="demo_period")
    
    with demo_info:
        st.markdown("**📈 Indicateurs**")
        st.metric("Total", "2,540€", "+15%")
        st.metric("Moyenne", "508€", "+8%")
    
    # 2. Métriques en ligne
    st.markdown("**📊 KPIs Executive**")
    demo_metrics = st.columns(4)
    demo_metrics_data = [
        ("Revenus", "125K€", "+12%"),
        ("Clients", "1,234", "+5%"), 
        ("Taux", "3.4%", "+0.8%"),
        ("Score", "4.8/5", "+0.2")
    ]
    
    for idx, (label, value, delta) in enumerate(demo_metrics_data):
        with demo_metrics[idx]:
            st.metric(label, value, delta)
    
    # 3. Mode d'affichage adaptatif
    demo_mode = st.radio("Test Responsive:", 
        ["Compact", "Standard", "Détaillé"], horizontal=True, key="demo_mode")
    
    if demo_mode == "Compact":
        demo_single = st.columns(1)[0]
        with demo_single:
            st.info("🔹 Mode compact activé")
    elif demo_mode == "Standard": 
        demo_left, demo_right = st.columns(2)
        demo_left.success("✅ Standard - Gauche")
        demo_right.success("✅ Standard - Droite")
    else:
        demo_c1, demo_c2, demo_c3, demo_c4 = st.columns(4)
        for i, col in enumerate([demo_c1, demo_c2, demo_c3, demo_c4]):
            col.info(f"Detail {i+1}")

st.divider()

# ================================
# 2. CONTENEURS ET GROUPEMENTS LOGIQUES
# ================================
st.subheader("2️⃣ Conteneurs et groupements logiques")

st.markdown("""
**📖 Description :**
Les conteneurs Streamlit permettent d'organiser le contenu en sections logiques et visuellement distinctes.
Ils offrent un contrôle granulaire sur l'affichage et permettent de créer des interfaces structurées
avec une hiérarchie claire de l'information.

**🎯 Types de conteneurs :**
- **`st.container()`** : Groupement logique invisible, contrôle programmatique
- **`st.expander()`** : Section collapsible pour organiser l'espace
- **`st.empty()`** : Placeholder dynamique pour contenu variable
- **Conteneurs avec bordures** : CSS personnalisé pour délimitation visuelle

**💡 Applications professionnelles :**
- Sections d'aide collapsibles (FAQ, documentation)
- Zones de configuration avancée (paramètres optionnels)
- Affichage conditionnel de contenu (résultats, alertes)
- Organisation modulaire d'applications complexes
- Mise à jour dynamique de zones spécifiques
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# 1. Conteneur logique avec contrôle programmatique
main_container = st.container()
with main_container:
    st.markdown("**🏗️ Section Principale**")
    user_input = st.text_input("Votre requête")
    
    if user_input:
        # Traitement conditionnel dans le conteneur
        st.success(f"Traitement: {user_input}")

# 2. Expanders pour organisation hiérarchique
with st.expander("🔧 Configuration Avancée", expanded=False):
    st.markdown("**Paramètres techniques:**")
    
    config_cols = st.columns(2)
    with config_cols[0]:
        api_key = st.text_input("Clé API", type="password")
        timeout = st.slider("Timeout (s)", 1, 30, 10)
    
    with config_cols[1]:
        debug_mode = st.checkbox("Mode Debug")
        log_level = st.selectbox("Log Level", 
            ["ERROR", "WARNING", "INFO", "DEBUG"])
    
    if st.button("💾 Sauvegarder Config"):
        st.success("Configuration sauvegardée!")

with st.expander("📊 Statistiques Détaillées"):
    st.markdown("**Analyse approfondie:**")
    
    # Simulation métriques avancées
    perf_data = {
        'Métrique': ['Latence', 'Débit', 'Erreurs', 'Disponibilité'],
        'Valeur': ['45ms', '1.2K rps', '0.1%', '99.9%'],
        'Statut': ['🟢', '🟡', '🟢', '🟢']
    }
    
    st.dataframe(pd.DataFrame(perf_data), hide_index=True)

# 3. Placeholders dynamiques pour contenu évolutif
status_placeholder = st.empty()
progress_placeholder = st.empty()

if st.button("🚀 Lancer Simulation"):
    for step in range(1, 6):
        status_placeholder.info(f"⏳ Étape {step}/5 en cours...")
        progress_placeholder.progress(step / 5)
        time.sleep(0.5)
    
    status_placeholder.success("✅ Simulation terminée!")
    progress_placeholder.empty()  # Nettoyer la barre

# 4. Conteneurs avec styling personnalisé
st.markdown('''
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    color: white;
">
    <h4>🎨 Conteneur Stylé</h4>
    <p>Design personnalisé avec CSS pour mise en valeur</p>
</div>
''', unsafe_allow_html=True)
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # 1. Conteneur principal
    demo_main_container = st.container()
    with demo_main_container:
        st.markdown("**🏗️ Zone de Travail**")
        demo_user_input = st.text_input("Tapez quelque chose", key="demo_input")
        
        if demo_user_input:
            st.success(f"✅ Reçu: {demo_user_input}")
    
    # 2. Expanders organisés
    with st.expander("🔧 Paramètres", expanded=False):
        demo_config_cols = st.columns(2)
        with demo_config_cols[0]:
            demo_api = st.text_input("API Key", type="password", key="demo_api")
            demo_timeout = st.slider("Timeout", 1, 30, 10, key="demo_timeout")
        
        with demo_config_cols[1]:
            demo_debug = st.checkbox("Debug", key="demo_debug")
            demo_log = st.selectbox("Log Level", 
                ["ERROR", "WARNING", "INFO"], key="demo_log")
    
    with st.expander("📈 Métriques Système"):
        demo_metrics_df = pd.DataFrame({
            'Composant': ['API', 'DB', 'Cache', 'Queue'],
            'Statut': ['🟢 OK', '🟢 OK', '🟡 Warning', '🟢 OK'],
            'Latence': ['12ms', '5ms', '89ms', '3ms']
        })
        st.dataframe(demo_metrics_df, hide_index=True)
    
    # 3. Placeholder dynamique
    demo_status = st.empty()
    demo_progress = st.empty()
    
    if st.button("🔄 Test Dynamique", key="demo_dynamic"):
        for i in range(1, 4):
            demo_status.info(f"⏳ Étape {i}/3...")
            demo_progress.progress(i / 3)
            time.sleep(0.3)
        
        demo_status.success("✅ Terminé!")
        demo_progress.empty()
    
    # 4. Conteneur stylé
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    ">
        <strong>🎨 Zone Mise en Valeur</strong><br>
        <small>Design personnalisé</small>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ================================
# 3. ONGLETS POUR NAVIGATION INTUITIVE
# ================================
st.subheader("3️⃣ Onglets pour navigation intuitive")

st.markdown("""
**📖 Description :**
Les onglets (`st.tabs()`) révolutionnent l'organisation du contenu en permettant une navigation
horizontale intuitive. Ils maximisent l'utilisation de l'espace tout en gardant le contenu
organisé et accessible, idéal pour les applications multi-fonctions.

**🎯 Avantages des onglets :**
- **Économie d'espace** : Plusieurs vues dans la même zone
- **Navigation familière** : UX similaire aux navigateurs web
- **Organisation logique** : Regroupement par thématique ou fonction
- **Réduction de scroll** : Contenu organisé horizontalement
- **Focus utilisateur** : Une section à la fois, moins de distraction

**💡 Stratégies d'organisation :**
- Onglets par étape de workflow (Config → Analyse → Résultats)
- Séparation par type de données (Données → Graphiques → Rapports)
- Organisation par audience (Utilisateur → Admin → Développeur)
- Division temporelle (Temps réel → Historique → Prévisions)
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Onglets thématiques pour application analytique
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", "🔍 Analyse", "⚙️ Configuration", "📄 Rapports"
])

with tab1:
    st.markdown("### 📊 Vue d'ensemble")
    
    # Métriques principales
    metrics_row = st.columns(3)
    with metrics_row[0]:
        st.metric("Chiffre d'affaires", "2.4M€", "+12%")
    with metrics_row[1]:
        st.metric("Commandes", "15,234", "+8%")
    with metrics_row[2]:
        st.metric("Clients actifs", "3,456", "+15%")
    
    # Graphique de tendance
    trend_data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=30),
        'Ventes': np.cumsum(np.random.randn(30)*100) + 10000
    })
    st.line_chart(trend_data.set_index('Date'))

with tab2:
    st.markdown("### 🔍 Analyse Avancée")
    
    analysis_cols = st.columns(2)
    
    with analysis_cols[0]:
        st.markdown("**Segmentation Clients**")
        segment_data = pd.DataFrame({
            'Segment': ['Premium', 'Standard', 'Basique'],
            'Pourcentage': [25, 45, 30],
            'Revenus': [800000, 1200000, 400000]
        })
        
        fig = px.pie(segment_data, values='Pourcentage', 
                    names='Segment', title="Répartition Clients")
        st.plotly_chart(fig, use_container_width=True)
    
    with analysis_cols[1]:
        st.markdown("**Performance Produits**")
        product_data = pd.DataFrame({
            'Produit': ['A', 'B', 'C', 'D', 'E'],
            'Ventes': [850, 720, 690, 480, 320],
            'Marge': [15, 22, 18, 12, 8]
        })
        
        fig2 = px.scatter(product_data, x='Ventes', y='Marge',
                         size='Ventes', hover_name='Produit')
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### ⚙️ Paramètres Système")
    
    config_sections = st.columns(2)
    
    with config_sections[0]:
        st.markdown("**🔐 Sécurité**")
        auth_method = st.selectbox("Authentification", 
            ["OAuth", "LDAP", "Local"])
        session_timeout = st.number_input("Session (min)", 5, 480, 60)
        
        st.markdown("**🔔 Notifications**")
        email_alerts = st.checkbox("Alertes email")
        sms_alerts = st.checkbox("Alertes SMS")
    
    with config_sections[1]:
        st.markdown("**📡 API & Intégrations**")
        api_rate_limit = st.slider("Limite API/min", 100, 10000, 1000)
        webhook_url = st.text_input("URL Webhook")
        
        st.markdown("**💾 Backup**")
        backup_frequency = st.selectbox("Fréquence", 
            ["Quotidien", "Hebdomadaire", "Mensuel"])

with tab4:
    st.markdown("### 📄 Génération de Rapports")
    
    report_cols = st.columns(2)
    
    with report_cols[0]:
        st.markdown("**📋 Configuration Rapport**")
        report_type = st.selectbox("Type de rapport", 
            ["Exécutif", "Technique", "Financier", "Opérationnel"])
        
        date_range = st.date_input("Période", 
            value=[datetime.now().date() - timedelta(days=30),
                   datetime.now().date()])
        
        include_charts = st.checkbox("Inclure graphiques", value=True)
        include_raw_data = st.checkbox("Données brutes")
    
    with report_cols[1]:
        st.markdown("**📤 Export & Distribution**")
        export_format = st.radio("Format", ["PDF", "Excel", "PowerPoint"])
        
        recipients = st.text_area("Destinataires (emails)", 
            placeholder="email1@company.com\\nemail2@company.com")
        
        if st.button("📧 Générer et Envoyer"):
            st.success("Rapport généré et envoyé avec succès!")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Onglets de démonstration
    demo_tab1, demo_tab2, demo_tab3 = st.tabs([
        "📊 Vue", "🔍 Analyse", "⚙️ Config"
    ])
    
    with demo_tab1:
        st.markdown("**📈 Tableau de Bord**")
        
        demo_metrics_row = st.columns(3)
        with demo_metrics_row[0]:
            st.metric("CA", "2.4M€", "+12%")
        with demo_metrics_row[1]:
            st.metric("Orders", "15K", "+8%")
        with demo_metrics_row[2]:
            st.metric("Users", "3.5K", "+15%")
        
        # Mini graphique
        demo_trend = pd.DataFrame({
            'Jour': range(1, 8),
            'Ventes': [100, 120, 140, 130, 160, 180, 200]
        })
        st.line_chart(demo_trend.set_index('Jour'), height=200)
    
    with demo_tab2:
        st.markdown("**🔍 Insights**")
        
        demo_analysis_cols = st.columns(2)
        
        with demo_analysis_cols[0]:
            st.markdown("*Segments:*")
            demo_segments = pd.DataFrame({
                'Type': ['Premium', 'Standard'],
                'Part': [35, 65]
            })
            fig = px.pie(demo_segments, values='Part', names='Type')
            st.plotly_chart(fig, use_container_width=True, height=200)
        
        with demo_analysis_cols[1]:
            st.markdown("*Top Produits:*")
            demo_products = pd.DataFrame({
                'Produit': ['A', 'B', 'C'],
                'Ventes': [850, 720, 690]
            })
            st.bar_chart(demo_products.set_index('Produit'), height=200)
    
    with demo_tab3:
        st.markdown("**⚙️ Paramètres**")
        
        demo_config_cols = st.columns(2)
        
        with demo_config_cols[0]:
            st.markdown("*Sécurité:*")
            demo_auth = st.selectbox("Auth", ["OAuth", "LDAP"], key="demo_auth")
            demo_session = st.slider("Session", 5, 120, 30, key="demo_session")
        
        with demo_config_cols[1]:
            st.markdown("*Notifications:*")
            demo_email = st.checkbox("Email", key="demo_email")
            demo_sms = st.checkbox("SMS", key="demo_sms")

st.divider()

# ================================
# 4. SIDEBAR POUR NAVIGATION LATÉRALE
# ================================
st.subheader("4️⃣ Sidebar pour navigation et contrôles globaux")

st.markdown("""
**📖 Description :**
La sidebar (`st.sidebar`) offre un espace de navigation permanent et accessible,
idéal pour les contrôles globaux, la navigation entre sections, et les paramètres
qui doivent rester visibles en permanence.

**🎯 Utilisations optimales de la sidebar :**
- **Navigation principale** : Menu de sections, liens vers pages
- **Filtres globaux** : Sélections qui affectent toute l'application
- **Paramètres utilisateur** : Préférences, configuration personnelle
- **État de session** : Informations de connexion, statut système
- **Actions rapides** : Boutons d'export, refresh, aide

**💡 Bonnes pratiques sidebar :**
- Gardez le contenu essentiel et permanent
- Organisez par ordre de fréquence d'utilisation (plus utilisé en haut)
- Utilisez des groupes visuels avec `st.sidebar.divider()`
- Limitez le scroll dans la sidebar
- Ajoutez des tooltips pour les icônes ou termes techniques
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Configuration de la sidebar avec sections organisées

# 1. Header avec informations utilisateur
st.sidebar.markdown("### 👤 Session Utilisateur")
st.sidebar.info("👋 Bienvenue, Admin!")
st.sidebar.metric("Dernière connexion", "Il y a 2h")

st.sidebar.divider()

# 2. Navigation principale
st.sidebar.markdown("### 🧭 Navigation")
page_selection = st.sidebar.selectbox(
    "Aller à la page:",
    ["Dashboard", "Analytics", "Reports", "Settings", "Help"]
)

# Navigation par radio buttons pour sections
section = st.sidebar.radio(
    "Section active:",
    ["Vue d'ensemble", "Détails", "Configuration"],
    index=0
)

st.sidebar.divider()

# 3. Filtres globaux appliqués à toute l'app
st.sidebar.markdown("### 🔍 Filtres Globaux")

# Filtre temporel
date_filter = st.sidebar.date_input(
    "Période d'analyse",
    value=[datetime.now().date() - timedelta(days=30),
           datetime.now().date()]
)

# Filtre par catégorie
category_filter = st.sidebar.multiselect(
    "Catégories:",
    ["Ventes", "Marketing", "Support", "Produit"],
    default=["Ventes", "Marketing"]
)

# Filtre numérique avec slider
threshold_filter = st.sidebar.slider(
    "Seuil de performance", 
    0, 100, 75, 
    help="Afficher uniquement les éléments au-dessus de ce seuil"
)

st.sidebar.divider()

# 4. Paramètres et préférences
st.sidebar.markdown("### ⚙️ Préférences")

# Mode d'affichage
display_mode = st.sidebar.selectbox(
    "Mode d'affichage:",
    ["Clair", "Sombre", "Auto"]
)

# Options d'affichage
show_tooltips = st.sidebar.checkbox("Afficher les tooltips", value=True)
auto_refresh = st.sidebar.checkbox("Actualisation auto")

if auto_refresh:
    refresh_interval = st.sidebar.selectbox(
        "Intervalle (sec):",
        [30, 60, 300, 600]
    )

st.sidebar.divider()

# 5. Actions rapides et outils
st.sidebar.markdown("### 🛠️ Actions Rapides")

# Boutons d'action avec colonnes pour compacité
action_col1, action_col2 = st.sidebar.columns(2)

with action_col1:
    if st.button("📁 Export"):
        st.sidebar.success("Export lancé!")

with action_col2:
    if st.button("🔄 Refresh"):
        st.sidebar.info("Données actualisées")

# Actions moins fréquentes
if st.sidebar.button("🧹 Nettoyer Cache"):
    st.sidebar.warning("Cache nettoyé")

if st.sidebar.button("❓ Aide & Support"):
    st.sidebar.info("Centre d'aide ouvert")

# 6. Informations système en bas
st.sidebar.divider()
st.sidebar.markdown("### 📊 État Système")
st.sidebar.text("🟢 Tous les services opérationnels")
st.sidebar.text(f"⚡ Version: 2.1.3")
st.sidebar.text(f"🕐 Dernière MAJ: {datetime.now().strftime('%H:%M')}")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    st.info("👈 **Regardez la sidebar à gauche** pour voir le résultat de ce code en action!")
    
    st.markdown("""
    **Éléments visibles dans la sidebar :**
    
    🔹 **Section Utilisateur** : Informations de session et bienvenue
    
    🔹 **Navigation** : Sélecteurs pour changer de page/section
    
    🔹 **Filtres Globaux** : 
    - Filtre de dates pour la période d'analyse
    - Sélecteur multiple pour catégories
    - Slider pour seuil de performance
    
    🔹 **Préférences** : 
    - Mode d'affichage (Clair/Sombre)
    - Options d'interface (tooltips, auto-refresh)
    
    🔹 **Actions Rapides** : 
    - Boutons Export et Refresh en colonnes
    - Actions moins fréquentes (cache, aide)
    
    🔹 **État Système** : Informations techniques en bas
    """)
    
    # Affichage conditionnel basé sur les sélections sidebar
    if 'sidebar_demo_initialized' not in st.session_state:
        st.session_state.sidebar_demo_initialized = True
    
    # Simulation de contenu réactif aux contrôles sidebar
    st.markdown("**📱 Contenu Réactif aux Contrôles Sidebar :**")
    
    # Ces valeurs changeraient selon les vrais contrôles sidebar
    demo_info_col1, demo_info_col2 = st.columns(2)
    
    with demo_info_col1:
        st.metric("Filtres Actifs", "3")
        st.metric("Page Active", "Dashboard")
    
    with demo_info_col2:
        st.metric("Mode Affichage", "Clair")
        st.metric("Auto-Refresh", "Activé")

# Simulation réelle de sidebar pour cette démonstration
with st.sidebar:
    st.markdown("### 🎯 Démo Layout T_05")
    
    st.markdown("**Navigation :**")
    demo_page = st.selectbox("Page", ["Layout", "Widgets", "Data"], key="sidebar_demo_page")
    
    st.divider()
    
    st.markdown("**Filtres :**")
    demo_category = st.multiselect("Catégories", ["A", "B", "C"], default=["A"], key="sidebar_demo_cat")
    demo_threshold = st.slider("Seuil", 0, 100, 50, key="sidebar_demo_threshold")
    
    st.divider()
    
    st.markdown("**Actions :**")
    if st.button("🔄 Actualiser", key="sidebar_demo_refresh"):
        st.success("✅ Actualisé!")

st.divider()

# ================================
# 5. RESPONSIVE DESIGN ET ADAPTABILITÉ
# ================================
st.subheader("5️⃣ Responsive design et adaptabilité")

st.markdown("""
**📖 Description :**
Le responsive design garantit une expérience utilisateur optimale sur tous les appareils
et tailles d'écran. Streamlit offre des outils pour créer des interfaces qui s'adaptent
intelligemment au contexte d'affichage, de l'écran large au mobile.

**🎯 Techniques de responsive design :**
- **Colonnes adaptatives** : Proportions qui s'ajustent selon l'espace
- **Largeur de conteneur** : `use_container_width=True` pour graphiques
- **Breakpoints conditionnels** : Affichage différent selon la largeur
- **Navigation adaptive** : Menu hamburger vs barre horizontale
- **Métriques empilables** : Passage de ligne à colonne selon l'espace

**💡 Stratégies d'optimisation :**
- Testez sur différentes résolutions (mobile, tablette, desktop)
- Privilégiez les proportions relatives aux tailles fixes
- Utilisez `st.columns()` avec ratios adaptatifs
- Minimisez le scroll horizontal sur petit écran
- Gardez les actions importantes accessibles même sur mobile
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# 1. Colonnes adaptatives selon le contenu
def create_responsive_metrics(metrics_data, mobile_view=False):
    if mobile_view:
        # Mode mobile : une colonne
        for label, value, delta in metrics_data:
            st.metric(label, value, delta)
    else:
        # Mode desktop : plusieurs colonnes
        cols = st.columns(len(metrics_data))
        for idx, (label, value, delta) in enumerate(metrics_data):
            cols[idx].metric(label, value, delta)

# Simulation détection mobile (en réalité via JS)
mobile_simulation = st.checkbox("Simuler affichage mobile")

metrics = [
    ("Revenus", "125K€", "+12%"),
    ("Clients", "2,340", "+8%"),
    ("Conversion", "4.2%", "+0.5%"),
    ("Satisfaction", "4.8⭐", "+0.2")
]

create_responsive_metrics(metrics, mobile_simulation)

# 2. Layout adaptatif pour dashboards
view_mode = st.selectbox("Mode d'affichage:", 
    ["Compact", "Standard", "Étendu"])

if view_mode == "Compact":
    # Tout en une colonne pour espaces réduits
    st.markdown("**📊 Vue Compacte**")
    
    # Métriques empilées
    for label, value, delta in metrics[:2]:  # Limiter pour gain de place
        st.metric(label, value, delta)
    
    # Graphique simplifié
    simple_data = pd.DataFrame({
        'x': [1, 2, 3, 4],
        'y': [10, 15, 13, 17]
    })
    st.line_chart(simple_data.set_index('x'), height=200)

elif view_mode == "Standard":
    # Layout équilibré pour usage normal
    st.markdown("**📊 Vue Standard**")
    
    # Métriques en 2x2
    metric_rows = [metrics[:2], metrics[2:]]
    for row in metric_rows:
        cols = st.columns(len(row))
        for idx, (label, value, delta) in enumerate(row):
            cols[idx].metric(label, value, delta)
    
    # Graphiques côte à côte
    chart_left, chart_right = st.columns(2)
    
    with chart_left:
        st.markdown("*Tendance:*")
        trend_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=7),
            'Valeur': [100, 110, 105, 120, 115, 130, 125]
        })
        st.line_chart(trend_data.set_index('Date'), height=250)
    
    with chart_right:
        st.markdown("*Répartition:*")
        pie_data = pd.DataFrame({
            'Segment': ['A', 'B', 'C'],
            'Part': [40, 35, 25]
        })
        fig = px.pie(pie_data, values='Part', names='Segment')
        st.plotly_chart(fig, use_container_width=True, height=250)

else:  # Étendu
    # Layout large pour grands écrans
    st.markdown("**📊 Vue Étendue**")
    
    # Toutes les métriques en ligne
    metric_cols = st.columns(len(metrics))
    for idx, (label, value, delta) in enumerate(metrics):
        metric_cols[idx].metric(label, value, delta)
    
    # Dashboard 3 colonnes
    dash_left, dash_center, dash_right = st.columns([2, 3, 2])
    
    with dash_left:
        st.markdown("**🎯 KPIs Clés**")
        st.success("✅ Objectifs atteints")
        st.info("📈 Croissance: +15%")
        st.warning("⚠️ Attention Stock")
    
    with dash_center:
        st.markdown("**📊 Analyse Principale**")
        extended_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=30),
            'Ventes': np.cumsum(np.random.randn(30)*10) + 1000,
            'Profit': np.cumsum(np.random.randn(30)*5) + 200
        })
        st.line_chart(extended_data.set_index('Date'), height=300)
    
    with dash_right:
        st.markdown("**📋 Actions**")
        st.button("📄 Rapport Détaillé", use_container_width=True)
        st.button("📧 Envoyer Résumé", use_container_width=True)
        st.button("⚙️ Configuration", use_container_width=True)

# 3. Navigation responsive
nav_mode = st.radio("Navigation:", ["Horizontale", "Verticale"], horizontal=True)

if nav_mode == "Horizontale":
    # Menu horizontal pour grands écrans
    nav_cols = st.columns(5)
    nav_items = ["🏠 Home", "📊 Analytics", "📈 Reports", "⚙️ Settings", "❓ Help"]
    
    for idx, item in enumerate(nav_items):
        if nav_cols[idx].button(item, use_container_width=True):
            st.success(f"Navigation vers: {item}")

else:
    # Menu vertical pour petits écrans ou sidebar
    st.markdown("**📱 Menu Vertical (Mobile-friendly):**")
    nav_items = ["🏠 Accueil", "📊 Données", "📈 Rapports", "⚙️ Paramètres"]
    
    for item in nav_items:
        if st.button(item, use_container_width=True):
            st.success(f"Sélectionné: {item}")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Démonstration responsive
    st.markdown("**📱 Test Responsive Design**")
    
    # Simulation mobile
    demo_mobile = st.checkbox("Mode Mobile", key="demo_mobile_view")
    
    # Métriques adaptatives
    demo_metrics = [
        ("CA", "125K€", "+12%"),
        ("Clients", "2.3K", "+8%"),
        ("Conv", "4.2%", "+0.5%")
    ]
    
    if demo_mobile:
        st.info("📱 Affichage Mobile: Métriques empilées")
        for label, value, delta in demo_metrics:
            st.metric(label, value, delta)
    else:
        st.info("💻 Affichage Desktop: Métriques en ligne")
        demo_metric_cols = st.columns(len(demo_metrics))
        for idx, (label, value, delta) in enumerate(demo_metrics):
            demo_metric_cols[idx].metric(label, value, delta)
    
    # Layout adaptatif
    demo_view = st.selectbox("Vue Dashboard:", 
        ["Compacte", "Standard", "Étendue"], key="demo_view")
    
    if demo_view == "Compacte":
        st.markdown("*Vue Compacte:*")
        demo_compact_data = pd.DataFrame({
            'Jour': [1, 2, 3, 4, 5],
            'Ventes': [100, 120, 110, 140, 130]
        })
        st.line_chart(demo_compact_data.set_index('Jour'), height=150)
    
    elif demo_view == "Standard":
        st.markdown("*Vue Standard:*")
        demo_std_left, demo_std_right = st.columns(2)
        
        with demo_std_left:
            st.metric("Total", "500K€")
        with demo_std_right:
            st.metric("Moyenne", "25K€")
    
    else:  # Étendue
        st.markdown("*Vue Étendue:*")
        demo_ext_cols = st.columns(3)
        
        with demo_ext_cols[0]:
            st.metric("Revenus", "500K€")
        with demo_ext_cols[1]:
            st.metric("Coûts", "300K€")
        with demo_ext_cols[2]:
            st.metric("Profit", "200K€")

st.markdown("---")

st.success("🎉 **Félicitations !** Vous maîtrisez maintenant l'organisation et la structuration d'interfaces Streamlit professionnelles !")

st.markdown("""
**🚀 Points clés à retenir :**

**📐 Colonnes et Grilles :**
- Utilisez les proportions pour hiérarchiser (ex: [3, 1, 2])
- Imbriquez les colonnes pour des layouts complexes
- Adaptez le nombre de colonnes selon le contenu

**📦 Conteneurs :**
- `st.container()` pour groupement logique
- `st.expander()` pour économiser l'espace
- `st.empty()` pour contenu dynamique

**📑 Onglets :**
- Organisez par thématique ou workflow
- Maximisez l'utilisation de l'espace
- Navigation intuitive et familière

**🎯 Sidebar :**
- Réservez aux contrôles globaux permanents
- Organisez par fréquence d'utilisation
- Gardez accessible l'essentiel

**📱 Responsive Design :**
- Testez sur différentes résolutions
- Utilisez des proportions adaptatives
- Privilégiez la flexibilité aux tailles fixes

**🔗 Prochaine étape :** Explorez T_06_Fichiers pour maîtriser la gestion des fichiers et l'import/export de données !
""")

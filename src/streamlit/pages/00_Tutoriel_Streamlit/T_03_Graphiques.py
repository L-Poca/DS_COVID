import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.header("📊 T_03 - Graphiques & Visualisation")

st.markdown("**📋 Objectif :** Maîtriser toutes les techniques de visualisation pour créer des graphiques impactants, interactifs et professionnels avec Streamlit.")

st.markdown("---")

# ================================
# 1. GRAPHIQUES STREAMLIT NATIFS
# ================================
st.subheader("1️⃣ Graphiques Streamlit natifs")

st.markdown("""
**📖 Description :**
Streamlit offre des fonctions graphiques intégrées ultra-simples à utiliser.
Ces graphiques sont automatiquement responsives, interactifs (zoom, pan) et 
s'adaptent parfaitement au thème de votre application.

**🎯 Avantages des graphiques natifs :**
- Syntaxe ultra-simple (une seule ligne de code)
- Interactivité automatique (zoom, panoramique, sélection)
- Thème cohérent avec l'application
- Performance optimisée pour Streamlit
- Responsive design automatique

**💡 Quand les utiliser :**
- Prototypage rapide de visualisations
- Dashboards simples et efficaces
- Quand la simplicité prime sur la personnalisation
- Exploration de données en temps réel
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Données de démonstration
dates = pd.date_range('2024-01-01', periods=30, freq='D')
data_trend = pd.DataFrame({
    'date': dates,
    'ventes': np.random.randint(100, 300, 30) + np.arange(30) * 2,
    'profits': np.random.randint(20, 80, 30) + np.arange(30) * 1.5
})
data_trend.set_index('date', inplace=True)

# 1. LINE CHART - Évolution temporelle
st.markdown("**📈 Évolution des ventes (Line Chart)**")
st.line_chart(data_trend[['ventes']], 
             color=['#FF6B6B'])  # Couleur personnalisée

# 2. AREA CHART - Volume cumulé
st.markdown("**📊 Volume cumulé (Area Chart)**")
st.area_chart(data_trend)

# 3. BAR CHART - Comparaisons
categories_data = pd.DataFrame({
    'Q1': [1200, 800],
    'Q2': [1500, 900], 
    'Q3': [1100, 1200],
    'Q4': [1800, 1000]
}, index=['Ventes', 'Coûts'])

st.markdown("**📊 Comparaison trimestrielle (Bar Chart)**")
st.bar_chart(categories_data.T)  # Transposer pour bon affichage

# 4. SCATTER CHART - Relations entre variables
np.random.seed(42)  # Pour reproductibilité
scatter_data = pd.DataFrame({
    'prix': np.random.uniform(10, 100, 50),
    'qualite': np.random.uniform(1, 10, 50) 
})
# Ajouter corrélation logique
scatter_data['satisfaction'] = (
    scatter_data['qualite'] * 0.6 + 
    (100 - scatter_data['prix']) * 0.01 + 
    np.random.normal(0, 1, 50)
)

st.markdown("**🎯 Relation Prix-Qualité-Satisfaction**")
st.scatter_chart(data=scatter_data, x='prix', y='qualite', 
                size='satisfaction', color='satisfaction')
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Génération des données
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    data_trend = pd.DataFrame({
        'date': dates,
        'ventes': np.random.randint(100, 300, 30) + np.arange(30) * 2,
        'profits': np.random.randint(20, 80, 30) + np.arange(30) * 1.5
    })
    data_trend.set_index('date', inplace=True)
    
    # Line Chart
    st.markdown("**📈 Évolution des ventes**")
    st.line_chart(data_trend[['ventes']], color=['#FF6B6B'])
    
    # Area Chart
    st.markdown("**📊 Volume cumulé**")
    st.area_chart(data_trend)
    
    # Bar Chart
    categories_data = pd.DataFrame({
        'Q1': [1200, 800],
        'Q2': [1500, 900], 
        'Q3': [1100, 1200],
        'Q4': [1800, 1000]
    }, index=['Ventes', 'Coûts'])
    
    st.markdown("**📊 Comparaison trimestrielle**")
    st.bar_chart(categories_data.T)
    
    # Scatter Chart
    np.random.seed(42)
    scatter_data = pd.DataFrame({
        'prix': np.random.uniform(10, 100, 50),
        'qualite': np.random.uniform(1, 10, 50) 
    })
    scatter_data['satisfaction'] = (
        scatter_data['qualite'] * 0.6 + 
        (100 - scatter_data['prix']) * 0.01 + 
        np.random.normal(0, 1, 50)
    )
    
    st.markdown("**🎯 Relation Prix-Qualité-Satisfaction**")
    st.scatter_chart(data=scatter_data, x='prix', y='qualite', 
                    size='satisfaction', color='satisfaction')

st.markdown("""
**🔍 Caractéristiques techniques :**
- **Line Chart** : Parfait pour les séries temporelles, tendances, évolutions
- **Area Chart** : Montre le volume et les proportions, empilage automatique
- **Bar Chart** : Comparaisons entre catégories, valeurs discrètes
- **Scatter Chart** : Relations entre variables, corrélations, clustering
- **Interactivité** : Zoom molette, pan clic-glisser, reset double-clic
- **Responsive** : S'adapte automatiquement à la taille de l'écran
""")

st.divider()

# ================================
# 2. MATPLOTLIB INTÉGRATION AVANCÉE
# ================================
st.subheader("2️⃣ Matplotlib - Graphiques personnalisés")

st.markdown("""
**📖 Description :**
Matplotlib offre un contrôle total sur l'apparence et le comportement de vos graphiques.
C'est la référence en Python pour les visualisations scientifiques et publications.
Parfait quand vous avez besoin de personnalisation poussée ou de types de graphiques spécifiques.

**🎯 Avantages de Matplotlib :**
- Contrôle total sur chaque élément visuel
- Qualité publication scientifique
- Types de graphiques spécialisés (polaires, 3D, etc.)
- Styles et thèmes prédéfinis
- Compatibilité avec toutes les bibliothèques scientifiques

**💡 Cas d'usage :**
- Rapports scientifiques et techniques
- Graphiques pour publications
- Visualisations complexes multi-axes
- Graphiques avec annotations détaillées
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Configuration du style Matplotlib
plt.style.use('seaborn-v0_8')  # Style moderne
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# 1. GRAPHIQUE MULTI-AXES avec annotations
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Données financières simulées
x = np.linspace(0, 12, 100)
prix_action = 100 + 20 * np.sin(x) + np.random.normal(0, 5, 100)
volume = 1000 + 500 * np.cos(x * 1.5) + np.random.normal(0, 100, 100)

# Graphique 1: Prix de l'action avec zones
ax1.plot(x, prix_action, color='#2E86C1', linewidth=2, label='Prix Action')
ax1.fill_between(x, prix_action, alpha=0.3, color='#2E86C1')
ax1.axhline(y=np.mean(prix_action), color='red', linestyle='--', 
           label=f'Moyenne: {np.mean(prix_action):.1f}€')
ax1.set_title('📈 Évolution du Prix de l\'Action', fontsize=14, fontweight='bold')
ax1.set_ylabel('Prix (€)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Graphique 2: Volume avec barres colorées
colors = ['green' if v > np.mean(volume) else 'red' for v in volume]
ax2.bar(x, volume, color=colors, alpha=0.7, width=0.1)
ax2.set_title('📊 Volume des Échanges', fontsize=14, fontweight='bold')
ax2.set_xlabel('Temps (mois)')
ax2.set_ylabel('Volume')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# 2. GRAPHIQUE POLAIRE (Radar Chart)
fig_polar, ax_polar = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

# Données de performance (compétences)
categories = ['Python', 'SQL', 'Machine Learning', 'Statistiques', 'Visualisation', 'Communication']
values_user = [8, 7, 6, 8, 9, 7]
values_expert = [9, 9, 9, 9, 8, 8]

# Angles pour chaque catégorie
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
values_user += values_user[:1]  # Fermer le polygone
values_expert += values_expert[:1]
angles += angles[:1]

# Tracer les polygones
ax_polar.plot(angles, values_user, 'o-', linewidth=2, label='Votre niveau', color='#E74C3C')
ax_polar.fill(angles, values_user, alpha=0.25, color='#E74C3C')
ax_polar.plot(angles, values_expert, 'o-', linewidth=2, label='Expert', color='#27AE60')
ax_polar.fill(angles, values_expert, alpha=0.15, color='#27AE60')

# Personnalisation
ax_polar.set_xticks(angles[:-1])
ax_polar.set_xticklabels(categories)
ax_polar.set_ylim(0, 10)
ax_polar.set_title('🎯 Radar des Compétences', fontsize=16, fontweight='bold', pad=20)
ax_polar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

st.pyplot(fig_polar)
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Configuration du style
    plt.style.use('default')  # Style par défaut plus compatible
    
    # 1. Graphique multi-axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Données financières
    x = np.linspace(0, 12, 100)
    prix_action = 100 + 20 * np.sin(x) + np.random.normal(0, 5, 100)
    volume = 1000 + 500 * np.cos(x * 1.5) + np.random.normal(0, 100, 100)
    
    # Prix de l'action
    ax1.plot(x, prix_action, color='#2E86C1', linewidth=2, label='Prix Action')
    ax1.fill_between(x, prix_action, alpha=0.3, color='#2E86C1')
    ax1.axhline(y=np.mean(prix_action), color='red', linestyle='--', 
               label=f'Moyenne: {np.mean(prix_action):.1f}€')
    ax1.set_title('📈 Évolution du Prix de l\'Action', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Prix (€)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volume
    colors = ['green' if v > np.mean(volume) else 'red' for v in volume]
    ax2.bar(x, volume, color=colors, alpha=0.7, width=0.1)
    ax2.set_title('📊 Volume des Échanges', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Temps (mois)')
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 2. Graphique polaire
    fig_polar, ax_polar = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    categories = ['Python', 'SQL', 'ML', 'Stats', 'Dataviz', 'Communication']
    values_user = [8, 7, 6, 8, 9, 7]
    values_expert = [9, 9, 9, 9, 8, 8]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_user += values_user[:1]
    values_expert += values_expert[:1]
    angles += angles[:1]
    
    ax_polar.plot(angles, values_user, 'o-', linewidth=2, label='Votre niveau', color='#E74C3C')
    ax_polar.fill(angles, values_user, alpha=0.25, color='#E74C3C')
    ax_polar.plot(angles, values_expert, 'o-', linewidth=2, label='Expert', color='#27AE60')
    ax_polar.fill(angles, values_expert, alpha=0.15, color='#27AE60')
    
    ax_polar.set_xticks(angles[:-1])
    ax_polar.set_xticklabels(categories)
    ax_polar.set_ylim(0, 10)
    ax_polar.set_title('🎯 Radar des Compétences', fontsize=16, fontweight='bold', pad=20)
    ax_polar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    st.pyplot(fig_polar)

st.markdown("""
**🔍 Fonctionnalités avancées Matplotlib :**
- **Subplots** : Créer plusieurs graphiques dans une figure (`plt.subplots()`)
- **Styles** : Thèmes prédéfinis (`plt.style.use()`)
- **Annotations** : Textes, flèches, zones (`annotate()`, `axhline()`)
- **Projections** : Polaire, 3D (`subplot_kw=dict(projection='polar')`)
- **Couleurs conditionnelles** : Barres colorées selon les valeurs
- **Légendes avancées** : Positionnement personnalisé (`bbox_to_anchor`)
""")

st.divider()

# ================================
# 3. PLOTLY - INTERACTIVITÉ AVANCÉE
# ================================
st.subheader("3️⃣ Plotly - Graphiques interactifs professionnels")

st.markdown("""
**📖 Description :**
Plotly est la référence pour les graphiques interactifs en Python.
Il offre une interactivité riche (hover, click, zoom, sélection) et des graphiques 
de qualité publication directement dans le navigateur.

**🎯 Superpuissances de Plotly :**
- Interactivité native (hover, zoom, pan, sélection)
- Graphiques 3D et animations
- Dashboards professionnels
- Export haute qualité (PNG, SVG, PDF)
- Compatible JavaScript/web

**💡 Utilisations privilégiées :**
- Dashboards interactifs
- Présentations clients
- Exploration de données complexes
- Applications web professionnelles
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# 1. SCATTER PLOT 3D avec animation
# Données de ventes par région et trimestre
np.random.seed(42)
data_3d = pd.DataFrame({
    'trimestre': np.repeat(['Q1', 'Q2', 'Q3', 'Q4'], 20),
    'region': np.tile(['Nord', 'Sud', 'Est', 'Ouest', 'Centre'], 16),
    'ventes': np.random.uniform(50, 200, 80),
    'profit': np.random.uniform(10, 50, 80),
    'clients': np.random.randint(10, 100, 80)
})

fig_3d = px.scatter_3d(data_3d, 
                      x='ventes', y='profit', z='clients',
                      color='region', 
                      animation_frame='trimestre',
                      size='ventes',
                      hover_data=['region', 'trimestre'],
                      title='📊 Analyse 3D: Ventes-Profit-Clients par Région')

fig_3d.update_layout(height=500)
st.plotly_chart(fig_3d, use_container_width=True)

# 2. SUNBURST - Hiérarchie des données
# Structure hiérarchique des ventes
hierarchy_data = pd.DataFrame({
    'categories': ['Tech', 'Tech', 'Tech', 'Vêtements', 'Vêtements', 'Maison', 'Maison'],
    'sous_categories': ['Laptops', 'Phones', 'Tablets', 'Hommes', 'Femmes', 'Cuisine', 'Salon'],
    'valeurs': [450, 380, 220, 300, 420, 180, 150],
    'parents': ['', '', '', '', '', '', '']
})

fig_sunburst = px.sunburst(hierarchy_data,
                          path=['categories', 'sous_categories'],
                          values='valeurs',
                          title='🌟 Répartition des Ventes par Catégorie')
st.plotly_chart(fig_sunburst, use_container_width=True)

# 3. GRAPHIQUE FINANCIER (Candlestick)
# Simulation données boursières
dates_finance = pd.date_range('2024-01-01', periods=30, freq='D')
np.random.seed(123)
prices = pd.DataFrame({
    'date': dates_finance,
    'open': 100 + np.random.randn(30).cumsum(),
    'close': 100 + np.random.randn(30).cumsum(),
    'high': 100 + np.random.randn(30).cumsum() + 2,
    'low': 100 + np.random.randn(30).cumsum() - 2,
    'volume': np.random.randint(1000, 5000, 30)
})

fig_finance = go.Figure(data=go.Candlestick(
    x=prices['date'],
    open=prices['open'],
    high=prices['high'],
    low=prices['low'],
    close=prices['close'],
    name='Prix Action'
))

fig_finance.update_layout(
    title='📈 Graphique Financier (Candlestick)',
    yaxis_title='Prix (€)',
    xaxis_title='Date'
)
st.plotly_chart(fig_finance, use_container_width=True)
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    try:
        # 1. Scatter 3D
        np.random.seed(42)
        data_3d = pd.DataFrame({
            'trimestre': np.repeat(['Q1', 'Q2', 'Q3', 'Q4'], 20),
            'region': np.tile(['Nord', 'Sud', 'Est', 'Ouest', 'Centre'], 16),
            'ventes': np.random.uniform(50, 200, 80),
            'profit': np.random.uniform(10, 50, 80),
            'clients': np.random.randint(10, 100, 80)
        })
        
        fig_3d = px.scatter_3d(data_3d, 
                              x='ventes', y='profit', z='clients',
                              color='region', 
                              animation_frame='trimestre',
                              size='ventes',
                              hover_data=['region', 'trimestre'],
                              title='📊 Analyse 3D: Ventes-Profit-Clients')
        
        fig_3d.update_layout(height=400)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # 2. Sunburst
        hierarchy_data = pd.DataFrame({
            'categories': ['Tech', 'Tech', 'Tech', 'Vêtements', 'Vêtements', 'Maison', 'Maison'],
            'sous_categories': ['Laptops', 'Phones', 'Tablets', 'Hommes', 'Femmes', 'Cuisine', 'Salon'],
            'valeurs': [450, 380, 220, 300, 420, 180, 150],
        })
        
        fig_sunburst = px.sunburst(hierarchy_data,
                                  path=['categories', 'sous_categories'],
                                  values='valeurs',
                                  title='🌟 Répartition des Ventes')
        st.plotly_chart(fig_sunburst, use_container_width=True)
        
        # 3. Candlestick
        dates_finance = pd.date_range('2024-01-01', periods=30, freq='D')
        np.random.seed(123)
        base_price = 100
        prices_list = [base_price]
        
        for i in range(29):
            change = np.random.randn() * 2
            prices_list.append(prices_list[-1] + change)
        
        prices = pd.DataFrame({
            'date': dates_finance,
            'open': prices_list,
            'close': [p + np.random.randn() for p in prices_list],
            'high': [p + abs(np.random.randn()) + 1 for p in prices_list],
            'low': [p - abs(np.random.randn()) - 1 for p in prices_list],
        })
        
        fig_finance = go.Figure(data=go.Candlestick(
            x=prices['date'],
            open=prices['open'],
            high=prices['high'],
            low=prices['low'],
            close=prices['close'],
            name='Prix Action'
        ))
        
        fig_finance.update_layout(
            title='📈 Graphique Financier (Candlestick)',
            yaxis_title='Prix (€)',
            xaxis_title='Date',
            height=400
        )
        st.plotly_chart(fig_finance, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur Plotly: {e}")
        st.info("📦 Installez Plotly: `pip install plotly`")

st.markdown("""
**🔍 Types de graphiques Plotly avancés :**
- **Scatter 3D** : Exploration de données multidimensionnelles avec animation
- **Sunburst** : Hiérarchies et proportions (alternative aux camemberts)
- **Candlestick** : Données financières OHLC (Open-High-Low-Close)
- **Heatmaps** : Matrices de corrélation, cartes de chaleur
- **Subplots** : Combinaisons de graphiques différents
- **Animations** : Évolution temporelle avec contrôles de lecture
""")

st.divider()

# ================================
# 4. MÉTRIQUES ET INDICATEURS KPI
# ================================
st.subheader("4️⃣ Métriques et indicateurs KPI")

st.markdown("""
**📖 Description :**
Les métriques et indicateurs visuels permettent d'afficher instantanément les KPI clés.
Streamlit propose des composants spécialisés pour créer des tableaux de bord efficaces.

**🎯 Composants disponibles :**
- `st.metric()` → KPI avec variation et couleurs
- `st.progress()` → Barres de progression
- Combinaisons pour tableaux de bord

**💡 Bonnes pratiques :**
- Utilisez des couleurs significatives (vert=positif, rouge=négatif)
- Ajoutez des deltas pour montrer l'évolution
- Organisez en colonnes pour la lisibilité
- Accompagnez de graphiques de contexte
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# 1. MÉTRIQUES BUSINESS avec variations
st.markdown("### 📊 Tableau de Bord Business")

# Données KPI simulées
kpi_data = {
    'ca_actuel': 2_500_000,
    'ca_precedent': 2_200_000,
    'utilisateurs_actifs': 15_847,
    'utilisateurs_precedent': 16_203,
    'taux_conversion': 3.2,
    'taux_precedent': 2.8,
    'satisfaction': 8.7,
    'satisfaction_precedente': 8.5
}

# Organisation en colonnes
col1, col2, col3, col4 = st.columns(4)

with col1:
    delta_ca = kpi_data['ca_actuel'] - kpi_data['ca_precedent']
    st.metric(
        label="💰 Chiffre d'Affaires",
        value=f"{kpi_data['ca_actuel']:,.0f} €",
        delta=f"{delta_ca:,.0f} €",
        delta_color="normal"
    )

with col2:
    delta_users = kpi_data['utilisateurs_actifs'] - kpi_data['utilisateurs_precedent']
    st.metric(
        label="👥 Utilisateurs Actifs",
        value=f"{kpi_data['utilisateurs_actifs']:,}",
        delta=f"{delta_users:,}",
        delta_color="inverse"  # Rouge si négatif
    )

with col3:
    delta_conversion = kpi_data['taux_conversion'] - kpi_data['taux_precedent']
    st.metric(
        label="🎯 Taux Conversion",
        value=f"{kpi_data['taux_conversion']:.1f}%",
        delta=f"{delta_conversion:.1f}%"
    )

with col4:
    delta_satisfaction = kpi_data['satisfaction'] - kpi_data['satisfaction_precedente']
    st.metric(
        label="😊 Satisfaction",
        value=f"{kpi_data['satisfaction']:.1f}/10",
        delta=f"+{delta_satisfaction:.1f}"
    )

# 2. BARRES DE PROGRESSION avec contexte
st.markdown("### 📈 Objectifs et Progression")

objectifs = {
    'Ventes Annuelles': {'actuel': 2_500_000, 'objectif': 3_000_000},
    'Nouveaux Clients': {'actuel': 1_847, 'objectif': 2_000},
    'Satisfaction Client': {'actuel': 8.7, 'objectif': 9.0},
    'Retention Rate': {'actuel': 0.78, 'objectif': 0.80}
}

for nom, data in objectifs.items():
    progression = data['actuel'] / data['objectif']
    col_metric, col_progress = st.columns([1, 2])
    
    with col_metric:
        st.markdown(f"**{nom}**")
        st.write(f"{data['actuel']:,.0f} / {data['objectif']:,.0f}")
    
    with col_progress:
        st.progress(min(progression, 1.0))
        if progression >= 1.0:
            st.success(f"🎉 Objectif atteint ! ({progression:.1%})")
        else:
            restant = (1 - progression) * 100
            st.info(f"📊 {progression:.1%} - reste {restant:.0f}%")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Métriques business
    st.markdown("### 📊 Tableau de Bord Business")
    
    kpi_data = {
        'ca_actuel': 2_500_000,
        'ca_precedent': 2_200_000,
        'utilisateurs_actifs': 15_847,
        'utilisateurs_precedent': 16_203,
        'taux_conversion': 3.2,
        'taux_precedent': 2.8,
        'satisfaction': 8.7,
        'satisfaction_precedente': 8.5
    }
    
    # KPI en colonnes
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        delta_ca = kpi_data['ca_actuel'] - kpi_data['ca_precedent']
        st.metric(
            label="💰 CA",
            value=f"{kpi_data['ca_actuel']/1000:.0f}k €",
            delta=f"{delta_ca/1000:.0f}k €"
        )
    
    with kpi_col2:
        delta_users = kpi_data['utilisateurs_actifs'] - kpi_data['utilisateurs_precedent']
        st.metric(
            label="👥 Users",
            value=f"{kpi_data['utilisateurs_actifs']:,}",
            delta=f"{delta_users:,}",
            delta_color="inverse"
        )
    
    with kpi_col3:
        delta_conversion = kpi_data['taux_conversion'] - kpi_data['taux_precedent']
        st.metric(
            label="🎯 Conv.",
            value=f"{kpi_data['taux_conversion']:.1f}%",
            delta=f"{delta_conversion:.1f}%"
        )
    
    with kpi_col4:
        delta_satisfaction = kpi_data['satisfaction'] - kpi_data['satisfaction_precedente']
        st.metric(
            label="😊 Satisf.",
            value=f"{kpi_data['satisfaction']:.1f}/10",
            delta=f"+{delta_satisfaction:.1f}"
        )
    
    # Barres de progression
    st.markdown("### 📈 Objectifs et Progression")
    
    objectifs = {
        'Ventes Annuelles': {'actuel': 2_500_000, 'objectif': 3_000_000},
        'Nouveaux Clients': {'actuel': 1_847, 'objectif': 2_000},
        'Satisfaction': {'actuel': 8.7, 'objectif': 9.0},
        'Rétention': {'actuel': 0.78, 'objectif': 0.80}
    }
    
    for nom, data in objectifs.items():
        progression = data['actuel'] / data['objectif']
        prog_col_metric, prog_col_progress = st.columns([1, 2])
        
        with prog_col_metric:
            st.markdown(f"**{nom}**")
            if nom in ['Satisfaction', 'Rétention']:
                st.write(f"{data['actuel']:.1f} / {data['objectif']:.1f}")
            else:
                st.write(f"{data['actuel']:,.0f} / {data['objectif']:,.0f}")
        
        with prog_col_progress:
            st.progress(min(progression, 1.0))
            if progression >= 1.0:
                st.success(f"🎉 Atteint ! ({progression:.1%})")
            else:
                restant = (1 - progression) * 100
                st.info(f"{progression:.1%} - reste {restant:.0f}%")

st.markdown("""
**🔍 Techniques de design pour KPI :**
- **Delta color** : `normal` (vert=positif), `inverse` (rouge=positif pour des métriques comme les coûts)
- **Formatage des nombres** : Utilisez `f"{value:,.0f}"` pour les milliers, `f"{value:.1%}"` pour les pourcentages
- **Organisation spatiale** : Colonnes pour grouper les métriques liées
- **Codes couleur** : Vert/rouge universels, bleu pour l'information neutre
- **Progression contextuelle** : Barres accompagnées de métriques pour la compréhension
""")

st.divider()

# ================================
# 5. CARTES ET GÉOVISUALISATION
# ================================
st.subheader("5️⃣ Cartes et géovisualisation")

st.markdown("""
**📖 Description :**
La visualisation géographique permet d'analyser des données dans leur contexte spatial.
Streamlit offre des cartes intégrées simples et une intégration Plotly pour des cartes avancées.

**🎯 Types de cartes disponibles :**
- `st.map()` → Points simples sur carte OpenStreetMap
- Plotly Mapbox → Cartes interactives avancées
- Cartes choroplèthes pour les régions
- Cartes de densité (heatmaps géographiques)

**💡 Applications pratiques :**
- Localisation de magasins, clients, événements
- Analyse de densité de population
- Visualisation de données météorologiques
- Tracking de livraisons et logistique
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# 1. CARTE SIMPLE avec points d'intérêt
# Données géographiques - Grandes villes françaises
villes_france = pd.DataFrame({
    'ville': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice', 'Nantes', 'Montpellier', 'Strasbourg'],
    'lat': [48.8566, 45.7640, 43.2965, 43.6047, 43.7102, 47.2184, 43.6110, 48.5734],
    'lon': [2.3522, 4.8357, 5.3698, 1.4442, 7.2620, -1.5536, 3.8767, 7.7521],
    'population': [2_161_000, 516_000, 861_000, 479_000, 342_000, 309_000, 285_000, 280_000],
    'ventes_2024': [15_200, 4_800, 6_200, 3_900, 2_800, 2_400, 2_100, 2_300]
})

st.markdown("**🗺️ Carte Simple - Localisation des Villes**")
st.map(villes_france[['lat', 'lon']], size='population', color='#FF6B6B')

# 2. CARTE PLOTLY INTERACTIVE avec informations détaillées
try:
    fig_map = px.scatter_mapbox(
        villes_france,
        lat='lat', lon='lon',
        size='population',
        color='ventes_2024',
        hover_name='ville',
        hover_data={
            'population': ':,',
            'ventes_2024': ':,',
            'lat': False,
            'lon': False
        },
        color_continuous_scale='Viridis',
        size_max=25,
        zoom=5,
        center={'lat': 46.2276, 'lon': 2.2137},  # Centre de la France
        title='💰 Ventes par Ville (Carte Interactive)'
    )
    
    fig_map.update_layout(
        mapbox_style='open-street-map',
        height=500,
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
except ImportError:
    st.warning("📦 Plotly requis pour les cartes avancées")

# 3. SIMULATION DONNÉES DE DENSITÉ (Heatmap géographique)
# Génération de points autour de Paris
np.random.seed(42)
n_points = 200
paris_lat, paris_lon = 48.8566, 2.3522

# Points distribués autour de Paris
density_data = pd.DataFrame({
    'lat': np.random.normal(paris_lat, 0.05, n_points),
    'lon': np.random.normal(paris_lon, 0.07, n_points),
    'intensity': np.random.exponential(2, n_points)
})

st.markdown("**🔥 Simulation - Densité d'Activité autour de Paris**")
try:
    fig_density = px.density_mapbox(
        density_data,
        lat='lat', lon='lon',
        z='intensity',
        radius=15,
        center={'lat': paris_lat, 'lon': paris_lon},
        zoom=10,
        mapbox_style='open-street-map',
        title='Heatmap - Densité d\'Activité'
    )
    
    fig_density.update_layout(height=400)
    st.plotly_chart(fig_density, use_container_width=True)
    
except:
    st.info("Heatmap nécessite Plotly avec token Mapbox (optionnel)")
    # Fallback: carte simple
    st.map(density_data[['lat', 'lon']], size='intensity')
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Données géographiques
    villes_france = pd.DataFrame({
        'ville': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice', 'Nantes', 'Montpellier', 'Strasbourg'],
        'lat': [48.8566, 45.7640, 43.2965, 43.6047, 43.7102, 47.2184, 43.6110, 48.5734],
        'lon': [2.3522, 4.8357, 5.3698, 1.4442, 7.2620, -1.5536, 3.8767, 7.7521],
        'population': [2_161_000, 516_000, 861_000, 479_000, 342_000, 309_000, 285_000, 280_000],
        'ventes_2024': [15_200, 4_800, 6_200, 3_900, 2_800, 2_400, 2_100, 2_300]
    })
    
    # Carte simple
    st.markdown("**🗺️ Localisation des Villes**")
    st.map(villes_france[['lat', 'lon']])
    
    # Carte Plotly interactive
    try:
        fig_map = px.scatter_mapbox(
            villes_france,
            lat='lat', lon='lon',
            size='population',
            color='ventes_2024',
            hover_name='ville',
            hover_data={
                'population': ':,',
                'ventes_2024': ':,',
                'lat': False,
                'lon': False
            },
            color_continuous_scale='Viridis',
            size_max=25,
            zoom=5,
            center={'lat': 46.2276, 'lon': 2.2137},
            title='💰 Ventes par Ville'
        )
        
        fig_map.update_layout(
            mapbox_style='open-street-map',
            height=400,
            margin={"r":0,"t":50,"l":0,"b":0}
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
    except Exception as e:
        st.info("Carte Plotly non disponible")
        st.write(f"Détail: {e}")
    
    # Données de densité
    np.random.seed(42)
    n_points = 100
    paris_lat, paris_lon = 48.8566, 2.3522
    
    density_data = pd.DataFrame({
        'lat': np.random.normal(paris_lat, 0.02, n_points),
        'lon': np.random.normal(paris_lon, 0.03, n_points),
        'intensity': np.random.exponential(2, n_points)
    })
    
    st.markdown("**🔥 Densité d'Activité (Paris)**")
    st.map(density_data[['lat', 'lon']])

st.markdown("""
**🔍 Types de visualisations géographiques :**
- **Points simples** : Localisation d'événements, magasins, clients
- **Taille variable** : Représenter des quantités (population, ventes)
- **Couleurs** : Catégories ou valeurs continues
- **Heatmaps** : Densité et concentration spatiale
- **Choroplèthes** : Données par régions administratives
- **Trajectoires** : Mouvements, routes, migrations
""")

st.divider()

# ================================
# 6. IMAGES ET MÉDIAS AVANCÉS
# ================================
st.subheader("6️⃣ Images et médias avancés")

st.markdown("""
**📖 Description :**
Streamlit permet d'intégrer facilement des médias riches dans vos applications.
Au-delà de l'affichage simple, vous pouvez traiter, analyser et manipuler les images.

**🎯 Capacités multimédia :**
- Images (URL, fichiers locaux, arrays NumPy)
- Vidéos et audio (streaming, fichiers locaux)
- Génération d'images programmatiques
- Traitement d'images avec OpenCV/PIL

**💡 Applications créatives :**
- Galeries d'images dynamiques
- Visualisation de données scientifiques
- Prototypes d'applications visuelles
- Rapports avec graphiques personnalisés
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# 1. GALERIE D'IMAGES avec contrôles
images_demo = {
    'Streamlit Logo': 'https://docs.streamlit.io/logo.svg',
    'Python Logo': 'https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg',
    'Plotly Logo': 'https://upload.wikimedia.org/wikipedia/commons/8/8a/Plotly-logo.png'
}

# Sélecteur d'image
selected_image = st.selectbox("🖼️ Choisir une image:", list(images_demo.keys()))

# Contrôles d'affichage
col_width, col_caption = st.columns(2)
with col_width:
    width = st.slider("Largeur", 100, 500, 300)
with col_caption:
    show_caption = st.checkbox("Afficher légende", True)

# Affichage conditionnel
caption_text = f"Image: {selected_image}" if show_caption else None
st.image(images_demo[selected_image], 
         caption=caption_text,
         width=width)

# 2. GÉNÉRATION D'IMAGES PROGRAMMATIQUES
st.markdown("**🎨 Génération d'Images Dynamiques**")

# Contrôles pour la génération
fig_params = st.columns(3)
with fig_params[0]:
    pattern_type = st.radio("Type:", ["Sinus", "Cercles", "Damier"])
with fig_params[1]:
    resolution = st.slider("Résolution", 50, 200, 100)
with fig_params[2]:
    color_mode = st.selectbox("Couleurs:", ["Viridis", "Plasma", "Grayscale"])

# Génération de l'image selon les paramètres
x = np.linspace(0, 4*np.pi, resolution)
y = np.linspace(0, 4*np.pi, resolution)
X, Y = np.meshgrid(x, y)

if pattern_type == "Sinus":
    Z = np.sin(X) * np.cos(Y)
elif pattern_type == "Cercles":
    Z = np.sin(np.sqrt(X**2 + Y**2))
else:  # Damier
    Z = np.sin(X) * np.sin(Y)

# Conversion en image colorée
if color_mode == "Grayscale":
    # Image en niveaux de gris
    image_array = ((Z + 1) * 127.5).astype(np.uint8)
    image_array = np.stack([image_array]*3, axis=-1)  # RGB
else:
    # Image colorée avec matplotlib colormap
    import matplotlib.cm as cm
    cmap = cm.viridis if color_mode == "Viridis" else cm.plasma
    image_array = (cmap((Z + 1) / 2) * 255).astype(np.uint8)[:, :, :3]

st.image(image_array, 
         caption=f"Image générée: {pattern_type} ({resolution}x{resolution})",
         width=400)

# 3. ANALYSE D'IMAGE SIMPLE
st.markdown("**📊 Analyse d'Image**")
if st.button("🔍 Analyser l'image générée"):
    # Statistiques de l'image
    mean_color = np.mean(image_array, axis=(0, 1))
    std_color = np.std(image_array, axis=(0, 1))
    
    analysis_cols = st.columns(3)
    with analysis_cols[0]:
        st.metric("Rouge moyen", f"{mean_color[0]:.0f}")
    with analysis_cols[1]:
        st.metric("Vert moyen", f"{mean_color[1]:.0f}")
    with analysis_cols[2]:
        st.metric("Bleu moyen", f"{mean_color[2]:.0f}")
    
    # Histogramme des couleurs
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        ax_hist.hist(image_array[:, :, i].flatten(), 
                    bins=50, alpha=0.7, color=color, label=f'Canal {color}')
    ax_hist.set_title('Distribution des Couleurs')
    ax_hist.set_xlabel('Intensité (0-255)')
    ax_hist.set_ylabel('Fréquence')
    ax_hist.legend()
    st.pyplot(fig_hist)
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Galerie d'images
    images_demo = {
        'Streamlit Logo': 'https://docs.streamlit.io/logo.svg',
        'Python Logo': 'https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg'
    }
    
    selected_image = st.selectbox("🖼️ Choisir:", list(images_demo.keys()), key="img_select")
    
    img_col_width, img_col_caption = st.columns(2)
    with img_col_width:
        width = st.slider("Largeur", 100, 400, 250, key="img_width")
    with img_col_caption:
        show_caption = st.checkbox("Légende", True, key="img_caption")
    
    caption_text = f"Image: {selected_image}" if show_caption else None
    try:
        st.image(images_demo[selected_image], 
                 caption=caption_text,
                 width=width)
    except:
        st.error("Erreur de chargement de l'image")
    
    # Génération d'images
    st.markdown("**🎨 Génération Dynamique**")
    
    gen_params = st.columns(3)
    with gen_params[0]:
        pattern_type = st.radio("Type:", ["Sinus", "Cercles"], key="pattern")
    with gen_params[1]:
        resolution = st.slider("Résolution", 50, 150, 100, key="resolution")
    with gen_params[2]:
        color_mode = st.selectbox("Couleurs:", ["Viridis", "Plasma"], key="color")
    
    # Génération
    x = np.linspace(0, 4*np.pi, resolution)
    y = np.linspace(0, 4*np.pi, resolution)
    X, Y = np.meshgrid(x, y)
    
    if pattern_type == "Sinus":
        Z = np.sin(X) * np.cos(Y)
    else:  # Cercles
        Z = np.sin(np.sqrt(X**2 + Y**2))
    
    # Conversion en couleurs
    if color_mode == "Viridis":
        import matplotlib.cm as cm
        image_array = (cm.viridis((Z + 1) / 2) * 255).astype(np.uint8)[:, :, :3]
    else:
        image_array = (cm.plasma((Z + 1) / 2) * 255).astype(np.uint8)[:, :, :3]
    
    st.image(image_array, 
             caption=f"{pattern_type} ({resolution}x{resolution})",
             width=300)
    
    # Analyse simple
    if st.button("🔍 Analyser", key="analyze_img"):
        mean_color = np.mean(image_array, axis=(0, 1))
        
        analysis_cols = st.columns(3)
        with analysis_cols[0]:
            st.metric("R", f"{mean_color[0]:.0f}")
        with analysis_cols[1]:
            st.metric("G", f"{mean_color[1]:.0f}")
        with analysis_cols[2]:
            st.metric("B", f"{mean_color[2]:.0f}")

st.markdown("""
**🔍 Techniques avancées pour les médias :**
- **Images dynamiques** : Génération avec NumPy, modification en temps réel
- **Contrôles interactifs** : Sliders et boutons pour manipuler les paramètres
- **Analyse visuelle** : Statistiques, histogrammes, propriétés d'images
- **Galeries** : Collections organisées avec navigation
- **Formats supportés** : PNG, JPG, SVG, GIF pour images ; MP4, AVI pour vidéos
- **Performance** : Optimisez la taille des images pour des applications rapides
""")

st.markdown("---")

st.success("🎉 **Félicitations !** Vous maîtrisez maintenant toutes les techniques de visualisation avancées avec Streamlit !")

st.markdown("""
**📚 Récapitulatif des techniques de visualisation :**

📊 **Graphiques Natifs** → Simplicité et efficacité pour prototypes rapides  
🎨 **Matplotlib** → Contrôle total et qualité publication scientifique  
⚡ **Plotly** → Interactivité riche et dashboards professionnels  
📈 **Métriques KPI** → Tableaux de bord business avec indicateurs visuels  
🗺️ **Cartes** → Géovisualisation et analyse spatiale  
🖼️ **Médias** → Images, vidéos, génération dynamique et analyse  

**🚀 Prochaine étape :** Explorez le module T_04 sur la gestion des données et DataFrames !
""")

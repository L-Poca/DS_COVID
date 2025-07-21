import streamlit as st
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta

st.header("⚡ T_07 - Fonctionnalités Avancées de Streamlit")

st.markdown("**📋 Objectif :** Maîtriser les fonctionnalités avancées pour créer des applications Streamlit professionnelles et performantes.")

st.markdown("---")

# ================================
# 1. SESSION STATE - GESTION D'ÉTAT
# ================================
st.subheader("1️⃣ Session State - La mémoire de votre application")

st.markdown("""
**📖 Description :**
Le Session State permet de conserver des données entre les interactions utilisateur. 
Par défaut, Streamlit re-exécute tout le script à chaque action, perdant toutes les variables.
Le Session State résout ce problème en agissant comme la "mémoire" de votre application.

**🎯 Utilisations principales :**
- Conserver des compteurs ou scores
- Sauvegarder les préférences utilisateur
- Maintenir l'historique des actions
- Créer des formulaires multi-étapes
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Initialisation (obligatoire avant utilisation)
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# Afficher la valeur
st.write(f"Compteur: {st.session_state.counter}")

# Bouton pour incrémenter
if st.button("➕ Ajouter 1"):
    st.session_state.counter += 1
    st.rerun()  # Rafraîchir l'interface

# Bouton pour réinitialiser
if st.button("🔄 Reset"):
    st.session_state.counter = 0
    st.rerun()
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Initialisation
    if 'demo_counter' not in st.session_state:
        st.session_state.demo_counter = 0
    
    st.write(f"**Compteur:** {st.session_state.demo_counter}")
    
    demo_col1, demo_col2 = st.columns(2)
    with demo_col1:
        if st.button("➕ Ajouter 1", key="add_demo"):
            st.session_state.demo_counter += 1
            st.rerun()
    
    with demo_col2:
        if st.button("🔄 Reset", key="reset_demo"):
            st.session_state.demo_counter = 0
            st.rerun()

st.divider()

# ================================
# 2. CALLBACKS AVEC SESSION STATE
# ================================
st.subheader("2️⃣ Callbacks et fonctions de rappel")

st.markdown("""
**📖 Description :**
Les callbacks permettent d'exécuter du code automatiquement lors d'un événement 
(clic de bouton, changement de slider, etc.) sans avoir besoin de conditions if.
Plus élégant et plus performant !

**🎯 Avantages des callbacks :**
- Code plus propre et organisé
- Pas besoin de `st.rerun()` manuel
- Exécution immédiate lors de l'événement
- Meilleure séparation des responsabilités
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Définir les fonctions callbacks
def increment_counter():
    st.session_state.counter_cb += 1

def decrement_counter():
    st.session_state.counter_cb -= 1

def reset_counter():
    st.session_state.counter_cb = 0

# Initialisation
if 'counter_cb' not in st.session_state:
    st.session_state.counter_cb = 0

# Interface avec callbacks
col1, col2, col3 = st.columns(3)
with col1:
    st.button("➕", on_click=increment_counter)
with col2:
    st.button("➖", on_click=decrement_counter)
with col3:
    st.button("🔄", on_click=reset_counter)

st.metric("Compteur", st.session_state.counter_cb)
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Callbacks pour la démo
    def demo_increment():
        st.session_state.demo_counter_cb += 1
    
    def demo_decrement():
        st.session_state.demo_counter_cb -= 1
    
    def demo_reset():
        st.session_state.demo_counter_cb = 0
    
    # Initialisation
    if 'demo_counter_cb' not in st.session_state:
        st.session_state.demo_counter_cb = 0
    
    # Interface
    cb_col1, cb_col2, cb_col3 = st.columns(3)
    with cb_col1:
        st.button("➕", on_click=demo_increment, key="demo_inc")
    with cb_col2:
        st.button("➖", on_click=demo_decrement, key="demo_dec")
    with cb_col3:
        st.button("🔄", on_click=demo_reset, key="demo_reset")
    
    st.metric("**Compteur Callback**", st.session_state.demo_counter_cb)

st.divider()

# ================================
# 3. SESSION STATE AVEC STRUCTURES COMPLEXES
# ================================
st.subheader("3️⃣ Session State avec structures complexes")

st.markdown("""
**📖 Description :**
Le Session State peut stocker bien plus que des nombres simples ! 
Dictionnaires, listes, objets... tout y passe pour créer des applications riches et interactives.

**🎯 Applications pratiques :**
- Historique des actions utilisateur
- Profil utilisateur persistant
- Panier d'achat ou liste de favoris
- Configuration d'application
- Données temporaires d'analyse
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Initialiser une structure complexe
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'name': '',
        'preferences': [],
        'history': [],
        'score': 0
    }

# Interface de saisie
name = st.text_input("Votre nom:", 
    value=st.session_state.user_profile['name'])

# Mettre à jour si changement
if name != st.session_state.user_profile['name']:
    st.session_state.user_profile['name'] = name
    st.session_state.user_profile['history'].append(
        f"Nom modifié: {name}")

# Sélection de préférences
prefs = st.multiselect("Préférences:", 
    ["Sport", "Tech", "Art", "Cuisine"],
    default=st.session_state.user_profile['preferences'])

if prefs != st.session_state.user_profile['preferences']:
    st.session_state.user_profile['preferences'] = prefs
    st.session_state.user_profile['score'] += 10

# Afficher le profil
st.json(st.session_state.user_profile)
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Initialiser la structure démo
    if 'demo_user_profile' not in st.session_state:
        st.session_state.demo_user_profile = {
            'name': '',
            'preferences': [],
            'history': [],
            'score': 0
        }
    
    # Interface
    demo_name = st.text_input("Votre nom:", 
                             value=st.session_state.demo_user_profile['name'],
                             key="demo_user_name")
    
    if demo_name != st.session_state.demo_user_profile['name']:
        st.session_state.demo_user_profile['name'] = demo_name
        st.session_state.demo_user_profile['history'].append(
            f"Nom: {demo_name} ({time.strftime('%H:%M:%S')})"
        )
    
    demo_prefs = st.multiselect("Préférences:", 
                               ["Sport", "Tech", "Art", "Cuisine"],
                               default=st.session_state.demo_user_profile['preferences'],
                               key="demo_user_prefs")
    
    if demo_prefs != st.session_state.demo_user_profile['preferences']:
        st.session_state.demo_user_profile['preferences'] = demo_prefs
        st.session_state.demo_user_profile['score'] += 10
    
    # Affichage du profil
    st.markdown("**📋 Profil Utilisateur :**")
    st.json(st.session_state.demo_user_profile)

st.divider()

# ================================
# 4. CACHE POUR OPTIMISER LES PERFORMANCES
# ================================
st.subheader("4️⃣ Cache pour optimiser les performances")

st.markdown("""
**📖 Description :**
Le cache évite de recalculer des opérations coûteuses à chaque interaction.
Streamlit propose deux types de cache : `@st.cache_data` pour les données 
et `@st.cache_resource` pour les ressources (connexions DB, modèles ML).

**🎯 Avantages du cache :**
- Accélération drastique de l'application
- Économie de ressources serveur
- Meilleure expérience utilisateur
- Gestion automatique de la mémoire
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Cache pour données (TTL = Time To Live)
@st.cache_data(ttl=3600)  # Cache pendant 1 heure
def load_large_dataset():
    # Simulation chargement lourd
    time.sleep(2)
    return pd.DataFrame({
        'data': np.random.randn(10000),
        'category': np.random.choice(['A', 'B', 'C'], 10000)
    })

# Cache pour ressources (connexions, modèles)
@st.cache_resource
def init_model():
    # Simulation chargement modèle ML
    time.sleep(1)
    return {"model": "DemoModel", "version": "1.0"}

# Cache avec paramètres avancés
@st.cache_data(
    ttl=1800,                    # 30 minutes
    max_entries=100,             # Max 100 entrées
    show_spinner="Chargement..."  # Message
)
def expensive_computation(param):
    time.sleep(1)
    return param ** 2

# Utilisation
if st.button("Charger données"):
    data = load_large_dataset()
    st.write(f"Données: {len(data)} lignes")

# Nettoyer le cache
if st.button("🗑️ Vider cache"):
    st.cache_data.clear()
    st.success("Cache vidé!")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Fonctions avec cache pour la démo
    @st.cache_data(ttl=300)  # 5 minutes pour la démo
    def demo_heavy_computation(n):
        time.sleep(0.5)  # Simulation calcul lourd
        return sum(range(n))
    
    @st.cache_resource
    def demo_load_model():
        time.sleep(0.3)
        return {"model": "DemoModel", "version": "1.0"}
    
    st.write("**🚀 Test de Performance :**")
    
    number_input = st.number_input("Nombre pour calcul", 
                                  min_value=1, max_value=1000, 
                                  value=100, key="cache_number")
    
    if st.button("🔄 Calculer (avec cache)", key="calc_with_cache"):
        start_time = time.time()
        result = demo_heavy_computation(int(number_input))
        end_time = time.time()
        
        st.success(f"✅ Résultat: {result:,}")
        st.info(f"⏱️ Temps: {end_time - start_time:.3f}s")
        st.write("💡 Le même calcul sera instantané grâce au cache!")
    
    if st.button("🤖 Charger modèle", key="load_model_demo"):
        model_info = demo_load_model()
        st.json(model_info)
    
    # Contrôles du cache
    cache_col1, cache_col2 = st.columns(2)
    with cache_col1:
        if st.button("🗑️ Vider cache données", key="clear_data_cache"):
            st.cache_data.clear()
            st.success("Cache données vidé!")
    
    with cache_col2:
        if st.button("🗑️ Vider cache ressources", key="clear_resource_cache"):
            st.cache_resource.clear()
            st.success("Cache ressources vidé!")

st.divider()

# ================================
# 5. CONTRÔLE DE FLUX AVANCÉ
# ================================
st.subheader("5️⃣ Contrôle de flux avancé")

st.markdown("""
**📖 Description :**
Le contrôle de flux permet de gérer l'exécution de votre application :
arrêter l'exécution conditionnellement, forcer des rafraîchissements,
créer des placeholders dynamiques, et afficher du code en temps réel.

**🎯 Outils de contrôle :**
- `st.stop()` → Arrêter l'exécution
- `st.rerun()` → Forcer un rafraîchissement
- `st.empty()` → Placeholder dynamique
- `st.echo()` → Afficher et exécuter du code
- `st.spinner()` → Indicateur de chargement
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Contrôle d'accès
user_connected = st.checkbox("Utilisateur connecté")
if not user_connected:
    st.warning("⚠️ Veuillez vous connecter")
    st.stop()  # Arrête l'exécution ici

st.success("✅ Vous êtes connecté!")

# Placeholder dynamique
placeholder = st.empty()
if st.button("🎲 Simuler processus"):
    for i in range(3):
        placeholder.info(f"⏳ Étape {i+1}/3")
        time.sleep(0.5)
    placeholder.success("✅ Terminé!")

# Echo - afficher le code ET l'exécuter
with st.echo():
    demo_value = 42
    st.write(f"Valeur: {demo_value}")

# Spinner
if st.button("⏳ Opération longue"):
    with st.spinner("Traitement..."):
        time.sleep(1)
    st.success("Terminé!")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Contrôle d'accès
    demo_auth = st.checkbox("Utilisateur connecté", key="demo_auth")
    
    if not demo_auth:
        st.warning("⚠️ Veuillez vous connecter pour accéder au contenu")
    else:
        st.success("✅ Accès autorisé!")
        
        # Placeholder dynamique
        demo_placeholder = st.empty()
        
        if st.button("🎲 Test Processus", key="demo_process"):
            for i in range(3):
                demo_placeholder.info(f"⏳ Étape {i+1}/3 en cours...")
                time.sleep(0.3)
            demo_placeholder.success("✅ Processus terminé!")
        
        # Echo démo
        st.write("**🔍 Code Echo :**")
        with st.echo():
            demo_calculation = 2 + 2
            st.write(f"Résultat: {demo_calculation}")
        
        # Spinner
        if st.button("⏳ Test Spinner", key="demo_spinner"):
            with st.spinner("Chargement..."):
                time.sleep(1)
            st.success("✅ Chargé!")

st.divider()

# ================================
# 6. CSS PERSONNALISÉ
# ================================
st.subheader("6️⃣ CSS personnalisé et styling")

st.markdown("""
**📖 Description :**
Streamlit permet d'injecter du CSS personnalisé pour créer des interfaces 
sur mesure avec `st.markdown()` et `unsafe_allow_html=True`.

**🎯 Possibilités de customisation :**
- Couleurs et gradients personnalisés
- Animations et transitions
- Layouts complexes avec Flexbox/Grid
- Styles responsive
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# CSS personnalisé
st.markdown('''
<style>
.custom-header {
    background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 1rem 0;
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    border-left: 4px solid #4ECDC4;
    margin: 1rem 0;
}
</style>
''', unsafe_allow_html=True)

# Utiliser les styles
st.markdown('''
<div class="custom-header">
    <h2>🎨 En-tête Stylé</h2>
    <p>Design moderne avec CSS</p>
</div>
''', unsafe_allow_html=True)

st.markdown('''
<div class="metric-card">
    <h3 style='color: #4ECDC4; margin-top: 0;'>
        📊 Métrique Custom
    </h3>
    <h2 style='color: #2c3e50; margin: 0;'>1,234</h2>
    <p style='color: #7f8c8d; margin-bottom: 0;'>
        ↗️ +12% ce mois
    </p>
</div>
''', unsafe_allow_html=True)
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # CSS pour cette démo
    st.markdown("""
    <style>
    .demo-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .demo-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .demo-metric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # En-tête stylé
    st.markdown("""
    <div class="demo-header">
        <h3 style='margin: 0; color: white;'>🎨 Design Personnalisé</h3>
        <p style='margin: 5px 0 0 0; opacity: 0.9;'>Avec CSS custom</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cartes métriques
    metric_col1, metric_col2 = st.columns(2)
    
    with metric_col1:
        st.markdown("""
        <div class="demo-metric">
            <h4 style='color: #667eea; margin: 0;'>📈 Ventes</h4>
            <h2 style='color: #2c3e50; margin: 5px 0;'>1,234</h2>
            <p style='color: #27ae60; margin: 0; font-size: 0.9em;'>↗️ +15%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown("""
        <div class="demo-metric">
            <h4 style='color: #e74c3c; margin: 0;'>👥 Users</h4>
            <h2 style='color: #2c3e50; margin: 5px 0;'>567</h2>
            <p style='color: #f39c12; margin: 0; font-size: 0.9em;'>↗️ +8%</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.success("🎉 **Félicitations !** Vous maîtrisez maintenant les fonctionnalités avancées de Streamlit !")

st.markdown("""
**🚀 Pour aller plus loin :**
- Explorez les autres modules de ce tutoriel
- Créez votre première application complète
- Consultez la documentation officielle Streamlit
- Participez à la communauté Streamlit
""")

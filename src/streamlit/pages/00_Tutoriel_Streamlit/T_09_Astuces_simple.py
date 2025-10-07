import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

st.header("💡 T_09 - Astuces (Version Simple)")

st.markdown("**📋 Objectif :** Découvrir les trucs et astuces pour créer des applications Streamlit professionnelles et user-friendly.")

st.markdown("---")

# ================================
# 1. MISE EN PAGE PROFESSIONNELLE
# ================================
st.subheader("1️⃣ Mise en page professionnelle")

st.markdown("""
**📖 Principe :**
Une bonne présentation fait toute la différence !
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Header avec style")
    st.code("""
# Header avec émojis et style
st.markdown("# 🚀 Mon Application")
st.markdown("### 📊 Tableau de bord des ventes")

# Ligne de séparation
st.markdown("---")

# Informations contextuelles
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="📈 Ventes du jour",
        value="1,234 €",
        delta="123 €"
    )

with col2:
    st.metric(
        label="👥 Visiteurs",
        value="456",
        delta="-23"
    )

with col3:
    st.metric(
        label="💰 Revenus",
        value="2,345 €",
        delta="234 €"
    )

# Note d'information
st.info("💡 Données mises à jour en temps réel")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    with st.container():
        st.markdown("**🚀 Mon Application Demo**")
        st.markdown("*📊 Tableau de bord des ventes*")
        st.markdown("---")
        
        # Métriques
        demo_col1, demo_col2, demo_col3 = st.columns(3)
        
        with demo_col1:
            st.metric(
                label="📈 Ventes du jour",
                value="1,234 €",
                delta="123 €"
            )
        
        with demo_col2:
            st.metric(
                label="👥 Visiteurs",
                value="456",
                delta="-23"
            )
        
        with demo_col3:
            st.metric(
                label="💰 Revenus",
                value="2,345 €",
                delta="234 €"
            )
        
        st.info("💡 Données mises à jour en temps réel")

st.divider()

# ================================
# 2. SIDEBAR INTELLIGENTE
# ================================
st.subheader("2️⃣ Sidebar intelligente")

st.markdown("""
**📖 Astuce :**
Une sidebar bien organisée pour les contrôles principaux.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Sidebar organisée")
    st.code("""
# Sidebar avec sections
with st.sidebar:
    st.markdown("# ⚙️ Paramètres")
    
    # Section 1 : Données
    st.markdown("## 📊 Données")
    date_debut = st.date_input("Date de début")
    date_fin = st.date_input("Date de fin")
    
    # Section 2 : Filtres
    st.markdown("## 🔍 Filtres")
    categories = st.multiselect(
        "Catégories:",
        ["A", "B", "C"],
        default=["A", "B"]
    )
    
    # Section 3 : Affichage
    st.markdown("## 👁️ Affichage")
    show_details = st.checkbox("Afficher détails")
    chart_type = st.selectbox("Type graphique:", 
                             ["Barres", "Lignes", "Aires"])
    
    # Section 4 : Actions
    st.markdown("## 🎬 Actions")
    if st.button("🔄 Actualiser"):
        st.success("Données actualisées!")
    
    if st.button("📥 Exporter"):
        st.success("Export lancé!")
    
    # Informations
    st.markdown("---")
    st.markdown("### ℹ️ Aide")
    st.markdown("Application de démonstration")
""")

with col2:
    st.markdown("#### 🎯 Aperçu de la sidebar")
    st.markdown("""
    👈 **Regardez la sidebar à gauche !**
    
    Elle contient maintenant une version démo des éléments suivants :
    - ⚙️ Titre des paramètres
    - 📊 Section données (dates)
    - 🔍 Section filtres (catégories)
    - 👁️ Section affichage (options)
    - 🎬 Section actions (boutons)
    - ℹ️ Section aide
    
    **💡 Avantages :**
    - Interface claire et organisée
    - Contrôles facilement accessibles  
    - Plus d'espace pour le contenu principal
    - Navigation intuitive
    """)

# Implémentation dans la vraie sidebar
with st.sidebar:
    st.markdown("# ⚙️ Paramètres Demo")
    
    # Section 1 : Données
    st.markdown("## 📊 Données")
    date_debut_demo = st.date_input("Date de début", key="demo_date_debut")
    date_fin_demo = st.date_input("Date de fin", key="demo_date_fin")
    
    # Section 2 : Filtres
    st.markdown("## 🔍 Filtres")
    categories_demo = st.multiselect(
        "Catégories:",
        ["A", "B", "C"],
        default=["A", "B"],
        key="demo_categories"
    )
    
    # Section 3 : Affichage
    st.markdown("## 👁️ Affichage")
    show_details_demo = st.checkbox("Afficher détails", key="demo_show_details")
    chart_type_demo = st.selectbox("Type graphique:", 
                         ["Barres", "Lignes", "Aires"],
                         key="demo_chart_type")
    
    # Section 4 : Actions
    st.markdown("## 🎬 Actions")
    if st.button("🔄 Actualiser", key="demo_refresh"):
        st.success("Données actualisées!")
    
    if st.button("📥 Exporter", key="demo_export"):
        st.success("Export lancé!")
    
    # Informations
    st.markdown("---")
    st.markdown("### ℹ️ Aide")
    st.markdown("Application de démonstration")

st.divider()

# ================================
# 3. MESSAGES UTILISATEUR
# ================================
st.subheader("3️⃣ Messages utilisateur efficaces")

st.markdown("""
**📖 Principe :**
Bien communiquer avec l'utilisateur pour une expérience fluide.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Messages variés")
    st.code("""
# Messages de différents types
st.success("✅ Opération réussie !")
st.info("💡 Information utile")
st.warning("⚠️ Attention à ce point")
st.error("❌ Erreur détectée")

# Messages avec expanders
with st.expander("💡 Aide - Comment utiliser"):
    st.write("Instructions détaillées ici...")

# Toast (message temporaire)
if st.button("📨 Message toast"):
    st.toast("Message temporaire !", icon="🎉")

# Progress avec message
if st.button("⏳ Traitement long"):
    progress = st.progress(0)
    status = st.empty()
    
    for i in range(101):
        progress.progress(i)
        status.text(f"Étape {i}/100 en cours...")
        time.sleep(0.01)
    
    status.success("Traitement terminé !")

# Messages contextuels
def valider_donnees(data):
    if data is None:
        st.error("❌ Aucune donnée fournie")
        return False
    elif len(data) == 0:
        st.warning("⚠️ Données vides")
        return False
    else:
        st.success(f"✅ {len(data)} éléments validés")
        return True
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Messages de différents types
    st.success("✅ Opération réussie !")
    st.info("💡 Information utile")
    st.warning("⚠️ Attention à ce point")
    st.error("❌ Erreur détectée")
    
    # Messages avec expanders
    with st.expander("💡 Aide - Comment utiliser"):
        st.write("Instructions détaillées ici...")
        st.markdown("""
        1. Sélectionnez vos paramètres
        2. Cliquez sur le bouton d'action
        3. Consultez les résultats
        """)
    
    # Toast
    if st.button("📨 Message toast", key="demo_toast"):
        st.toast("Message temporaire !", icon="🎉")
    
    # Progress avec message
    if st.button("⏳ Traitement long", key="demo_progress_msg"):
        progress = st.progress(0)
        status = st.empty()
        
        for i in range(0, 101, 20):
            progress.progress(i)
            status.text(f"Étape {i}/100 en cours...")
            time.sleep(0.2)
        
        status.success("Traitement terminé !")

st.divider()

# ================================
# 4. GESTION D'ERREURS
# ================================
st.subheader("4️⃣ Gestion d'erreurs élégante")

st.markdown("""
**📖 Principe :**
Anticiper les problèmes et les gérer proprement.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Gestion d'erreurs")
    st.code("""
# Fonction avec gestion d'erreur
def diviser_nombres(a, b):
    try:
        if b == 0:
            st.error("❌ Division par zéro impossible")
            return None
        
        resultat = a / b
        st.success(f"✅ Résultat: {resultat:.2f}")
        return resultat
        
    except Exception as e:
        st.error(f"❌ Erreur inattendue: {e}")
        return None

# Interface avec validation
col1, col2 = st.columns(2)

with col1:
    nombre1 = st.number_input("Premier nombre:", value=10.0)

with col2:
    nombre2 = st.number_input("Deuxième nombre:", value=2.0)

if st.button("➗ Diviser"):
    if nombre1 is not None and nombre2 is not None:
        diviser_nombres(nombre1, nombre2)
    else:
        st.warning("⚠️ Veuillez entrer deux nombres")

# Validation de fichier
fichier = st.file_uploader("Choisir un fichier CSV")

if fichier:
    try:
        df = pd.read_csv(fichier)
        if len(df) == 0:
            st.warning("⚠️ Fichier vide")
        else:
            st.success(f"✅ {len(df)} lignes chargées")
            st.dataframe(df.head())
            
    except Exception as e:
        st.error(f"❌ Erreur lecture fichier: {e}")
        st.info("💡 Vérifiez que c'est bien un fichier CSV")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Fonction avec gestion d'erreur
    def diviser_nombres_demo(a, b):
        try:
            if b == 0:
                st.error("❌ Division par zéro impossible")
                return None
            
            resultat = a / b
            st.success(f"✅ Résultat: {resultat:.2f}")
            return resultat
            
        except Exception as e:
            st.error(f"❌ Erreur inattendue: {e}")
            return None
    
    # Interface
    demo_calc_col1, demo_calc_col2 = st.columns(2)
    
    with demo_calc_col1:
        nombre1_demo = st.number_input("Premier nombre:", value=10.0, key="demo_num1")
    
    with demo_calc_col2:
        nombre2_demo = st.number_input("Deuxième nombre:", value=2.0, key="demo_num2")
    
    if st.button("➗ Diviser", key="demo_divide"):
        if nombre1_demo is not None and nombre2_demo is not None:
            diviser_nombres_demo(nombre1_demo, nombre2_demo)
        else:
            st.warning("⚠️ Veuillez entrer deux nombres")
    
    # Test avec zéro
    st.markdown("**💡 Essayez avec 0 comme deuxième nombre !**")

st.divider()

# ================================
# 5. INTERFACE RESPONSIVE
# ================================
st.subheader("5️⃣ Interface adaptative")

st.markdown("""
**📖 Principe :**
Une interface qui s'adapte à la taille d'écran.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Layout adaptatif")
    st.code("""
# Colonnes adaptatives
def create_responsive_layout():
    # Sur mobile : une colonne
    # Sur desktop : plusieurs colonnes
    
    # Détection approximative (par la largeur)
    col_count = st.selectbox("Colonnes:", [1, 2, 3, 4])
    
    if col_count == 1:
        # Layout mobile
        st.metric("Métrique 1", "100")
        st.metric("Métrique 2", "200") 
        st.metric("Métrique 3", "300")
        
    elif col_count == 2:
        # Layout tablette
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Métrique 1", "100")
            st.metric("Métrique 3", "300")
        with col2:
            st.metric("Métrique 2", "200")
            
    else:
        # Layout desktop
        cols = st.columns(col_count)
        metrics = ["100", "200", "300", "400"]
        
        for i, col in enumerate(cols):
            if i < len(metrics):
                with col:
                    st.metric(f"Métrique {i+1}", metrics[i])

create_responsive_layout()

# Containers pour organiser
with st.container():
    st.markdown("### 📊 Section principale")
    # Contenu principal ici
    
with st.container():
    st.markdown("### 📈 Section secondaire")
    # Contenu secondaire ici
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    def create_responsive_layout_demo():
        col_count = st.selectbox("Nombre de colonnes:", [1, 2, 3, 4], key="demo_col_count")
        
        if col_count == 1:
            # Layout mobile
            st.metric("Métrique 1", "100", "↗️ +10")
            st.metric("Métrique 2", "200", "↘️ -5") 
            st.metric("Métrique 3", "300", "↗️ +15")
            
        elif col_count == 2:
            # Layout tablette
            resp_col1, resp_col2 = st.columns(2)
            with resp_col1:
                st.metric("Métrique 1", "100", "↗️ +10")
                st.metric("Métrique 3", "300", "↗️ +15")
            with resp_col2:
                st.metric("Métrique 2", "200", "↘️ -5")
                
        else:
            # Layout desktop
            cols = st.columns(col_count)
            metrics_data = [
                ("Métrique 1", "100", "↗️ +10"),
                ("Métrique 2", "200", "↘️ -5"),
                ("Métrique 3", "300", "↗️ +15"),
                ("Métrique 4", "400", "↗️ +20")
            ]
            
            for i, col in enumerate(cols):
                if i < len(metrics_data):
                    with col:
                        name, value, delta = metrics_data[i]
                        st.metric(name, value, delta)
    
    create_responsive_layout_demo()
    
    st.info("💡 Changez le nombre de colonnes pour voir l'adaptation !")

st.divider()

# ================================
# 6. PERSONNALISATION CSS
# ================================
st.subheader("6️⃣ Personnalisation avec CSS")

st.markdown("""
**📖 Principe :**
Ajouter du style personnalisé avec du CSS.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - CSS custom")
    st.code("""
# CSS personnalisé
st.markdown('''
<style>
.custom-metric {
    background: linear-gradient(90deg, #ff6b6b, #feca57);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin: 10px 0;
}

.highlight-box {
    background: #f8f9fa;
    border-left: 5px solid #007bff;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}

.success-badge {
    background: #28a745;
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    display: inline-block;
    margin: 5px;
}
</style>
''', unsafe_allow_html=True)

# Utiliser les styles
st.markdown('''
<div class="custom-metric">
    <h2>🚀 Ventes du jour</h2>
    <h1>1,234 €</h1>
</div>
''', unsafe_allow_html=True)

st.markdown('''
<div class="highlight-box">
    💡 <strong>Conseil :</strong> Cette information est importante !
</div>
''', unsafe_allow_html=True)

st.markdown('''
<span class="success-badge">✅ Validé</span>
<span class="success-badge">🔥 Tendance</span>
<span class="success-badge">⭐ Premium</span>
''', unsafe_allow_html=True)
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # CSS personnalisé
    st.markdown('''
    <style>
    .demo-custom-metric {
        background: linear-gradient(90deg, #ff6b6b, #feca57);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    .demo-highlight-box {
        background: #f8f9fa;
        border-left: 5px solid #007bff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    .demo-success-badge {
        background: #28a745;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
        margin: 5px;
    }
    </style>
    ''', unsafe_allow_html=True)
    
    # Utiliser les styles
    st.markdown('''
    <div class="demo-custom-metric">
        <h3>🚀 Ventes du jour</h3>
        <h2>1,234 €</h2>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="demo-highlight-box">
        💡 <strong>Conseil :</strong> Cette information est importante !
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
    <span class="demo-success-badge">✅ Validé</span>
    <span class="demo-success-badge">🔥 Tendance</span>
    <span class="demo-success-badge">⭐ Premium</span>
    ''', unsafe_allow_html=True)

st.divider()

# ================================
# 7. RACCOURCIS CLAVIER
# ================================
st.subheader("7️⃣ Raccourcis et interactions")

st.markdown("""
**📖 Principe :**
Améliorer l'expérience utilisateur avec des interactions avancées.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Interactions avancées")
    st.code("""
# Auto-focus sur un élément
st.text_input("Champ auto-focus:", key="focus_field")

# Validation avec Enter
with st.form("quick_form"):
    user_input = st.text_input("Tapez et appuyez sur Entrée:")
    submit = st.form_submit_button("Valider")
    
    if submit and user_input:
        st.success(f"Vous avez tapé: {user_input}")

# Boutons avec raccourcis visuels
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Actualiser (F5)"):
        st.info("Actualisation...")

with col2:
    if st.button("💾 Sauver (Ctrl+S)"):
        st.success("Sauvegardé!")

with col3:
    if st.button("❓ Aide (F1)"):
        st.info("Aide affichée")

# Double-clic simulation
if 'click_count' not in st.session_state:
    st.session_state.click_count = 0

if st.button("🖱️ Double-clic (test)"):
    st.session_state.click_count += 1
    
    if st.session_state.click_count >= 2:
        st.success("Double-clic détecté!")
        st.session_state.click_count = 0
    else:
        st.info("Premier clic... cliquez encore!")

# Confirmation d'action
if st.button("🗑️ Supprimer"):
    if st.button("⚠️ Confirmer suppression"):
        st.success("Élément supprimé")
    else:
        st.warning("Cliquez sur 'Confirmer' pour valider")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Auto-focus (simulation)
    st.text_input("Champ focus:", key="demo_focus_field", 
                 help="💡 Ce champ aurait le focus automatiquement")
    
    # Validation avec Enter
    with st.form("demo_quick_form"):
        user_input_demo = st.text_input("Tapez et appuyez sur Entrée:", key="demo_user_input")
        submit_demo = st.form_submit_button("Valider")
        
        if submit_demo and user_input_demo:
            st.success(f"Vous avez tapé: {user_input_demo}")
    
    # Boutons avec raccourcis
    shortcut_col1, shortcut_col2, shortcut_col3 = st.columns(3)
    
    with shortcut_col1:
        if st.button("🔄 Actualiser", key="demo_refresh_f5"):
            st.info("Actualisation...")
    
    with shortcut_col2:
        if st.button("💾 Sauver", key="demo_save_ctrl_s"):
            st.success("Sauvegardé!")
    
    with shortcut_col3:
        if st.button("❓ Aide", key="demo_help_f1"):
            st.info("Aide affichée")
    
    # Double-clic simulation
    if 'demo_click_count' not in st.session_state:
        st.session_state.demo_click_count = 0
    
    if st.button("🖱️ Double-clic (test)", key="demo_double_click"):
        st.session_state.demo_click_count += 1
        
        if st.session_state.demo_click_count >= 2:
            st.success("Double-clic détecté!")
            st.session_state.demo_click_count = 0
        else:
            st.info("Premier clic... cliquez encore!")

st.divider()

# ================================
# 8. EXERCICE FINAL
# ================================
st.subheader("8️⃣ Exercice final : Application complète")

st.markdown("""
**🎯 Mission :**
Créer une mini-application qui combine toutes les astuces apprises.
""")

with st.expander("💻 Template de l'exercice"):
    st.code("""
import streamlit as st
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Mon App Pro",
    page_icon="🚀",
    layout="wide"
)

# CSS personnalisé
st.markdown('''
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 30px;
}
</style>
''', unsafe_allow_html=True)

# Header principal
st.markdown('''
<div class="main-header">
    <h1>🚀 Mon Dashboard Professionnel</h1>
    <p>Application de démonstration avec toutes les astuces</p>
</div>
''', unsafe_allow_html=True)

# Sidebar organisée
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Paramètres utilisateur
    user_name = st.text_input("Nom utilisateur:", "Utilisateur")
    show_advanced = st.checkbox("Mode avancé")
    
    # Filtres
    st.markdown("## 🔍 Filtres")
    date_range = st.date_input("Période:", value=[])
    categories = st.multiselect("Catégories:", ["A", "B", "C"])
    
    # Actions
    st.markdown("## 🎬 Actions")
    if st.button("🔄 Actualiser"):
        st.success("Actualisé!")
    
    if st.button("📊 Générer rapport"):
        st.success("Rapport généré!")

# Contenu principal
if user_name:
    st.markdown(f"### 👋 Bonjour {user_name} !")

# Métriques principales
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📈 Ventes", "1,234 €", "+12%")
with col2:
    st.metric("👥 Clients", "567", "+8%")
with col3:
    st.metric("📦 Commandes", "89", "-2%")
with col4:
    st.metric("💰 Profit", "456 €", "+15%")

# Graphiques avec tabs
tab1, tab2, tab3 = st.tabs(["📊 Ventes", "📈 Tendances", "📋 Données"])

with tab1:
    # Générer des données de démonstration
    data = pd.DataFrame({
        'Mois': ['Jan', 'Fév', 'Mar', 'Avr', 'Mai'],
        'Ventes': [1000, 1200, 1100, 1400, 1300]
    })
    st.bar_chart(data.set_index('Mois'))

with tab2:
    # Graphique de tendance
    trend_data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=30),
        'Valeur': np.cumsum(np.random.randn(30))
    })
    st.line_chart(trend_data.set_index('Date'))

with tab3:
    # Tableau avec données
    if show_advanced:
        st.dataframe(trend_data, use_container_width=True)
    else:
        st.dataframe(trend_data.head(), use_container_width=True)

# Messages contextuels
if len(categories) == 0:
    st.warning("⚠️ Aucune catégorie sélectionnée")
elif len(categories) > 2:
    st.info("💡 Beaucoup de catégories sélectionnées")
else:
    st.success("✅ Configuration optimale")

# Footer
st.markdown("---")
st.markdown("### 📊 Statistiques de session")

session_col1, session_col2, session_col3 = st.columns(3)

with session_col1:
    st.metric("⏰ Temps session", "15 min")
with session_col2:
    st.metric("🔄 Actualisations", "3")
with session_col3:
    st.metric("📱 Type appareil", "Desktop")
""")

# Implémentation de l'exemple
st.markdown("**🎯 Exemple fonctionnel :**")

# Mini version de l'app complète
st.markdown('''
<style>
.demo-main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
}
</style>
''', unsafe_allow_html=True)

st.markdown('''
<div class="demo-main-header">
    <h3>🚀 Mon Dashboard Demo</h3>
    <p>Version simplifiée avec les astuces</p>
</div>
''', unsafe_allow_html=True)

# Interface simplifiée
demo_user = st.text_input("Nom utilisateur:", "Demo User", key="demo_final_user")

if demo_user:
    st.markdown(f"**👋 Bonjour {demo_user} !**")

# Métriques
final_col1, final_col2, final_col3 = st.columns(3)

with final_col1:
    st.metric("📈 Ventes", "1,234 €", "+12%")
with final_col2:
    st.metric("👥 Clients", "567", "+8%")
with final_col3:
    st.metric("💰 Profit", "456 €", "+15%")

# Actions rapides
action_col1, action_col2, action_col3 = st.columns(3)

with action_col1:
    if st.button("🔄 Actualiser", key="demo_final_refresh"):
        st.success("✅ Données actualisées")

with action_col2:
    if st.button("📊 Rapport", key="demo_final_report"):
        st.success("✅ Rapport généré")

with action_col3:
    if st.button("📥 Export", key="demo_final_export"):
        st.success("✅ Export lancé")

st.divider()

# ================================
# 9. RÉCAPITULATIF FINAL
# ================================
st.subheader("9️⃣ Récapitulatif des astuces")

st.markdown("""
**🎓 Toutes les astuces apprises :**

### 🎨 **Présentation**
✅ Headers avec émojis et style  
✅ Métriques avec deltas  
✅ CSS personnalisé  
✅ Layout adaptatif  

### 🗂️ **Organisation**
✅ Sidebar bien structurée  
✅ Sections avec containers  
✅ Tabs pour organiser le contenu  
✅ Expanders pour les détails  

### 💬 **Communication**
✅ Messages appropriés (success, error, warning, info)  
✅ Toasts pour les notifications  
✅ Progress bars avec messages  
✅ Gestion d'erreurs élégante  

### ⚡ **Interactions**
✅ Formulaires pour grouper  
✅ Boutons avec raccourcis visuels  
✅ Validations en temps réel  
✅ Confirmations d'actions  

**💡 Les 10 règles d'or :**

1. **Émojis** : Rendent l'interface vivante
2. **Messages clairs** : L'utilisateur doit comprendre
3. **Gestion d'erreurs** : Toujours prévoir les problèmes
4. **Organisation** : Sidebar pour les contrôles
5. **Responsive** : Adapter à toutes les tailles
6. **Performance** : Cache et optimisation
7. **Style** : CSS pour personnaliser
8. **Feedback** : Toujours confirmer les actions
9. **Aide** : Expanders et tooltips
10. **Tests** : Vérifier tous les cas d'usage

**🏆 Vous êtes maintenant prêt(e) à créer des applications Streamlit professionnelles !**
""")

# Navigation finale
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⬅️ Module précédent (T_08)", key="nav_prev_t9"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_08 - Performance")

with col2:
    if st.button("📚 Retour au sommaire", key="nav_home_t9"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_00 - Sommaire")

with col3:
    if st.button("🎉 Félicitations !", key="nav_finish"):
        st.balloons()
        st.success("🏆 Vous avez terminé tous les tutoriels Streamlit !")
        st.info("💡 Vous pouvez maintenant créer vos propres applications !")

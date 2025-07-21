import streamlit as st
import datetime

st.header("🧩 T_02 - Widgets Interactifs")

st.markdown("**📋 Objectif :** Maîtriser tous les widgets interactifs pour créer des interfaces utilisateur riches et capturer efficacement les actions et données utilisateur.")

st.markdown("---")

# ================================
# 1. BOUTONS ET ACTIONS
# ================================
st.subheader("1️⃣ Boutons et actions")

st.markdown("""
**📖 Description :**
Les boutons sont les éléments d'interface les plus fondamentaux pour déclencher des actions.
Ils retournent `True` au moment précis du clic, `False` le reste du temps.
C'est crucial de comprendre que l'état `True` ne dure qu'un instant !

**🎯 Cas d'usage principaux :**
- Déclencher des calculs coûteux
- Valider des formulaires
- Lancer des processus (téléchargement, sauvegarde)
- Réinitialiser des données
- Naviguer entre des étapes

**💡 Points clés à retenir :**
- Le bouton retourne `True` uniquement lors du clic
- Utilisez `key` pour différencier des boutons identiques
- `disabled=True` empêche toute interaction
- Idéal pour les actions ponctuelles, pas les états permanents
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Bouton simple - action ponctuelle
if st.button("🔄 Rafraîchir les données"):
    st.balloons()  # Animation de célébration
    st.write("✅ Données rafraîchies !")

# Bouton avec clé unique (évite les conflits)
if st.button("💾 Sauvegarder", key="save_btn"):
    st.success("📁 Fichier sauvegardé avec succès")

# Bouton désactivé (état non-interactif)
st.button("🚫 Action indisponible", disabled=True)

# Bouton avec emoji pour plus d'attrait visuel
if st.button("🎲 Lancer le dé"):
    import random
    result = random.randint(1, 6)
    st.write(f"🎯 Résultat : {result}")

# Utilisation avec session state pour conserver l'état
if 'counter' not in st.session_state:
    st.session_state.counter = 0

if st.button("➕ Incrémenter compteur"):
    st.session_state.counter += 1
    
st.write(f"🔢 Compteur: {st.session_state.counter}")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    if st.button("🔄 Rafraîchir les données"):
        st.balloons()
        st.write("✅ Données rafraîchies !")
    
    if st.button("💾 Sauvegarder", key="save_btn"):
        st.success("📁 Fichier sauvegardé avec succès")
    
    st.button("🚫 Action indisponible", disabled=True)
    
    if st.button("🎲 Lancer le dé"):
        import random
        result = random.randint(1, 6)
        st.write(f"🎯 Résultat : {result}")
    
    if 'counter' not in st.session_state:
        st.session_state.counter = 0
    
    if st.button("➕ Incrémenter compteur"):
        st.session_state.counter += 1
        
    st.write(f"🔢 Compteur: {st.session_state.counter}")

st.markdown("""
**🔍 Analyse technique :**
- `st.button()` retourne un booléen : `True` lors du clic, `False` sinon
- La valeur `True` n'existe que pendant l'exécution du script après le clic
- Utilisez `st.session_state` pour conserver des données entre les clics
- La propriété `key` permet d'avoir plusieurs boutons avec le même nom
- `disabled=True` rend le bouton visuellement et fonctionnellement inactif
""")

st.divider()

# ================================
# 2. CASES À COCHER ET TOGGLE
# ================================
st.subheader("2️⃣ Cases à cocher et toggle")

st.markdown("""
**📖 Description :**
Les cases à cocher et toggles capturent des états binaires (Vrai/Faux) de manière persistante.
Contrairement aux boutons, ces widgets conservent leur état jusqu'à ce que l'utilisateur les modifie.
Parfaits pour les préférences, options, et configurations.

**🎯 Différences importantes :**
- `st.checkbox()` → Style classique, case carrée ☑️
- `st.toggle()` → Style moderne, interrupteur 🎛️
- Valeur persistante (ne change que si l'utilisateur agit)
- Parfait pour activer/désactiver des fonctionnalités

**💡 Utilisations recommandées :**
- Préférences utilisateur (mode sombre, notifications)
- Filtres dans des tableaux de données
- Options de configuration (affichage, paramètres)
- Conditions d'activation de fonctionnalités
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Case à cocher classique
show_details = st.checkbox("📋 Afficher les détails")

if show_details:
    st.info("🔍 Mode détaillé activé")
    st.write("• Information 1")
    st.write("• Information 2") 
    st.write("• Information 3")

# Case cochée par défaut
debug_mode = st.checkbox("🐛 Mode debug", value=True)

# Toggle moderne (équivalent mais plus stylé)
dark_mode = st.toggle("🌙 Mode sombre")

# Utilisation pour filtrer des données
show_advanced = st.checkbox("⚙️ Options avancées")

if show_advanced:
    st.slider("Paramètre expert", 0, 100, 50)
    st.selectbox("Configuration", ["Mode A", "Mode B"])

# Combinaison de plusieurs options
st.markdown("**🎛️ Préférences d'affichage :**")
col_a, col_b, col_c = st.columns(3)
with col_a:
    show_graphs = st.checkbox("📊 Graphiques")
with col_b: 
    show_tables = st.checkbox("📋 Tableaux", value=True)
with col_c:
    show_maps = st.checkbox("🗺️ Cartes")

# Affichage conditionnel basé sur les préférences
if show_graphs: st.success("📊 Section graphiques activée")
if show_tables: st.success("📋 Section tableaux activée") 
if show_maps: st.success("🗺️ Section cartes activée")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    show_details = st.checkbox("📋 Afficher les détails")
    
    if show_details:
        st.info("🔍 Mode détaillé activé")
        st.write("• Information 1")
        st.write("• Information 2") 
        st.write("• Information 3")
    
    debug_mode = st.checkbox("🐛 Mode debug", value=True)
    
    dark_mode = st.toggle("🌙 Mode sombre")
    
    show_advanced = st.checkbox("⚙️ Options avancées")
    
    if show_advanced:
        st.slider("Paramètre expert", 0, 100, 50)
        st.selectbox("Configuration", ["Mode A", "Mode B"])
    
    st.markdown("**🎛️ Préférences d'affichage :**")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        show_graphs = st.checkbox("📊 Graphiques")
    with col_b: 
        show_tables = st.checkbox("📋 Tableaux", value=True)
    with col_c:
        show_maps = st.checkbox("🗺️ Cartes")
    
    if show_graphs: st.success("📊 Section graphiques activée")
    if show_tables: st.success("📋 Section tableaux activée") 
    if show_maps: st.success("🗺️ Section cartes activée")

st.markdown("""
**🔍 Conseils d'utilisation :**
- **Checkbox** : Interface familière, reconnaissable par tous les utilisateurs
- **Toggle** : Plus moderne et attrayant visuellement, idéal pour les apps mobile-friendly
- **value=True** : Définit l'état initial (coché/activé par défaut)
- **Persistance** : L'état reste identique tant que l'utilisateur ne le change pas
- **Affichage conditionnel** : Parfait pour montrer/cacher du contenu dynamiquement
""")

st.divider()

# ================================
# 3. SÉLECTEURS ET LISTES
# ================================
st.subheader("3️⃣ Sélecteurs et listes")

st.markdown("""
**📖 Description :**
Les sélecteurs permettent de choisir parmi des options prédéfinies avec différentes interfaces.
Chaque type a ses avantages selon le nombre d'options et l'usage.

**🎯 Widgets de sélection disponibles :**
- `st.selectbox()` → Liste déroulante (économise l'espace) 📋
- `st.radio()` → Boutons radio (toutes options visibles) 🔘 
- `st.multiselect()` → Sélection multiple (plusieurs choix) ☑️

**💡 Comment choisir le bon widget :**
- **2-4 options** → `radio` (tout visible)
- **5+ options** → `selectbox` (gain d'espace)
- **Sélection multiple** → `multiselect` (plusieurs valeurs)
- **Catégories** → `selectbox` avec groupes
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Liste déroulante - économise l'espace
ville = st.selectbox("🏙️ Choisissez votre ville", 
    ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes"],
    index=0  # Paris sélectionné par défaut
)

# Boutons radio - toutes options visibles
transport = st.radio("🚀 Mode de transport préféré", 
    ["🚗 Voiture", "🚆 Train", "✈️ Avion", "🚲 Vélo"],
    index=1  # Train par défaut
)

# Sélection multiple - plusieurs choix possibles  
competences = st.multiselect("💻 Vos compétences techniques",
    ["Python", "JavaScript", "Java", "C++", "R", "SQL", "HTML/CSS"],
    default=["Python", "SQL"]  # Pré-sélectionnées
)

# Sélecteur avec options dynamiques
nb_items = st.selectbox("📊 Nombre d'éléments à afficher", 
    [5, 10, 20, 50, 100]
)

# Utilisation des valeurs sélectionnées
st.markdown("**📝 Résumé de vos choix :**")
st.write(f"📍 Ville : {ville}")
st.write(f"🚀 Transport : {transport}")
st.write(f"💻 Compétences : {', '.join(competences) if competences else 'Aucune'}")
st.write(f"📊 Affichage : {nb_items} éléments")

# Sélection conditionnelle
if ville == "Paris":
    arrondissement = st.selectbox("🏛️ Arrondissement", 
        [f"{i}ème" for i in range(1, 21)]
    )
    st.write(f"📍 Localisation précise : Paris {arrondissement}")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    ville = st.selectbox("🏙️ Choisissez votre ville", 
        ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes"],
        index=0
    )
    
    transport = st.radio("🚀 Mode de transport préféré", 
        ["🚗 Voiture", "🚆 Train", "✈️ Avion", "🚲 Vélo"],
        index=1
    )
    
    competences = st.multiselect("💻 Vos compétences techniques",
        ["Python", "JavaScript", "Java", "C++", "R", "SQL", "HTML/CSS"],
        default=["Python", "SQL"]
    )
    
    nb_items = st.selectbox("📊 Nombre d'éléments à afficher", 
        [5, 10, 20, 50, 100]
    )
    
    st.markdown("**📝 Résumé de vos choix :**")
    st.write(f"📍 Ville : {ville}")
    st.write(f"🚀 Transport : {transport}")
    st.write(f"💻 Compétences : {', '.join(competences) if competences else 'Aucune'}")
    st.write(f"📊 Affichage : {nb_items} éléments")
    
    if ville == "Paris":
        arrondissement = st.selectbox("🏛️ Arrondissement", 
            [f"{i}ème" for i in range(1, 21)]
        )
        st.write(f"📍 Localisation précise : Paris {arrondissement}")

st.markdown("""
**🔍 Détails techniques :**
- **index** : Position de l'option sélectionnée par défaut (commence à 0)
- **default** : Liste des valeurs pré-sélectionnées pour `multiselect`
- **Valeur retournée** : String pour `selectbox/radio`, Liste pour `multiselect`
- **Options dynamiques** : Les listes peuvent être générées par du code Python
- **Sélection conditionnelle** : Afficher de nouveaux widgets selon la sélection précédente
""")

st.divider()

# ================================
# 4. CURSEURS ET VALEURS NUMÉRIQUES
# ================================
st.subheader("4️⃣ Curseurs et valeurs numériques")

st.markdown("""
**📖 Description :**
Les widgets numériques permettent de saisir des nombres avec différents niveaux de précision et contraintes.
Essentiels pour les paramètres, filtres, et toute donnée quantitative.

**🎯 Types de widgets numériques :**
- `st.slider()` → Interface visuelle intuitive avec curseur 🎚️
- `st.number_input()` → Saisie précise avec validation 🔢
- Slider simple → Une seule valeur
- Slider range → Plage de valeurs (min, max)

**💡 Avantages de chaque widget :**
- **Slider** : Interface intuitive, limites visuelles claires, adapté mobile
- **Number input** : Précision exacte, validation automatique, saisie rapide au clavier
- **Range slider** : Parfait pour définir des intervalles (prix, dates, etc.)
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Curseur simple - interface intuitive
age = st.slider("👤 Votre âge", 
    min_value=0, max_value=120, value=25, step=1
)

# Curseur avec plage (deux valeurs)
budget = st.slider("💰 Budget disponible (€)", 
    min_value=0, max_value=10000, 
    value=(2000, 8000),  # Tuple pour plage
    step=100
)

# Saisie numérique précise
prix = st.number_input("💵 Prix exact du produit (€)", 
    min_value=0.0, max_value=9999.99, 
    value=19.99, step=0.01,
    format="%.2f"  # 2 décimales
)

# Curseur flottant pour pourcentages
taux = st.slider("📈 Taux d'intérêt (%)", 
    min_value=0.0, max_value=10.0, 
    value=2.5, step=0.1,
    format="%.1f%%"  # Affichage en pourcentage
)

# Nombre entier avec grandes valeurs
population = st.number_input("🏘️ Population de la ville", 
    min_value=1, max_value=50000000,
    value=100000, step=1000
)

# Utilisation des valeurs dans des calculs
st.markdown("**🧮 Calculs automatiques :**")
if budget[0] <= prix <= budget[1]:
    st.success(f"✅ Le prix {prix}€ est dans votre budget {budget[0]}-{budget[1]}€")
else:
    st.error(f"❌ Le prix {prix}€ dépasse votre budget {budget[0]}-{budget[1]}€")

# Calcul d'intérêts
interets = prix * (taux / 100)
st.info(f"💡 Intérêts annuels : {interets:.2f}€ au taux de {taux}%")

# Densité de population
if population > 0 and age > 0:
    ratio = population / 1000
    st.write(f"📊 Ratio : {ratio:.1f}k habitants | Groupe d'âge : {age} ans")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    age = st.slider("👤 Votre âge", 
        min_value=0, max_value=120, value=25, step=1
    )
    
    budget = st.slider("💰 Budget disponible (€)", 
        min_value=0, max_value=10000, 
        value=(2000, 8000),
        step=100
    )
    
    prix = st.number_input("💵 Prix exact du produit (€)", 
        min_value=0.0, max_value=9999.99, 
        value=19.99, step=0.01,
        format="%.2f"
    )
    
    taux = st.slider("📈 Taux d'intérêt (%)", 
        min_value=0.0, max_value=10.0, 
        value=2.5, step=0.1,
        format="%.1f%%"
    )
    
    population = st.number_input("🏘️ Population de la ville", 
        min_value=1, max_value=50000000,
        value=100000, step=1000
    )
    
    st.markdown("**🧮 Calculs automatiques :**")
    if budget[0] <= prix <= budget[1]:
        st.success(f"✅ Le prix {prix}€ est dans votre budget {budget[0]}-{budget[1]}€")
    else:
        st.error(f"❌ Le prix {prix}€ dépasse votre budget {budget[0]}-{budget[1]}€")
    
    interets = prix * (taux / 100)
    st.info(f"💡 Intérêts annuels : {interets:.2f}€ au taux de {taux}%")
    
    if population > 0 and age > 0:
        ratio = population / 1000
        st.write(f"📊 Ratio : {ratio:.1f}k habitants | Groupe d'âge : {age} ans")

st.markdown("""
**🔍 Paramètres importants :**
- **min_value/max_value** : Définit les bornes (obligatoire)
- **value** : Valeur initiale (nombre ou tuple pour range)
- **step** : Incrément lors du déplacement du curseur
- **format** : Formatage d'affichage (%.2f pour 2 décimales, %.1f%% pour pourcentages)
- **Range slider** : Utilisez un tuple (min, max) pour la valeur initiale
""")

st.divider()

# ================================
# 5. CHAMPS DE TEXTE
# ================================
st.subheader("5️⃣ Champs de texte")

st.markdown("""
**📖 Description :**
Les champs de texte permettent la saisie libre de données textuelles avec différents formats et contraintes.
Essentiels pour capturer les informations utilisateur, commentaires, et données non-structurées.

**🎯 Types de champs texte :**
- `st.text_input()` → Ligne unique (noms, emails, mots-clés) 📝
- `st.text_area()` → Multi-lignes (commentaires, descriptions) 📄
- `type="password"` → Masquage pour sécurité 🔒
- `placeholder` → Texte d'aide indicatif 💡

**💡 Bonnes pratiques :**
- Utilisez des placeholders descriptifs
- Validez les données saisies (email, longueur)
- Masquez les informations sensibles (mots de passe)
- Adaptez la taille selon le contenu attendu
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Champ texte simple avec placeholder
nom = st.text_input("👤 Nom complet", 
    placeholder="Ex: Jean Dupont",
    max_chars=50  # Limite de caractères
)

# Email avec validation basique
email = st.text_input("📧 Adresse email", 
    placeholder="exemple@domaine.com"
)

# Mot de passe masqué
password = st.text_input("🔒 Mot de passe", 
    type="password",
    help="Minimum 8 caractères"
)

# Zone de texte multi-lignes
commentaire = st.text_area("💬 Votre avis", 
    placeholder="Partagez votre expérience...",
    height=120,  # Hauteur en pixels
    max_chars=500
)

# Champ avec valeur par défaut
ville_origine = st.text_input("🏙️ Ville d'origine", 
    value="Paris"  # Valeur pré-remplie
)

# Validation en temps réel
if nom:
    if len(nom) < 2:
        st.warning("⚠️ Le nom doit contenir au moins 2 caractères")
    else:
        st.success(f"✅ Bonjour {nom} !")

# Validation email simple
if email:
    if "@" in email and "." in email:
        st.success("✅ Format email valide")
    else:
        st.error("❌ Format email invalide")

# Compteur de caractères pour zone de texte
if commentaire:
    nb_chars = len(commentaire)
    nb_mots = len(commentaire.split())
    st.info(f"📊 {nb_chars} caractères | {nb_mots} mots")
    
    if nb_chars > 400:
        st.warning("⚠️ Commentaire très long")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    nom = st.text_input("👤 Nom complet", 
        placeholder="Ex: Jean Dupont",
        max_chars=50
    )
    
    email = st.text_input("📧 Adresse email", 
        placeholder="exemple@domaine.com"
    )
    
    password = st.text_input("🔒 Mot de passe", 
        type="password",
        help="Minimum 8 caractères"
    )
    
    commentaire = st.text_area("💬 Votre avis", 
        placeholder="Partagez votre expérience...",
        height=120,
        max_chars=500
    )
    
    ville_origine = st.text_input("🏙️ Ville d'origine", 
        value="Paris"
    )
    
    if nom:
        if len(nom) < 2:
            st.warning("⚠️ Le nom doit contenir au moins 2 caractères")
        else:
            st.success(f"✅ Bonjour {nom} !")
    
    if email:
        if "@" in email and "." in email:
            st.success("✅ Format email valide")
        else:
            st.error("❌ Format email invalide")
    
    if commentaire:
        nb_chars = len(commentaire)
        nb_mots = len(commentaire.split())
        st.info(f"📊 {nb_chars} caractères | {nb_mots} mots")
        
        if nb_chars > 400:
            st.warning("⚠️ Commentaire très long")

st.markdown("""
**🔍 Propriétés utiles :**
- **placeholder** : Texte d'aide affiché quand le champ est vide
- **max_chars** : Limite le nombre de caractères saisissables
- **value** : Valeur pré-remplie par défaut
- **height** : Hauteur en pixels pour `text_area`
- **help** : Info-bulle d'aide qui apparaît au survol
- **type="password"** : Masque la saisie avec des points ou astérisques
""")

st.divider()

# ================================
# 6. DATE ET HEURE
# ================================
st.subheader("6️⃣ Sélecteurs de date et heure")

st.markdown("""
**📖 Description :**
Les widgets temporels permettent de sélectionner des dates et heures avec des interfaces calendaires intuitives.
Essentiels pour les plannings, réservations, analyses temporelles, et filtres de données.

**🎯 Widgets temporels disponibles :**
- `st.date_input()` → Sélection de date(s) avec calendrier 📅
- `st.time_input()` → Sélection d'heure précise ⏰
- Support des plages de dates (début, fin)
- Intégration avec le module `datetime` de Python

**💡 Cas d'usage courants :**
- Réservations et plannings
- Filtres de données temporelles
- Rapports sur des périodes spécifiques
- Calculs d'âge, de durée, d'échéances
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
import datetime

# Date simple avec valeur par défaut
today = datetime.date.today()
date_naissance = st.date_input("🎂 Date de naissance", 
    value=datetime.date(1990, 1, 1),
    min_value=datetime.date(1900, 1, 1),
    max_value=today
)

# Heure précise
heure_rdv = st.time_input("⏰ Heure du rendez-vous", 
    value=datetime.time(14, 30),  # 14h30 par défaut
    step=datetime.timedelta(minutes=15)  # Créneaux de 15min
)

# Plage de dates pour vacances/période
periode_vacances = st.date_input("🏖️ Période de vacances",
    value=(today, today + datetime.timedelta(days=7)),
    min_value=today,
    max_value=today + datetime.timedelta(days=365)
)

# Date limite/échéance
echeance = st.date_input("📋 Date limite du projet",
    value=today + datetime.timedelta(days=30),
    min_value=today
)

# Calculs automatiques avec les dates
st.markdown("**🧮 Calculs temporels :**")

# Calcul de l'âge
if date_naissance:
    age = today.year - date_naissance.year
    if today.month < date_naissance.month or (today.month == date_naissance.month and today.day < date_naissance.day):
        age -= 1
    st.write(f"👤 Âge calculé : {age} ans")

# Durée des vacances
if len(periode_vacances) == 2:
    duree = (periode_vacances[1] - periode_vacances[0]).days + 1
    st.write(f"🏖️ Durée des vacances : {duree} jours")
    
    # Jours ouvrés (approximation)
    jours_ouvres = duree * 5 // 7
    st.write(f"💼 Jours ouvrés perdus : ~{jours_ouvres} jours")

# Temps restant jusqu'à l'échéance
jours_restants = (echeance - today).days
if jours_restants > 0:
    st.info(f"⏳ {jours_restants} jours restants jusqu'à l'échéance")
elif jours_restants == 0:
    st.warning("⚠️ Échéance aujourd'hui !")
else:
    st.error(f"❌ Échéance dépassée de {abs(jours_restants)} jours")

# Combinaison date + heure
datetime_complet = datetime.datetime.combine(echeance, heure_rdv)
st.write(f"🗓️ RDV complet : {datetime_complet.strftime('%d/%m/%Y à %H:%M')}")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    today = datetime.date.today()
    date_naissance = st.date_input("🎂 Date de naissance", 
        value=datetime.date(1990, 1, 1),
        min_value=datetime.date(1900, 1, 1),
        max_value=today
    )
    
    heure_rdv = st.time_input("⏰ Heure du rendez-vous", 
        value=datetime.time(14, 30),
        step=datetime.timedelta(minutes=15)
    )
    
    periode_vacances = st.date_input("🏖️ Période de vacances",
        value=(today, today + datetime.timedelta(days=7)),
        min_value=today,
        max_value=today + datetime.timedelta(days=365)
    )
    
    echeance = st.date_input("📋 Date limite du projet",
        value=today + datetime.timedelta(days=30),
        min_value=today
    )
    
    st.markdown("**🧮 Calculs temporels :**")
    
    if date_naissance:
        age = today.year - date_naissance.year
        if today.month < date_naissance.month or (today.month == date_naissance.month and today.day < date_naissance.day):
            age -= 1
        st.write(f"👤 Âge calculé : {age} ans")
    
    if len(periode_vacances) == 2:
        duree = (periode_vacances[1] - periode_vacances[0]).days + 1
        st.write(f"🏖️ Durée des vacances : {duree} jours")
        
        jours_ouvres = duree * 5 // 7
        st.write(f"💼 Jours ouvrés perdus : ~{jours_ouvres} jours")
    
    jours_restants = (echeance - today).days
    if jours_restants > 0:
        st.info(f"⏳ {jours_restants} jours restants jusqu'à l'échéance")
    elif jours_restants == 0:
        st.warning("⚠️ Échéance aujourd'hui !")
    else:
        st.error(f"❌ Échéance dépassée de {abs(jours_restants)} jours")
    
    datetime_complet = datetime.datetime.combine(echeance, heure_rdv)
    st.write(f"🗓️ RDV complet : {datetime_complet.strftime('%d/%m/%Y à %H:%M')}")

st.markdown("""
**🔍 Fonctionnalités avancées :**
- **min_value/max_value** : Limite les dates sélectionnables (ex: pas de date future pour naissance)
- **Plage de dates** : Tuple (début, fin) pour sélectionner une période
- **step** : Intervalle pour les heures (ex: créneaux de 15 minutes)
- **Calculs temporels** : Utilisez les opérateurs `+`, `-` avec `timedelta`
- **Formatage** : `strftime()` pour personnaliser l'affichage des dates
""")

st.divider()

# ================================
# 7. UPLOAD DE FICHIERS
# ================================
st.subheader("7️⃣ Upload de fichiers")

st.markdown("""
**📖 Description :**
`st.file_uploader()` permet aux utilisateurs de télécharger des fichiers directement dans votre application.
Le fichier est chargé en mémoire et accessible pour traitement immédiat.
Essentiel pour les applications de traitement de données, d'images, de documents.

**🎯 Capacités principales :**
- Upload simple ou multiple
- Filtrage par type de fichier (extensions)
- Accès aux métadonnées (nom, taille, type MIME)
- Contenu accessible via `.getvalue()` ou `.read()`

**💡 Types de fichiers supportés :**
- **Images** : PNG, JPG, JPEG, GIF, SVG
- **Documents** : PDF, TXT, CSV, XLSX, DOCX
- **Code** : PY, JS, HTML, CSS, JSON
- **Tous** : Laisser `type=None` pour accepter tout format
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Upload simple avec types spécifiés
fichier = st.file_uploader("📁 Choisissez un fichier", 
    type=['txt', 'csv', 'pdf', 'xlsx'],
    help="Formats acceptés: TXT, CSV, PDF, XLSX"
)

# Upload d'images avec prévisualisation
image = st.file_uploader("🖼️ Sélectionnez une image",
    type=['png', 'jpg', 'jpeg', 'gif']
)

# Upload multiple pour traitement par lots
fichiers_multiples = st.file_uploader("📂 Sélection multiple",
    accept_multiple_files=True,
    type=['csv', 'xlsx']
)

# Analyse du fichier uploadé
if fichier is not None:
    # Métadonnées du fichier
    st.success(f"✅ Fichier reçu : {fichier.name}")
    
    # Informations détaillées
    taille = len(fichier.getvalue())
    st.write(f"📏 Taille : {taille:,} bytes ({taille/1024:.1f} KB)")
    st.write(f"📋 Type MIME : {fichier.type}")
    
    # Traitement selon le type
    if fichier.name.endswith('.txt'):
        contenu = fichier.getvalue().decode('utf-8')
        nb_lignes = len(contenu.split('\\n'))
        nb_mots = len(contenu.split())
        st.info(f"📄 Fichier texte : {nb_lignes} lignes, {nb_mots} mots")
        
        # Aperçu du contenu
        if len(contenu) > 200:
            apercu = contenu[:200] + "..."
        else:
            apercu = contenu
        st.text_area("👀 Aperçu", apercu, height=100)
    
    elif fichier.name.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(fichier)
        st.info(f"📊 CSV : {len(df)} lignes, {len(df.columns)} colonnes")
        st.dataframe(df.head())  # Aperçu des 5 premières lignes

# Prévisualisation d'image
if image is not None:
    st.success(f"🖼️ Image reçue : {image.name}")
    st.image(image, caption=f"Image : {image.name}", width=300)
    
    # Informations sur l'image
    taille_img = len(image.getvalue())
    st.write(f"📐 Taille fichier : {taille_img/1024:.1f} KB")

# Traitement des fichiers multiples
if fichiers_multiples:
    st.success(f"📂 {len(fichiers_multiples)} fichiers reçus")
    
    taille_totale = 0
    for fichier_mult in fichiers_multiples:
        taille = len(fichier_mult.getvalue())
        taille_totale += taille
        st.write(f"• {fichier_mult.name} ({taille/1024:.1f} KB)")
    
    st.info(f"📊 Taille totale : {taille_totale/1024:.1f} KB")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    fichier = st.file_uploader("📁 Choisissez un fichier", 
        type=['txt', 'csv', 'pdf', 'xlsx'],
        help="Formats acceptés: TXT, CSV, PDF, XLSX"
    )
    
    image = st.file_uploader("🖼️ Sélectionnez une image",
        type=['png', 'jpg', 'jpeg', 'gif']
    )
    
    fichiers_multiples = st.file_uploader("📂 Sélection multiple",
        accept_multiple_files=True,
        type=['csv', 'xlsx']
    )
    
    if fichier is not None:
        st.success(f"✅ Fichier reçu : {fichier.name}")
        
        taille = len(fichier.getvalue())
        st.write(f"📏 Taille : {taille:,} bytes ({taille/1024:.1f} KB)")
        st.write(f"📋 Type MIME : {fichier.type}")
        
        if fichier.name.endswith('.txt'):
            contenu = fichier.getvalue().decode('utf-8')
            nb_lignes = len(contenu.split('\n'))
            nb_mots = len(contenu.split())
            st.info(f"📄 Fichier texte : {nb_lignes} lignes, {nb_mots} mots")
            
            if len(contenu) > 200:
                apercu = contenu[:200] + "..."
            else:
                apercu = contenu
            st.text_area("👀 Aperçu", apercu, height=100)
        
        elif fichier.name.endswith('.csv'):
            try:
                import pandas as pd
                df = pd.read_csv(fichier)
                st.info(f"📊 CSV : {len(df)} lignes, {len(df.columns)} colonnes")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"❌ Erreur lecture CSV : {e}")
    
    if image is not None:
        st.success(f"🖼️ Image reçue : {image.name}")
        st.image(image, caption=f"Image : {image.name}", width=300)
        
        taille_img = len(image.getvalue())
        st.write(f"📐 Taille fichier : {taille_img/1024:.1f} KB")
    
    if fichiers_multiples:
        st.success(f"📂 {len(fichiers_multiples)} fichiers reçus")
        
        taille_totale = 0
        for fichier_mult in fichiers_multiples:
            taille = len(fichier_mult.getvalue())
            taille_totale += taille
            st.write(f"• {fichier_mult.name} ({taille/1024:.1f} KB)")
        
        st.info(f"📊 Taille totale : {taille_totale/1024:.1f} KB")

st.markdown("""
**🔍 Méthodes importantes :**
- **fichier.name** : Nom du fichier original
- **fichier.type** : Type MIME (ex: 'text/plain', 'image/png')
- **fichier.getvalue()** : Contenu complet en bytes
- **fichier.read()** : Alternative à getvalue() (lecture une seule fois)
- **accept_multiple_files=True** : Permet la sélection multiple
- **type=['ext1', 'ext2']** : Filtre les extensions autorisées
""")

st.markdown("---")

st.success("🎉 **Félicitations !** Vous maîtrisez maintenant tous les widgets interactifs de Streamlit !")

st.markdown("""
**📚 Récapitulatif des widgets essentiels :**

🔲 **Boutons** → Actions ponctuelles et déclencheurs
☑️ **Cases/Toggle** → États binaires persistants  
📋 **Sélecteurs** → Choix dans des listes prédéfinies
🎚️ **Curseurs** → Valeurs numériques avec contraintes
📝 **Texte** → Saisie libre avec validation
📅 **Date/Heure** → Sélection temporelle précise
📁 **Upload** → Traitement de fichiers utilisateur

**🚀 Prochaine étape :** Explorez le module T_03 sur les graphiques et visualisations !
""")

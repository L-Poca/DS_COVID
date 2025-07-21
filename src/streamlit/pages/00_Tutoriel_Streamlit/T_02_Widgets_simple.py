import streamlit as st

st.header("🎛️ T_02 - Widgets Interactifs (Version Simple)")

st.markdown("**📋 Objectif :** Apprendre les widgets de base pour rendre votre application interactive - boutons, curseurs, zones de texte.")

st.markdown("---")

# ================================
# 1. BOUTONS SIMPLES
# ================================
st.subheader("1️⃣ Boutons")

st.markdown("""
**📖 Explication simple :**
Les boutons permettent à l'utilisateur de déclencher des actions.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Bouton simple")
    st.code("""
if st.button("Cliquez-moi"):
    st.write("Bouton cliqué !")
""")
    
    st.markdown("#### 💻 Code - Bouton coloré")
    st.code("""
if st.button("Bouton important", type="primary"):
    st.success("Action importante !")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    if st.button("Cliquez-moi", key="demo_button_1"):
        st.write("Bouton cliqué !")
    
    if st.button("Bouton important", type="primary", key="demo_button_2"):
        st.success("Action importante !")

st.divider()

# ================================
# 2. ZONES DE TEXTE
# ================================
st.subheader("2️⃣ Saisie de texte")

st.markdown("""
**📖 Explication simple :**
Permet à l'utilisateur de taper du texte.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Texte court")
    st.code("""
nom = st.text_input("Votre nom:")
if nom:
    st.write(f"Bonjour {nom} !")
""")
    
    st.markdown("#### 💻 Code - Texte long")
    st.code("""
message = st.text_area("Votre message:")
if message:
    st.write(f"Vous avez écrit: {message}")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    nom = st.text_input("Votre nom:", key="demo_name")
    if nom:
        st.write(f"Bonjour {nom} !")
    
    message = st.text_area("Votre message:", key="demo_message")
    if message:
        st.write(f"Vous avez écrit: {message}")

st.divider()

# ================================
# 3. NOMBRES
# ================================
st.subheader("3️⃣ Saisie de nombres")

st.markdown("""
**📖 Explication simple :**
Pour que l'utilisateur puisse entrer des nombres.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Nombre simple")
    st.code("""
age = st.number_input("Votre âge:", min_value=0, max_value=120)
st.write(f"Vous avez {age} ans")
""")
    
    st.markdown("#### 💻 Code - Curseur")
    st.code("""
temperature = st.slider("Température:", 0, 40, 20)
st.write(f"Il fait {temperature}°C")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    age = st.number_input("Votre âge:", min_value=0, max_value=120, key="demo_age")
    st.write(f"Vous avez {age} ans")
    
    temperature = st.slider("Température:", 0, 40, 20, key="demo_temp")
    st.write(f"Il fait {temperature}°C")

st.divider()

# ================================
# 4. CHOIX MULTIPLES
# ================================
st.subheader("4️⃣ Menus de choix")

st.markdown("""
**📖 Explication simple :**
Pour proposer des choix à l'utilisateur.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Menu déroulant")
    st.code("""
couleur = st.selectbox(
    "Votre couleur préférée:",
    ["Rouge", "Bleu", "Vert", "Jaune"]
)
st.write(f"Vous aimez le {couleur}")
""")
    
    st.markdown("#### 💻 Code - Boutons radio")
    st.code("""
animal = st.radio(
    "Votre animal préféré:",
    ["🐶 Chien", "🐱 Chat", "🐦 Oiseau"]
)
st.write(f"Vous préférez: {animal}")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    couleur = st.selectbox(
        "Votre couleur préférée:",
        ["Rouge", "Bleu", "Vert", "Jaune"],
        key="demo_couleur"
    )
    st.write(f"Vous aimez le {couleur}")
    
    animal = st.radio(
        "Votre animal préféré:",
        ["🐶 Chien", "🐱 Chat", "🐦 Oiseau"],
        key="demo_animal"
    )
    st.write(f"Vous préférez: {animal}")

st.divider()

# ================================
# 5. CASES À COCHER
# ================================
st.subheader("5️⃣ Cases à cocher")

st.markdown("""
**📖 Explication simple :**
Pour des choix oui/non ou des sélections multiples.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Case simple")
    st.code("""
accord = st.checkbox("J'accepte les conditions")
if accord:
    st.success("Merci pour votre accord !")
""")
    
    st.markdown("#### 💻 Code - Choix multiples")
    st.code("""
sports = st.multiselect(
    "Sports pratiqués:",
    ["⚽ Football", "🏀 Basketball", "🎾 Tennis", "🏊 Natation"]
)
if sports:
    st.write(f"Vous pratiquez: {', '.join(sports)}")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    accord = st.checkbox("J'accepte les conditions", key="demo_accord")
    if accord:
        st.success("Merci pour votre accord !")
    
    sports = st.multiselect(
        "Sports pratiqués:",
        ["⚽ Football", "🏀 Basketball", "🎾 Tennis", "🏊 Natation"],
        key="demo_sports"
    )
    if sports:
        st.write(f"Vous pratiquez: {', '.join(sports)}")

st.divider()

# ================================
# 6. EXERCICE PRATIQUE
# ================================
st.subheader("6️⃣ Exercice pratique")

st.markdown("""
**🎯 À vous de jouer !**
Créez un formulaire simple pour recueillir des informations.
""")

with st.expander("📝 Exercice : Créer un formulaire"):
    st.markdown("""
    **Créez un formulaire avec :**
    
    1. Nom et prénom (texte)
    2. Âge (nombre)
    3. Ville (menu déroulant)
    4. Hobbies (choix multiples)
    5. Bouton pour valider
    
    **Exemple de code :**
    """)
    
    st.code("""
st.subheader("📝 Formulaire d'inscription")

nom = st.text_input("Nom:")
prenom = st.text_input("Prénom:")
age = st.number_input("Âge:", min_value=0, max_value=120)

ville = st.selectbox("Ville:", ["Paris", "Lyon", "Marseille", "Toulouse"])

hobbies = st.multiselect("Hobbies:", [
    "🎵 Musique", "📚 Lecture", "🏃 Sport", "🎨 Art", "💻 Informatique"
])

if st.button("✅ Valider"):
    if nom and prenom:
        st.success("Formulaire envoyé !")
        st.write(f"**Nom:** {nom} {prenom}")
        st.write(f"**Âge:** {age} ans")
        st.write(f"**Ville:** {ville}")
        if hobbies:
            st.write(f"**Hobbies:** {', '.join(hobbies)}")
    else:
        st.error("Veuillez remplir le nom et prénom !")
""")

# Zone de test pour l'utilisateur
st.markdown("**💻 Testez votre formulaire ici :**")

st.markdown("**Voici un exemple fonctionnel :**")

with st.form("formulaire_exemple"):
    st.markdown("#### 📝 Formulaire d'inscription")
    
    nom = st.text_input("Nom:")
    prenom = st.text_input("Prénom:")
    age = st.number_input("Âge:", min_value=0, max_value=120, value=25)
    
    ville = st.selectbox("Ville:", ["Paris", "Lyon", "Marseille", "Toulouse"])
    
    hobbies = st.multiselect("Hobbies:", [
        "🎵 Musique", "📚 Lecture", "🏃 Sport", "🎨 Art", "💻 Informatique"
    ])
    
    valider = st.form_submit_button("✅ Valider")
    
    if valider:
        if nom and prenom:
            st.success("Formulaire envoyé !")
            st.write(f"**Nom:** {nom} {prenom}")
            st.write(f"**Âge:** {age} ans")
            st.write(f"**Ville:** {ville}")
            if hobbies:
                st.write(f"**Hobbies:** {', '.join(hobbies)}")
        else:
            st.error("Veuillez remplir le nom et prénom !")

st.divider()

# ================================
# 7. RÉCAPITULATIF
# ================================
st.subheader("7️⃣ Récapitulatif")

st.markdown("""
**🎓 Ce que vous avez appris :**

✅ **Boutons :** `st.button()` pour déclencher des actions  
✅ **Texte :** `st.text_input()` et `st.text_area()` pour la saisie  
✅ **Nombres :** `st.number_input()` et `st.slider()` pour les valeurs numériques  
✅ **Choix :** `st.selectbox()` et `st.radio()` pour sélectionner  
✅ **Cases :** `st.checkbox()` et `st.multiselect()` pour les options  
✅ **Formulaires :** `st.form()` pour grouper les widgets  

**🚀 Prochaine étape :** T_03 - Graphiques (créer des visualisations)
""")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⬅️ Module précédent (T_01)", key="nav_prev"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_01 - Texte")

with col2:
    if st.button("📚 Retour au sommaire", key="nav_home"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_00 - Sommaire")

with col3:
    if st.button("➡️ Module suivant (T_03)", key="nav_next"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_03 - Graphiques")

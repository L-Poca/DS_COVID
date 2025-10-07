import streamlit as st

st.header("📝 T_01 - Affichage de Texte & Markdown (Version Simple)")

st.markdown("**📋 Objectif :** Apprendre à afficher du texte et des titres dans Streamlit - Les bases essentielles pour débuter.")

st.markdown("---")

# ================================
# 1. TITRES SIMPLES
# ================================
st.subheader("1️⃣ Les différents types de titres")

st.markdown("""
**📖 Explication simple :**
Streamlit propose plusieurs types de titres pour organiser votre page.
C'est comme dans un livre : titre principal, chapitres, sous-chapitres.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Titre principal (le plus gros)
st.title("Mon application")

# Titre de section 
st.header("Section importante")

# Sous-titre
st.subheader("Sous-section")

# Texte normal
st.write("Texte normal")

# Petite note
st.caption("Petite explication")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Démonstration dans une boîte
    with st.container():
        st.markdown("**Aperçu des titres :**")
        st.markdown("# Mon application")
        st.markdown("## Section importante") 
        st.markdown("### Sous-section")
        st.write("Texte normal")
        st.caption("Petite explication")

st.divider()

# ================================
# 2. TEXTE MARKDOWN SIMPLE
# ================================
st.subheader("2️⃣ Formatage de texte avec Markdown")

st.markdown("""
**📖 Explication simple :**
Markdown permet de formater le texte facilement : gras, italique, listes, etc.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Texte en gras")
    st.code("""
st.markdown("**Texte en gras**")
""")
    
    st.markdown("#### 💻 Code - Texte en italique")
    st.code("""
st.markdown("*Texte en italique*")
""")
    
    st.markdown("#### 💻 Code - Liste simple")
    st.code("""
st.markdown('''
- Premier élément
- Deuxième élément
- Troisième élément
''')
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    st.markdown("**Texte en gras**")
    st.markdown("*Texte en italique*")
    st.markdown("""
    **Liste :**
    - Premier élément
    - Deuxième élément
    - Troisième élément
    """)

st.divider()

# ================================
# 3. COULEURS ET MESSAGES
# ================================
st.subheader("3️⃣ Messages colorés")

st.markdown("""
**📖 Explication simple :**
Streamlit propose des messages colorés pour attirer l'attention.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Message d'information")
    st.code("""
st.info("Information importante")
""")
    
    st.markdown("#### 💻 Code - Message de succès")
    st.code("""
st.success("Opération réussie !")
""")
    
    st.markdown("#### 💻 Code - Message d'avertissement")
    st.code("""
st.warning("Attention à ceci")
""")
    
    st.markdown("#### 💻 Code - Message d'erreur")
    st.code("""
st.error("Quelque chose ne va pas")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    st.info("Information importante")
    st.success("Opération réussie !")
    st.warning("Attention à ceci")
    st.error("Quelque chose ne va pas")

st.divider()

# ================================
# 4. ÉMOJIS SIMPLES
# ================================
st.subheader("4️⃣ Utiliser des émojis")

st.markdown("""
**📖 Explication simple :**
Les émojis rendent votre application plus sympa et facile à comprendre.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
st.write("🎉 Félicitations !")
st.write("📊 Mes données")
st.write("⚠️ Important")
st.write("✅ Terminé")
st.write("❌ Erreur")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    st.write("🎉 Félicitations !")
    st.write("📊 Mes données")
    st.write("⚠️ Important")
    st.write("✅ Terminé")
    st.write("❌ Erreur")

st.divider()

# ================================
# 5. EXERCICE PRATIQUE
# ================================
st.subheader("5️⃣ Exercice pratique")

st.markdown("""
**🎯 À vous de jouer !**
Créez votre propre page avec les éléments suivants :
""")

with st.expander("📝 Exercice : Créer ma première page"):
    st.markdown("""
    **Créez une page qui contient :**
    
    1. Un titre principal avec votre nom
    2. Une section "À propos de moi"
    3. Une liste de vos hobbies
    4. Un message de succès
    5. Des émojis pour décorer
    
    **Exemple de code :**
    """)
    
    st.code("""
st.title("👋 Bienvenue chez Jean Dupont")

st.header("📖 À propos de moi")
st.write("Je suis passionné par la programmation et les données.")

st.subheader("🎯 Mes hobbies")
st.markdown('''
- 📚 Lecture
- 🏃 Course à pied  
- 🎵 Musique
- 💻 Programmation
''')

st.success("✅ Merci d'avoir visité ma page !")
""")

# Zone de test pour l'utilisateur
st.markdown("**💻 Testez votre code ici :**")

user_code = st.text_area(
    "Écrivez votre code ici:",
    placeholder="st.title('Mon titre')",
    height=200
)

if st.button("▶️ Tester mon code"):
    if user_code:
        try:
            # Exécuter le code utilisateur de manière sécurisée
            exec(user_code)
        except Exception as e:
            st.error(f"Erreur dans votre code : {e}")
    else:
        st.warning("Écrivez du code avant de tester !")

st.divider()

# ================================
# 6. RÉCAPITULATIF
# ================================
st.subheader("6️⃣ Récapitulatif")

st.markdown("""
**🎓 Ce que vous avez appris :**

✅ **Titres :** `st.title()`, `st.header()`, `st.subheader()`  
✅ **Texte :** `st.write()`, `st.markdown()`  
✅ **Messages colorés :** `st.info()`, `st.success()`, `st.warning()`, `st.error()`  
✅ **Formatage :** **gras**, *italique*, listes  
✅ **Émojis :** Pour rendre l'interface plus sympa  

**🚀 Prochaine étape :** T_02 - Widgets (boutons, curseurs, etc.)
""")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📚 Retour au sommaire"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_00 - Sommaire")

with col2:
    st.write("") # Espace

with col3:
    if st.button("➡️ Module suivant (T_02)"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_02 - Widgets")

import streamlit as st

st.header("📝 T_01 - Affichage de Texte & Markdown")

st.markdown("**📋 Objectif :** Apprendre à afficher du texte, des titres et du contenu formaté avec Markdown dans Streamlit.")

st.markdown("---")

# ================================
# 1. TITRES ET HIÉRARCHIE
# ================================
st.subheader("1️⃣ Titres et hiérarchie")

st.markdown("""
**📖 Description :**
Les titres permettent de structurer votre application avec une hiérarchie visuelle claire. 
Streamlit propose plusieurs niveaux pour organiser votre contenu de manière logique.
C'est comme créer un plan de document : titre principal, chapitres, sous-chapitres, etc.

**🎯 Utilisations principales :**
- `st.title()` → Titre principal de l'application (le plus grand)
- `st.header()` → Sections principales (chapitres)
- `st.subheader()` → Sous-sections (sous-chapitres)  
- `st.write()` → Texte polyvalent (accepte markdown, dataframes, etc.)
- `st.caption()` → Notes explicatives ou légendes (plus petit)

**💡 Conseil :** Utilisez une hiérarchie cohérente dans toute votre application pour une navigation intuitive.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Titre principal (le plus grand)
st.title("Mon Application")

# Titre de section (plus petit que title)
st.header("Ma Section")

# Sous-titre (plus petit que header)
st.subheader("Ma Sous-Section")

# Texte normal (taille standard)
st.write("Du texte simple")

# Petite note (plus petit que write)
st.caption("Note explicative")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    st.title("Mon Application")
    st.header("Ma Section")
    st.subheader("Ma Sous-Section")
    st.write("Du texte simple")
    st.caption("Note explicative")

st.markdown("""
**🔍 Analyse du code :**
- Chaque fonction crée une taille de texte différente
- L'ordre d'importance : `title` > `header` > `subheader` > `write` > `caption`
- `st.write()` est polyvalent : il accepte du texte, du Markdown, des DataFrames, etc.
- `st.caption()` est parfait pour les notes, sources, ou explications courtes
""")

st.divider()

# ================================
# 2. TEXTE FORMATÉ ET MARKDOWN
# ================================
st.subheader("2️⃣ Texte formaté et Markdown")

st.markdown("""
**📖 Description :**
Markdown est un langage de balisage léger qui permet de formater du texte facilement.
Streamlit supporte la syntaxe Markdown standard pour créer du contenu riche.
C'est comme utiliser les boutons de formatage dans Word, mais avec du code !

**🎯 Syntaxes essentielles :**
- `**gras**` → **gras** (pour mettre en évidence)
- `*italique*` → *italique* (pour l'emphase)
- `~~barré~~` → ~~barré~~ (pour indiquer une suppression)
- `` `code` `` → `code` (pour les noms de variables/fonctions)
- `> citation` → bloc de citation (pour citer du contenu)

**💡 Astuce :** Le Markdown est plus rapide à écrire que du HTML et plus lisible en mode code.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Texte avec formatage Markdown
st.markdown("Voici du **texte en gras**")
st.markdown("Et du *texte en italique*")
st.markdown("Texte ~~barré~~")
st.markdown("Mot en `code`")

# Citation (indentée avec une barre verticale)
st.markdown("> Ceci est une citation")

# Liste à puces (non ordonnée)
st.markdown('''
- Élément 1
- Élément 2  
- Élément 3
''')

# Liste numérotée (ordonnée)
st.markdown('''
1. Premier élément
2. Deuxième élément
3. Troisième élément
''')
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    st.markdown("Voici du **texte en gras**")
    st.markdown("Et du *texte en italique*")
    st.markdown("Texte ~~barré~~")
    st.markdown("Mot en `code`")
    st.markdown("> Ceci est une citation")
    st.markdown('''
- Élément 1
- Élément 2
- Élément 3
''')
    st.markdown('''
1. Premier élément
2. Deuxième élément
3. Troisième élément
''')

st.markdown("""
**🔍 Analyse détaillée :**
- `st.markdown()` interprète le formatage Markdown automatiquement
- Utilisez `**` pour le gras (équivalent à `<strong>` en HTML)
- Utilisez `*` pour l'italique (équivalent à `<em>` en HTML)
- Les backticks `` ` `` créent un style monospace pour le code
- `>` au début d'une ligne crée une citation indentée
- `-` ou `*` créent des listes à puces, les chiffres créent des listes numérotées
""")

st.divider()

# ================================
# 3. LIENS ET IMAGES
# ================================
st.subheader("3️⃣ Liens et images")

st.markdown("""
**📖 Description :**
Markdown permet d'intégrer facilement des liens cliquables et des images dans votre application.
C'est essentiel pour créer une expérience interactive et référencer des sources externes.

**🎯 Syntaxes importantes :**
- `[Texte du lien](URL)` → Crée un lien cliquable
- `![Texte alternatif](URL_image)` → Affiche une image
- Les images peuvent venir d'URLs externes ou de fichiers locaux

**💡 Bonnes pratiques :**
- Utilisez des textes de liens descriptifs (évitez "cliquez ici")
- Ajoutez toujours un texte alternatif pour les images (accessibilité)
- Testez vos liens pour vous assurer qu'ils fonctionnent
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Lien cliquable avec texte descriptif
st.markdown("[📚 Documentation Streamlit](https://streamlit.io)")

# Lien vers une page spécifique
st.markdown("[🎯 Guide installation](https://docs.streamlit.io/get-started)")

# Image depuis une URL (avec texte alternatif)
st.markdown("![Logo Streamlit](https://docs.streamlit.io/logo.svg)")

# Lien et image combinés (image cliquable)
st.markdown("[![Streamlit](https://docs.streamlit.io/logo.svg)](https://streamlit.io)")

# Liens avec émojis pour plus d'attrait
st.markdown("🌐 [Site officiel](https://streamlit.io) | 📖 [Documentation](https://docs.streamlit.io)")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    st.markdown("[📚 Documentation Streamlit](https://streamlit.io)")
    st.markdown("[🎯 Guide installation](https://docs.streamlit.io/get-started)")
    st.markdown("![Logo Streamlit](https://docs.streamlit.io/logo.svg)")
    st.markdown("[![Streamlit](https://docs.streamlit.io/logo.svg)](https://streamlit.io)")
    st.markdown("🌐 [Site officiel](https://streamlit.io) | 📖 [Documentation](https://docs.streamlit.io)")

st.markdown("""
**🔍 Explication technique :**
- `[texte](URL)` → Le texte entre crochets devient cliquable
- `![alt](URL)` → Le point d'exclamation indique une image  
- `[![image](URL_img)](URL_lien)` → Image cliquable (image + lien combinés)
- Les URLs peuvent être absolues (https://...) ou relatives (./image.png)
- Streamlit gère automatiquement l'ouverture des liens dans un nouvel onglet
""")

st.divider()

# ================================
# 4. MESSAGES COLORÉS ET ALERTES
# ================================
st.subheader("4️⃣ Messages colorés et alertes")

st.markdown("""
**📖 Description :**
Streamlit fournit des composants spécialisés pour afficher des messages avec des couleurs 
et icônes appropriées selon leur importance. C'est crucial pour la communication avec l'utilisateur.

**🎯 Types de messages disponibles :**
- `st.success()` → Message de succès (vert) ✅
- `st.info()` → Information générale (bleu) ℹ️ 
- `st.warning()` → Avertissement (orange) ⚠️
- `st.error()` → Erreur critique (rouge) ❌
- `st.exception()` → Affichage d'exception avec traceback

**💡 Quand les utiliser :**
- Success : Confirmation d'action réussie
- Info : Conseils, instructions, informations neutres  
- Warning : Attention requise, mais pas critique
- Error : Problème empêchant le fonctionnement
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Message de succès (vert avec icône)
st.success("✅ Opération réussie avec succès")

# Message informatif (bleu avec icône)
st.info("ℹ️ Information importante à retenir")

# Message d'avertissement (orange avec icône)
st.warning("⚠️ Attention requise avant de continuer")

# Message d'erreur (rouge avec icône)
st.error("❌ Erreur critique détectée")

# Affichage d'exception avec traceback
st.exception(Exception("Exemple d'exception"))

# Messages avec contenu Markdown
st.success("**Succès !** Votre fichier a été *téléchargé*")
st.info("**Astuce :** Utilisez `Ctrl+C` pour copier")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    st.success("✅ Opération réussie avec succès")
    st.info("ℹ️ Information importante à retenir")
    st.warning("⚠️ Attention requise avant de continuer")
    st.error("❌ Erreur critique détectée")
    st.exception(Exception("Exemple d'exception"))
    st.success("**Succès !** Votre fichier a été *téléchargé*")
    st.info("**Astuce :** Utilisez `Ctrl+C` pour copier")

st.markdown("""
**🔍 Conseils d'utilisation :**
- **Success** : Confirmations d'actions (sauvegarde, envoi, calcul terminé)
- **Info** : Instructions, conseils, informations neutres
- **Warning** : Situations nécessitant l'attention (données manquantes, paramètres inhabituels)
- **Error** : Problèmes bloquants (fichier introuvable, erreur de format)
- **Exception** : Erreurs de programmation avec détails techniques
- Vous pouvez utiliser du Markdown dans tous ces messages pour plus de richesse
""")

st.divider()

# ================================
# 5. CODE AVEC COLORATION SYNTAXIQUE
# ================================
st.subheader("5️⃣ Code avec coloration syntaxique")

st.markdown("""
**📖 Description :**
`st.code()` permet d'afficher du code avec une coloration syntaxique appropriée selon le langage.
Idéal pour les tutoriels, la documentation, ou afficher des résultats de requêtes.

**🎯 Langages supportés :**
- `python` → Coloration Python (bleu, vert, orange)
- `sql` → Coloration SQL (mots-clés en majuscules)
- `javascript` → Coloration JavaScript  
- `json` → Coloration JSON (structure de données)
- `css` → Coloration CSS (propriétés et valeurs)
- `html` → Coloration HTML (balises et attributs)
- Et bien d'autres...

**💡 Avantages :**
- Police monospace pour une lecture facile
- Coloration automatique selon le langage
- Bouton de copie intégré
- Numérotation des lignes pour les longs codes
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Code Python avec coloration
st.code("print('Hello World')", language="python")

# Code SQL avec coloration
st.code("SELECT * FROM users WHERE age > 18;", language="sql")

# Code JavaScript
st.code("console.log('Hello');", language="javascript")

# Code JSON (données structurées)
st.code('{"name": "Alice", "age": 25}', language="json")

# Code multi-lignes
code_python = '''
def calculate_sum(a, b):
    \"\"\"Calcule la somme de deux nombres\"\"\"
    result = a + b
    return result

total = calculate_sum(10, 20)
print(f"Résultat: {total}")
'''
st.code(code_python, language="python")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    st.code("print('Hello World')", language="python")
    st.code("SELECT * FROM users WHERE age > 18;", language="sql")
    st.code("console.log('Hello');", language="javascript")
    st.code('{"name": "Alice", "age": 25}', language="json")
    
    code_python = """
def calculate_sum(a, b):
    '''Calcule la somme de deux nombres'''
    result = a + b
    return result

total = calculate_sum(10, 20)
print(f"Résultat: {total}")
"""
    st.code(code_python, language="python")

st.markdown("""
**🔍 Différence avec les autres méthodes :**
- `st.code()` : Police monospace + coloration + bouton copie
- `st.text()` : Police monospace simple, sans coloration
- `st.markdown()` avec backticks : Code inline dans le texte
- Utilisez `st.code()` pour des blocs de code complets
""")

st.divider()

# ================================
# 6. SÉPARATEURS ET ORGANISATION
# ================================
st.subheader("6️⃣ Séparateurs et organisation")

st.markdown("""
**📖 Description :**
Les séparateurs aident à organiser visuellement le contenu et améliorer la lisibilité.
Essentiels pour structurer des applications complexes.

**🎯 Types de séparateurs :**
- `st.divider()` → Séparateur Streamlit (recommandé)
- `st.markdown("---")` → Séparateur Markdown
- Espacement personnalisé avec HTML

**💡 Conseil :** Utilisez les séparateurs pour délimiter les sections logiques
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Séparateur Streamlit (moderne)
st.write("Contenu avant")
st.divider()
st.write("Contenu après")

# Séparateur Markdown (classique)
st.write("Avant")
st.markdown("---")
st.write("Après")

# Espacement personnalisé
st.write("Contenu")
st.markdown("<br>", unsafe_allow_html=True)
st.write("Contenu espacé")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    st.write("Contenu avant")
    st.divider()
    st.write("Contenu après")
    st.write("Avant")
    st.markdown("---")
    st.write("Après")
    st.write("Contenu")
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Contenu espacé")

st.markdown("""
**🔍 Quand utiliser chaque séparateur :**
- `st.divider()` : Séparation claire entre sections importantes
- `st.markdown("---")` : Compatible avec tous les environnements Markdown
- HTML personnalisé : Quand vous voulez un style spécifique
- Espacement simple : Pour aérer le contenu sans ligne visible
""")

st.markdown("---")
st.success("🎉 **Félicitations !** Vous maîtrisez maintenant l'affichage de texte et Markdown dans Streamlit !")

st.markdown("""
**📚 Récapitulatif des points clés :**
- Utilisez une hiérarchie claire avec `title`, `header`, `subheader`
- Exploitez la richesse du Markdown pour formater votre contenu
- Intégrez des liens et images pour enrichir l'expérience
- Utilisez les alertes colorées pour communiquer efficacement
- Affichez du code avec `st.code()` et la coloration syntaxique
- Organisez avec des séparateurs pour une meilleure lisibilité

**🚀 Prochaine étape :** Explorez le module T_02 sur les widgets interactifs !
""")

import streamlit as st
import pandas as pd
import time
import json

st.header("🚀 T_07 - Fonctionnalités Avancées (Version Simple)")

st.markdown("**📋 Objectif :** Découvrir les fonctionnalités plus avancées de Streamlit - session state, cache, formulaires - expliquées simplement.")

st.markdown("---")

# ================================
# 1. SESSION STATE (MÉMOIRE DE L'APP)
# ================================
st.subheader("1️⃣ Session State - Mémoire de l'application")

st.markdown("""
**📖 Explication simple :**
Le session state permet à votre application de "se souvenir" des informations.
Sans ça, chaque clic efface tout et repart de zéro.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Compteur simple")
    st.code("""
# Initialiser le compteur s'il n'existe pas
if 'compteur' not in st.session_state:
    st.session_state.compteur = 0

# Afficher la valeur actuelle
st.write(f"Compteur: {st.session_state.compteur}")

# Boutons pour modifier
if st.button("➕ Ajouter 1"):
    st.session_state.compteur += 1

if st.button("➖ Enlever 1"):
    st.session_state.compteur -= 1

if st.button("🔄 Remettre à zéro"):
    st.session_state.compteur = 0
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Initialiser le compteur s'il n'existe pas
    if 'demo_compteur' not in st.session_state:
        st.session_state.demo_compteur = 0
    
    # Afficher la valeur actuelle
    st.write(f"**Compteur:** {st.session_state.demo_compteur}")
    
    # Boutons pour modifier
    button_col1, button_col2, button_col3 = st.columns(3)
    
    with button_col1:
        if st.button("➕ Ajouter 1", key="demo_add"):
            st.session_state.demo_compteur += 1
            st.rerun()
    
    with button_col2:
        if st.button("➖ Enlever 1", key="demo_sub"):
            st.session_state.demo_compteur -= 1
            st.rerun()
    
    with button_col3:
        if st.button("🔄 Reset", key="demo_reset"):
            st.session_state.demo_compteur = 0
            st.rerun()

st.divider()

# ================================
# 2. LISTE DE TÂCHES
# ================================
st.subheader("2️⃣ Liste de tâches avec session state")

st.markdown("""
**📖 Explication simple :**
Exemple pratique : une liste de tâches qui se souvient de ce qu'on a ajouté.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Todo list")
    st.code("""
# Initialiser la liste des tâches
if 'taches' not in st.session_state:
    st.session_state.taches = []

# Zone pour ajouter une nouvelle tâche
nouvelle_tache = st.text_input("Nouvelle tâche:")

if st.button("➕ Ajouter") and nouvelle_tache:
    st.session_state.taches.append(nouvelle_tache)
    st.success(f"Tâche ajoutée: {nouvelle_tache}")

# Afficher toutes les tâches
if st.session_state.taches:
    st.write("**Mes tâches:**")
    for i, tache in enumerate(st.session_state.taches):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"{i+1}. {tache}")
        with col2:
            if st.button("🗑️", key=f"del_{i}"):
                st.session_state.taches.pop(i)
                st.rerun()
else:
    st.write("Aucune tâche pour le moment")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Initialiser la liste des tâches
    if 'demo_taches' not in st.session_state:
        st.session_state.demo_taches = []
    
    # Zone pour ajouter une nouvelle tâche
    nouvelle_tache = st.text_input("Nouvelle tâche:", key="demo_nouvelle_tache")
    
    if st.button("➕ Ajouter", key="demo_add_task") and nouvelle_tache:
        st.session_state.demo_taches.append(nouvelle_tache)
        st.success(f"Tâche ajoutée: {nouvelle_tache}")
        st.rerun()
    
    # Afficher toutes les tâches
    if st.session_state.demo_taches:
        st.write("**Mes tâches:**")
        for i, tache in enumerate(st.session_state.demo_taches):
            task_col1, task_col2 = st.columns([3, 1])
            with task_col1:
                st.write(f"{i+1}. {tache}")
            with task_col2:
                if st.button("🗑️", key=f"demo_del_{i}"):
                    st.session_state.demo_taches.pop(i)
                    st.rerun()
    else:
        st.write("Aucune tâche pour le moment")

st.divider()

# ================================
# 3. CACHE SIMPLE
# ================================
st.subheader("3️⃣ Cache - Accélérer l'application")

st.markdown("""
**📖 Explication simple :**
Le cache évite de refaire les mêmes calculs longs à chaque fois.
Comme mettre en mémoire le résultat d'un calcul compliqué.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Fonction avec cache")
    st.code("""
@st.cache_data  # Cette ligne active le cache
def calcul_long(nombre):
    # Simulation d'un calcul qui prend du temps
    time.sleep(2)  # Attendre 2 secondes
    resultat = nombre * nombre * nombre
    return resultat

# Widget pour choisir un nombre
nombre = st.number_input("Choisissez un nombre:", 1, 100, 5)

# Bouton pour lancer le calcul
if st.button("🧮 Calculer le cube"):
    with st.spinner("Calcul en cours..."):
        resultat = calcul_long(nombre)
        st.success(f"Le cube de {nombre} est {resultat}")
        st.info("💡 La prochaine fois avec le même nombre sera instantané !")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    @st.cache_data  # Cette ligne active le cache
    def calcul_long_demo(nombre):
        # Simulation d'un calcul qui prend du temps
        time.sleep(1)  # Attendre 1 seconde pour la démo
        resultat = nombre * nombre * nombre
        return resultat
    
    # Widget pour choisir un nombre
    nombre = st.number_input("Choisissez un nombre:", 1, 100, 5, key="demo_cache_number")
    
    # Bouton pour lancer le calcul
    if st.button("🧮 Calculer le cube", key="demo_cache_calc"):
        with st.spinner("Calcul en cours..."):
            resultat = calcul_long_demo(nombre)
            st.success(f"Le cube de {nombre} est {resultat}")
            st.info("💡 La prochaine fois avec le même nombre sera instantané !")

st.divider()

# ================================
# 4. FORMULAIRES
# ================================
st.subheader("4️⃣ Formulaires - Grouper les widgets")

st.markdown("""
**📖 Explication simple :**
Les formulaires permettent de grouper plusieurs champs et de les valider d'un coup.
Plus pratique que de valider chaque champ un par un.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Formulaire simple")
    st.code("""
# Créer un formulaire
with st.form("mon_formulaire"):
    st.write("📝 Informations personnelles")
    
    # Les widgets dans le formulaire
    nom = st.text_input("Nom:")
    prenom = st.text_input("Prénom:")
    age = st.number_input("Âge:", 0, 120, 25)
    ville = st.selectbox("Ville:", ["Paris", "Lyon", "Marseille"])
    
    # Bouton de validation du formulaire
    valider = st.form_submit_button("✅ Valider")
    
    # Actions quand on valide
    if valider:
        if nom and prenom:
            st.success("Formulaire validé !")
            st.write(f"Bonjour {prenom} {nom}, {age} ans, de {ville}")
        else:
            st.error("Veuillez remplir nom et prénom")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Créer un formulaire
    with st.form("demo_formulaire"):
        st.write("**📝 Informations personnelles**")
        
        # Les widgets dans le formulaire
        nom = st.text_input("Nom:", key="demo_form_nom")
        prenom = st.text_input("Prénom:", key="demo_form_prenom")
        age = st.number_input("Âge:", 0, 120, 25, key="demo_form_age")
        ville = st.selectbox("Ville:", ["Paris", "Lyon", "Marseille"], key="demo_form_ville")
        
        # Bouton de validation du formulaire
        valider = st.form_submit_button("✅ Valider")
        
        # Actions quand on valide
        if valider:
            if nom and prenom:
                st.success("Formulaire validé !")
                st.write(f"Bonjour {prenom} {nom}, {age} ans, de {ville}")
            else:
                st.error("Veuillez remplir nom et prénom")

st.divider()

# ================================
# 5. PROGRESS BAR
# ================================
st.subheader("5️⃣ Barre de progression")

st.markdown("""
**📖 Explication simple :**
Montrer le progrès d'une tâche qui prend du temps.
Rassure l'utilisateur que quelque chose se passe.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Progress bar")
    st.code("""
# Bouton pour démarrer une tâche longue
if st.button("🚀 Démarrer traitement"):
    # Créer la barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simuler un traitement qui prend du temps
    for i in range(101):
        # Mettre à jour la barre
        progress_bar.progress(i)
        status_text.text(f'Progression: {i}%')
        
        # Attendre un peu
        time.sleep(0.01)
    
    # Finir
    status_text.text('Traitement terminé!')
    st.success("✅ Tâche accomplie !")
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Bouton pour démarrer une tâche longue
    if st.button("🚀 Démarrer traitement", key="demo_progress"):
        # Créer la barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simuler un traitement qui prend du temps
        for i in range(101):
            # Mettre à jour la barre
            progress_bar.progress(i)
            status_text.text(f'Progression: {i}%')
            
            # Attendre un peu (plus rapide pour la démo)
            time.sleep(0.01)
        
        # Finir
        status_text.text('Traitement terminé!')
        st.success("✅ Tâche accomplie !")

st.divider()

# ================================
# 6. SPINNER (INDICATEUR DE CHARGEMENT)
# ================================
st.subheader("6️⃣ Spinner - Indicateur de chargement")

st.markdown("""
**📖 Explication simple :**
Le spinner montre qu'un calcul est en cours sans préciser le temps restant.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code - Spinner")
    st.code("""
if st.button("⏳ Chargement avec spinner"):
    with st.spinner('Opération en cours...'):
        # Simuler une opération
        time.sleep(3)
        
        # Créer des données
        data = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })
    
    # Afficher le résultat
    st.success("Données générées !")
    st.dataframe(data)
""")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    if st.button("⏳ Chargement avec spinner", key="demo_spinner"):
        with st.spinner('Opération en cours...'):
            # Simuler une opération (plus rapide pour la démo)
            time.sleep(1)
            
            # Créer des données
            data = pd.DataFrame({
                'A': [1, 2, 3, 4],
                'B': [10, 20, 30, 40]
            })
        
        # Afficher le résultat
        st.success("Données générées !")
        st.dataframe(data)

st.divider()

# ================================
# 7. EXERCICE PRATIQUE
# ================================
st.subheader("7️⃣ Exercice pratique")

st.markdown("""
**🎯 À vous de jouer !**
Créez une mini-application qui combine session state, formulaires et cache.
""")

with st.expander("📝 Exercice : Gestionnaire de notes"):
    st.markdown("""
    **Mission :**
    
    1. Formulaire pour ajouter des notes (nom, matière, note)
    2. Stockage dans session state
    3. Calcul de moyenne avec cache
    4. Affichage de toutes les notes
    
    **Code de départ :**
    """)
    
    st.code("""
# 1. Initialiser les données
if 'notes' not in st.session_state:
    st.session_state.notes = []

# 2. Formulaire pour ajouter
with st.form("ajout_note"):
    st.write("📝 Ajouter une note")
    nom = st.text_input("Nom de l'étudiant:")
    matiere = st.selectbox("Matière:", ["Maths", "Français", "Sciences"])
    note = st.number_input("Note (/20):", 0, 20, 10)
    
    if st.form_submit_button("➕ Ajouter"):
        if nom:
            st.session_state.notes.append({
                'nom': nom, 'matiere': matiere, 'note': note
            })
            st.success("Note ajoutée!")

# 3. Fonction avec cache pour calculer la moyenne
@st.cache_data
def calculer_moyenne(notes_data):
    if notes_data:
        total = sum([n['note'] for n in notes_data])
        return total / len(notes_data)
    return 0

# 4. Affichage
if st.session_state.notes:
    df = pd.DataFrame(st.session_state.notes)
    st.dataframe(df)
    
    moyenne = calculer_moyenne(st.session_state.notes)
    st.metric("Moyenne générale", f"{moyenne:.1f}/20")
""")

# Zone de test
st.markdown("**💻 Exemple fonctionnel :**")

# 1. Initialiser les données
if 'demo_notes' not in st.session_state:
    st.session_state.demo_notes = []

# 2. Formulaire pour ajouter
with st.form("demo_ajout_note"):
    st.write("**📝 Ajouter une note**")
    nom = st.text_input("Nom de l'étudiant:", key="demo_note_nom")
    matiere = st.selectbox("Matière:", ["Maths", "Français", "Sciences"], key="demo_note_matiere")
    note = st.number_input("Note (/20):", 0, 20, 10, key="demo_note_valeur")
    
    if st.form_submit_button("➕ Ajouter"):
        if nom:
            st.session_state.demo_notes.append({
                'nom': nom, 'matiere': matiere, 'note': note
            })
            st.success("Note ajoutée!")

# 3. Fonction avec cache pour calculer la moyenne
@st.cache_data
def calculer_moyenne_demo(notes_data):
    if notes_data:
        total = sum([n['note'] for n in notes_data])
        return total / len(notes_data)
    return 0

# 4. Affichage
if st.session_state.demo_notes:
    st.write("**📊 Toutes les notes:**")
    df_notes = pd.DataFrame(st.session_state.demo_notes)
    st.dataframe(df_notes, use_container_width=True)
    
    moyenne = calculer_moyenne_demo(st.session_state.demo_notes)
    
    # Métriques
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Moyenne générale", f"{moyenne:.1f}/20")
    with metric_col2:
        st.metric("Nombre de notes", len(st.session_state.demo_notes))
    with metric_col3:
        meilleure_note = max([n['note'] for n in st.session_state.demo_notes])
        st.metric("Meilleure note", f"{meilleure_note}/20")
    
    # Bouton pour effacer
    if st.button("🗑️ Effacer toutes les notes", key="demo_clear_notes"):
        st.session_state.demo_notes = []
        st.rerun()

else:
    st.info("Ajoutez quelques notes avec le formulaire ci-dessus !")

st.divider()

# ================================
# 8. RÉCAPITULATIF
# ================================
st.subheader("8️⃣ Récapitulatif")

st.markdown("""
**🎓 Ce que vous avez appris :**

✅ **Session State :** `st.session_state` pour la mémoire de l'app  
✅ **Cache :** `@st.cache_data` pour accélérer les calculs  
✅ **Formulaires :** `st.form()` pour grouper les widgets  
✅ **Progress Bar :** `st.progress()` pour les tâches longues  
✅ **Spinner :** `st.spinner()` pour les chargements  
✅ **Méthodes utiles :** `st.rerun()`, `st.empty()`  

**💡 Quand utiliser quoi :**
- **Session state** : Garder des données entre les interactions
- **Cache** : Éviter de recalculer les mêmes choses
- **Formulaires** : Plusieurs champs à valider ensemble
- **Progress/Spinner** : Rassurer l'utilisateur pendant l'attente

**🚀 Prochaine étape :** T_08 - Performance (optimiser votre app)
""")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⬅️ Module précédent (T_06)", key="nav_prev_t7"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_06 - Fichiers")

with col2:
    if st.button("📚 Retour au sommaire", key="nav_home_t7"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_00 - Sommaire")

with col3:
    if st.button("➡️ Module suivant (T_08)", key="nav_next_t7"):
        st.info("👈 Utilisez la barre latérale pour naviguer vers T_08 - Performance")

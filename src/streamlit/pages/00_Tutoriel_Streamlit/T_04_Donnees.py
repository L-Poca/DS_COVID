import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import io

st.header("📊 T_04 - Gestion Avancée des Données")

st.markdown("**📋 Objectif :** Maîtriser l'affichage, la manipulation, l'édition et l'analyse de données avec Streamlit pour créer des applications data-driven professionnelles.")

st.markdown("---")

# ================================
# 1. DATAFRAMES INTERACTIFS AVANCÉS
# ================================
st.subheader("1️⃣ DataFrames interactifs - La puissance des données")

st.markdown("""
**📖 Description :**
Les DataFrames Streamlit offrent bien plus qu'un simple affichage tabulaire.
Ils proposent une interactivité native : tri par colonnes, recherche globale, 
redimensionnement, et mise en forme conditionnelle pour une exploration intuitive des données.

**🎯 Fonctionnalités natives :**
- **Tri automatique** : Clic sur en-têtes pour trier (croissant/décroissant)
- **Recherche intégrée** : Filtre global sur toutes les colonnes
- **Redimensionnement** : Ajustement dynamique des colonnes
- **Sélection** : Sélection de lignes/cellules pour actions
- **Copie** : Export vers clipboard d'un clic

**💡 Différence clé avec `st.table()` :**
- `st.dataframe()` → Interactif, scrollable, fonctionnalités avancées
- `st.table()` → Statique, rendu complet, style markdown classique
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# Génération d'un dataset riche et réaliste
np.random.seed(42)
n_rows = 50

# Données simulées d'une entreprise tech
df_company = pd.DataFrame({
    'Employé_ID': [f'EMP{i:03d}' for i in range(1, n_rows+1)],
    'Nom': [f'Employee_{i}' for i in range(1, n_rows+1)],
    'Département': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], n_rows),
    'Poste': np.random.choice(['Junior', 'Senior', 'Lead', 'Manager', 'Director'], n_rows),
    'Salaire_Annuel': np.random.randint(35000, 120000, n_rows),
    'Années_Expérience': np.random.randint(0, 15, n_rows),
    'Performance_Score': np.random.uniform(2.5, 5.0, n_rows).round(1),
    'Remote': np.random.choice([True, False], n_rows),
    'Date_Embauche': pd.date_range('2015-01-01', '2024-01-01', periods=n_rows)
})

# AFFICHAGE INTERACTIF avec personnalisation
st.markdown("**🔍 Dataset Complet (Interactif)**")
st.dataframe(
    df_company,
    use_container_width=True,
    height=400,  # Hauteur fixe avec scroll
    hide_index=True,  # Masquer l'index par défaut
    column_config={
        "Salaire_Annuel": st.column_config.NumberColumn(
            "Salaire (€)",
            help="Salaire annuel brut",
            format="€%.0f"
        ),
        "Performance_Score": st.column_config.ProgressColumn(
            "Performance",
            help="Score de performance sur 5",
            min_value=0,
            max_value=5,
            format="%.1f"
        ),
        "Remote": st.column_config.CheckboxColumn(
            "Télétravail",
            help="Autorisé au télétravail"
        ),
        "Date_Embauche": st.column_config.DateColumn(
            "Embauche",
            format="DD/MM/YYYY"
        )
    }
)

# MISE EN FORME CONDITIONNELLE avec Pandas Styling
st.markdown("**🎨 Styling Conditionnel**")

# Fonction de coloration selon la performance
def color_performance(val):
    if val >= 4.0:
        return 'background-color: #d4edda; color: #155724'  # Vert
    elif val >= 3.0:
        return 'background-color: #fff3cd; color: #856404'  # Jaune
    else:
        return 'background-color: #f8d7da; color: #721c24'  # Rouge

# DataFrame stylé avec conditions
styled_df = df_company.head(10).style.applymap(
    color_performance, subset=['Performance_Score']
).format({
    'Salaire_Annuel': '€{:,.0f}',
    'Performance_Score': '{:.1f}',
    'Date_Embauche': lambda x: x.strftime('%d/%m/%Y')
}).highlight_max(subset=['Salaire_Annuel'], color='lightblue')

st.dataframe(styled_df, use_container_width=True)
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Génération du dataset
    np.random.seed(42)
    n_rows = 50
    
    df_company = pd.DataFrame({
        'Employé_ID': [f'EMP{i:03d}' for i in range(1, n_rows+1)],
        'Nom': [f'Employee_{i}' for i in range(1, n_rows+1)],
        'Département': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], n_rows),
        'Poste': np.random.choice(['Junior', 'Senior', 'Lead', 'Manager', 'Director'], n_rows),
        'Salaire_Annuel': np.random.randint(35000, 120000, n_rows),
        'Années_Expérience': np.random.randint(0, 15, n_rows),
        'Performance_Score': np.random.uniform(2.5, 5.0, n_rows).round(1),
        'Remote': np.random.choice([True, False], n_rows),
        'Date_Embauche': pd.date_range('2015-01-01', '2024-01-01', periods=n_rows)
    })
    
    # Affichage avec configuration des colonnes
    st.markdown("**🔍 Dataset Interactif**")
    st.dataframe(
        df_company,
        use_container_width=True,
        height=300,
        hide_index=True,
        column_config={
            "Salaire_Annuel": st.column_config.NumberColumn(
                "Salaire (€)",
                format="€%.0f"
            ),
            "Performance_Score": st.column_config.ProgressColumn(
                "Performance",
                min_value=0,
                max_value=5,
                format="%.1f"
            ),
            "Remote": st.column_config.CheckboxColumn("Télétravail"),
            "Date_Embauche": st.column_config.DateColumn(
                "Embauche",
                format="DD/MM/YYYY"
            )
        }
    )
    
    # Styling conditionnel
    st.markdown("**🎨 Styling Conditionnel**")
    
    def color_performance(val):
        if val >= 4.0:
            return 'background-color: #d4edda'
        elif val >= 3.0:
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #f8d7da'
    
    styled_sample = df_company.head(5).style.applymap(
        color_performance, subset=['Performance_Score']
    ).format({
        'Salaire_Annuel': '€{:,.0f}',
        'Performance_Score': '{:.1f}',
        'Date_Embauche': lambda x: x.strftime('%d/%m/%Y')
    })
    
    st.dataframe(styled_sample, use_container_width=True)

st.markdown("""
**🔍 Configuration avancée des colonnes :**
- **NumberColumn** : Formatage monétaire, scientifique, pourcentages
- **ProgressColumn** : Barres de progression visuelles pour métriques
- **CheckboxColumn** : Cases à cocher pour booléens
- **DateColumn** : Formats de date personnalisés
- **LinkColumn** : Liens cliquables vers URLs
- **ImageColumn** : Affichage d'images dans les cellules
""")

st.divider()

# ================================
# 2. ANALYSE STATISTIQUE AVANCÉE
# ================================
st.subheader("2️⃣ Analyse statistique et exploration de données")

st.markdown("""
**📖 Description :**
L'analyse exploratoire est la première étape cruciale de tout projet data.
Streamlit facilite la création de dashboards d'analyse avec des statistiques 
descriptives, des distributions, et des insights automatisés.

**🎯 Techniques d'analyse :**
- Statistiques descriptives complètes
- Détection d'anomalies et valeurs aberrantes
- Corrélations entre variables
- Distribution et histogrammes
- Analyse de la qualité des données (valeurs manquantes)

**💡 Objectifs de l'exploration :**
- Comprendre la structure des données
- Identifier les patterns et tendances
- Détecter les problèmes de qualité
- Préparer le nettoyage et la modélisation
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# PANNEAU D'ANALYSE COMPLET
st.markdown("### 📊 Tableau de Bord Analytique")

# 1. Vue d'ensemble rapide
overview_cols = st.columns(4)
with overview_cols[0]:
    st.metric("📋 Lignes", f"{len(df_company):,}")
with overview_cols[1]:
    st.metric("📊 Colonnes", len(df_company.columns))
with overview_cols[2]:
    avg_salary = df_company['Salaire_Annuel'].mean()
    st.metric("💰 Salaire Moyen", f"€{avg_salary:,.0f}")
with overview_cols[3]:
    top_dept = df_company['Département'].mode()[0]
    st.metric("🏢 Top Département", top_dept)

# 2. ANALYSE DES VALEURS MANQUANTES
st.markdown("**❓ Analyse des Valeurs Manquantes**")
missing_data = df_company.isnull().sum()
if missing_data.sum() == 0:
    st.success("✅ Aucune valeur manquante détectée")
else:
    missing_percent = (missing_data / len(df_company) * 100).round(1)
    missing_df = pd.DataFrame({
        'Colonnes': missing_data.index,
        'Valeurs_Manquantes': missing_data.values,
        'Pourcentage': missing_percent.values
    })
    st.dataframe(missing_df[missing_df['Valeurs_Manquantes'] > 0])

# 3. STATISTIQUES PAR DÉPARTEMENT
st.markdown("**📈 Analyse par Département**")
dept_analysis = df_company.groupby('Département').agg({
    'Salaire_Annuel': ['mean', 'median', 'std'],
    'Performance_Score': 'mean',
    'Années_Expérience': 'mean',
    'Remote': lambda x: (x == True).sum()
}).round(2)

# Aplatir les colonnes multi-niveaux
dept_analysis.columns = ['Salaire_Moyen', 'Salaire_Médian', 'Salaire_StdDev', 
                        'Performance_Moy', 'Expérience_Moy', 'Remote_Count']
st.dataframe(dept_analysis, use_container_width=True)

# 4. DÉTECTION D'ANOMALIES (Outliers)
st.markdown("**🚨 Détection d'Anomalies - Salaires**")

# Calcul des outliers avec méthode IQR
Q1 = df_company['Salaire_Annuel'].quantile(0.25)
Q3 = df_company['Salaire_Annuel'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_company[
    (df_company['Salaire_Annuel'] < lower_bound) | 
    (df_company['Salaire_Annuel'] > upper_bound)
]

if len(outliers) > 0:
    st.warning(f"⚠️ {len(outliers)} anomalie(s) détectée(s)")
    st.dataframe(outliers[['Nom', 'Département', 'Poste', 'Salaire_Annuel']])
else:
    st.success("✅ Aucune anomalie détectée dans les salaires")

# 5. MATRICE DE CORRÉLATION
st.markdown("**🔗 Corrélations entre Variables Numériques**")
numeric_columns = ['Salaire_Annuel', 'Années_Expérience', 'Performance_Score']
correlation_matrix = df_company[numeric_columns].corr().round(3)

# Affichage avec mise en forme
st.dataframe(correlation_matrix.style.background_gradient(cmap='coolwarm', center=0))

# Insights automatiques sur les corrélations
max_corr = correlation_matrix.abs().unstack().sort_values(ascending=False)
# Exclure les corrélations parfaites (diagonale)
max_corr = max_corr[max_corr < 1.0].head(1)
if len(max_corr) > 0:
    var1, var2 = max_corr.index[0]
    corr_value = correlation_matrix.loc[var1, var2]
    if abs(corr_value) > 0.5:
        st.info(f"💡 Corrélation forte détectée : {var1} ↔ {var2} (r={corr_value:.3f})")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Panneau d'analyse
    st.markdown("### 📊 Dashboard Analytique")
    
    # Vue d'ensemble
    overview_cols = st.columns(4)
    with overview_cols[0]:
        st.metric("📋 Lignes", f"{len(df_company):,}")
    with overview_cols[1]:
        st.metric("📊 Colonnes", len(df_company.columns))
    with overview_cols[2]:
        avg_salary = df_company['Salaire_Annuel'].mean()
        st.metric("💰 Moy. Salaire", f"€{avg_salary:,.0f}")
    with overview_cols[3]:
        top_dept = df_company['Département'].mode()[0]
        st.metric("🏢 Top Dept.", top_dept)
    
    # Valeurs manquantes
    st.markdown("**❓ Valeurs Manquantes**")
    missing_data = df_company.isnull().sum()
    if missing_data.sum() == 0:
        st.success("✅ Aucune valeur manquante")
    
    # Analyse par département
    st.markdown("**📈 Par Département**")
    dept_analysis = df_company.groupby('Département').agg({
        'Salaire_Annuel': 'mean',
        'Performance_Score': 'mean',
        'Remote': lambda x: (x == True).sum()
    }).round(1)
    
    dept_analysis.columns = ['Salaire_Moy', 'Perf_Moy', 'Remote_Count']
    st.dataframe(dept_analysis, use_container_width=True)
    
    # Détection anomalies
    st.markdown("**🚨 Anomalies Salariales**")
    Q1 = df_company['Salaire_Annuel'].quantile(0.25)
    Q3 = df_company['Salaire_Annuel'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_company[
        (df_company['Salaire_Annuel'] < lower_bound) | 
        (df_company['Salaire_Annuel'] > upper_bound)
    ]
    
    if len(outliers) > 0:
        st.warning(f"⚠️ {len(outliers)} anomalie(s)")
        st.dataframe(outliers[['Nom', 'Salaire_Annuel']].head(3))
    else:
        st.success("✅ Aucune anomalie")
    
    # Corrélations
    st.markdown("**🔗 Corrélations**")
    numeric_columns = ['Salaire_Annuel', 'Années_Expérience', 'Performance_Score']
    correlation_matrix = df_company[numeric_columns].corr().round(3)
    
    st.dataframe(correlation_matrix.style.background_gradient(cmap='coolwarm'))

st.markdown("""
**🔍 Techniques d'analyse automatisée :**
- **IQR (Interquartile Range)** : Détection robuste d'outliers
- **Matrice de corrélation** : Relations linéaires entre variables
- **Groupby & Aggregation** : Statistiques par catégories
- **Missing data analysis** : Diagnostic de qualité des données
- **Automated insights** : Extraction automatique de patterns intéressants
""")

st.divider()

# ================================
# 3. FILTRAGE INTERACTIF AVANCÉ
# ================================
st.subheader("3️⃣ Filtrage interactif et exploration dynamique")

st.markdown("""
**📖 Description :**
Le filtrage interactif transforme vos données en outils d'exploration puissants.
Les utilisateurs peuvent segmenter, analyser et découvrir des insights en temps réel
sans connaissances techniques préalables.

**🎯 Types de filtres avancés :**
- Filtres multiples combinés (ET/OU)
- Plages de dates et valeurs numériques
- Recherche textuelle fuzzy
- Filtres hiérarchiques (département → poste)
- Filtres conditionnels dynamiques

**💡 Objectifs UX :**
- Interface intuitive pour utilisateurs non-techniques
- Feedback immédiat sur le nombre de résultats
- Sauvegarde et partage de configurations de filtres
- Analyse comparative entre segments
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# PANNEAU DE FILTRAGE AVANCÉ
st.markdown("### 🔍 Centre de Filtrage Avancé")

# Organisation en colonnes pour une interface compacte
filter_cols = st.columns(3)

# 1. FILTRES CATÉGORIELS avec sélection multiple
with filter_cols[0]:
    st.markdown("**🏢 Départements**")
    selected_depts = st.multiselect(
        "Sélectionner département(s):",
        options=df_company['Département'].unique(),
        default=df_company['Département'].unique(),  # Tous sélectionnés
        key="dept_filter"
    )

with filter_cols[1]:
    st.markdown("**👔 Postes**")
    # Filtrage hiérarchique : postes selon départements sélectionnés
    available_positions = df_company[
        df_company['Département'].isin(selected_depts)
    ]['Poste'].unique()
    
    selected_positions = st.multiselect(
        "Sélectionner poste(s):",
        options=available_positions,
        default=available_positions,
        key="position_filter"
    )

with filter_cols[2]:
    st.markdown("**💼 Télétravail**")
    remote_filter = st.radio(
        "Mode de travail:",
        options=["Tous", "Télétravail uniquement", "Bureau uniquement"],
        key="remote_filter"
    )

# 2. FILTRES NUMÉRIQUES avec plages
st.markdown("**📊 Filtres Numériques**")
numeric_filter_cols = st.columns(3)

with numeric_filter_cols[0]:
    salary_range = st.slider(
        "💰 Plage de salaire (€):",
        min_value=int(df_company['Salaire_Annuel'].min()),
        max_value=int(df_company['Salaire_Annuel'].max()),
        value=(
            int(df_company['Salaire_Annuel'].min()),
            int(df_company['Salaire_Annuel'].max())
        ),
        step=1000,
        format="€%d"
    )

with numeric_filter_cols[1]:
    experience_range = st.slider(
        "🎯 Années d'expérience:",
        min_value=int(df_company['Années_Expérience'].min()),
        max_value=int(df_company['Années_Expérience'].max()),
        value=(
            int(df_company['Années_Expérience'].min()),
            int(df_company['Années_Expérience'].max())
        )
    )

with numeric_filter_cols[2]:
    performance_min = st.number_input(
        "⭐ Performance minimale:",
        min_value=float(df_company['Performance_Score'].min()),
        max_value=float(df_company['Performance_Score'].max()),
        value=float(df_company['Performance_Score'].min()),
        step=0.1,
        format="%.1f"
    )

# 3. FILTRE TEMPOREL
st.markdown("**📅 Filtre Temporel**")
date_filter_cols = st.columns(2)

with date_filter_cols[0]:
    start_date = st.date_input(
        "Date d'embauche - Début:",
        value=df_company['Date_Embauche'].min(),
        min_value=df_company['Date_Embauche'].min(),
        max_value=df_company['Date_Embauche'].max()
    )

with date_filter_cols[1]:
    end_date = st.date_input(
        "Date d'embauche - Fin:",
        value=df_company['Date_Embauche'].max(),
        min_value=df_company['Date_Embauche'].min(),
        max_value=df_company['Date_Embauche'].max()
    )

# 4. RECHERCHE TEXTUELLE
search_term = st.text_input(
    "🔍 Recherche dans les noms:",
    placeholder="Tapez un nom d'employé...",
    help="Recherche insensible à la casse"
)

# 5. APPLICATION DES FILTRES (cumulative)
filtered_df = df_company.copy()

# Appliquer tous les filtres
filtered_df = filtered_df[filtered_df['Département'].isin(selected_depts)]
filtered_df = filtered_df[filtered_df['Poste'].isin(selected_positions)]

# Filtre télétravail
if remote_filter == "Télétravail uniquement":
    filtered_df = filtered_df[filtered_df['Remote'] == True]
elif remote_filter == "Bureau uniquement":
    filtered_df = filtered_df[filtered_df['Remote'] == False]

# Filtres numériques
filtered_df = filtered_df[
    (filtered_df['Salaire_Annuel'] >= salary_range[0]) &
    (filtered_df['Salaire_Annuel'] <= salary_range[1])
]
filtered_df = filtered_df[
    (filtered_df['Années_Expérience'] >= experience_range[0]) &
    (filtered_df['Années_Expérience'] <= experience_range[1])
]
filtered_df = filtered_df[filtered_df['Performance_Score'] >= performance_min]

# Filtre temporel
filtered_df = filtered_df[
    (filtered_df['Date_Embauche'] >= pd.Timestamp(start_date)) &
    (filtered_df['Date_Embauche'] <= pd.Timestamp(end_date))
]

# Recherche textuelle (insensible à la casse)
if search_term:
    filtered_df = filtered_df[
        filtered_df['Nom'].str.contains(search_term, case=False, na=False)
    ]

# 6. AFFICHAGE DES RÉSULTATS avec métriques
st.markdown("### 📊 Résultats du Filtrage")

# Métriques de filtrage
result_cols = st.columns(4)
with result_cols[0]:
    st.metric("👥 Employés trouvés", len(filtered_df))
with result_cols[1]:
    if len(filtered_df) > 0:
        avg_filtered_salary = filtered_df['Salaire_Annuel'].mean()
        st.metric("💰 Salaire moyen", f"€{avg_filtered_salary:,.0f}")
    else:
        st.metric("💰 Salaire moyen", "N/A")
with result_cols[2]:
    if len(filtered_df) > 0:
        avg_performance = filtered_df['Performance_Score'].mean()
        st.metric("⭐ Performance moy.", f"{avg_performance:.1f}")
    else:
        st.metric("⭐ Performance moy.", "N/A")
with result_cols[3]:
    percentage_found = (len(filtered_df) / len(df_company)) * 100
    st.metric("📊 % du dataset", f"{percentage_found:.1f}%")

# Affichage conditionnel des résultats
if len(filtered_df) > 0:
    st.dataframe(filtered_df, use_container_width=True, height=400)
    
    # Option d'export des résultats filtrés
    csv_filtered = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Exporter résultats (CSV)",
        data=csv_filtered,
        file_name=f'employees_filtered_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
        mime='text/csv'
    )
else:
    st.warning("⚠️ Aucun employé ne correspond aux critères sélectionnés.")
    st.info("💡 Essayez d'élargir vos critères de filtrage.")
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Panneau de filtrage compact
    st.markdown("### 🔍 Filtrage Avancé")
    
    # Filtres catégoriels
    filter_cols = st.columns(2)
    with filter_cols[0]:
        selected_depts = st.multiselect(
            "🏢 Départements:",
            options=df_company['Département'].unique(),
            default=['Engineering', 'Sales'],
            key="demo_dept_filter"
        )
    
    with filter_cols[1]:
        remote_filter = st.radio(
            "💼 Mode travail:",
            options=["Tous", "Télétravail", "Bureau"],
            key="demo_remote_filter"
        )
    
    # Filtres numériques
    salary_range = st.slider(
        "💰 Salaire (€):",
        min_value=35000,
        max_value=120000,
        value=(40000, 100000),
        step=5000,
        key="demo_salary"
    )
    
    performance_min = st.slider(
        "⭐ Performance min:",
        min_value=2.5,
        max_value=5.0,
        value=3.0,
        step=0.1,
        key="demo_performance"
    )
    
    # Application des filtres
    demo_filtered = df_company.copy()
    demo_filtered = demo_filtered[demo_filtered['Département'].isin(selected_depts)]
    
    if remote_filter == "Télétravail":
        demo_filtered = demo_filtered[demo_filtered['Remote'] == True]
    elif remote_filter == "Bureau":
        demo_filtered = demo_filtered[demo_filtered['Remote'] == False]
    
    demo_filtered = demo_filtered[
        (demo_filtered['Salaire_Annuel'] >= salary_range[0]) &
        (demo_filtered['Salaire_Annuel'] <= salary_range[1])
    ]
    demo_filtered = demo_filtered[demo_filtered['Performance_Score'] >= performance_min]
    
    # Résultats
    st.markdown("### 📊 Résultats")
    
    result_cols = st.columns(3)
    with result_cols[0]:
        st.metric("👥 Trouvés", len(demo_filtered))
    with result_cols[1]:
        if len(demo_filtered) > 0:
            avg_sal = demo_filtered['Salaire_Annuel'].mean()
            st.metric("💰 Moy.", f"€{avg_sal:,.0f}")
        else:
            st.metric("💰 Moy.", "N/A")
    with result_cols[2]:
        pct = (len(demo_filtered) / len(df_company)) * 100
        st.metric("📊 %", f"{pct:.1f}%")
    
    if len(demo_filtered) > 0:
        st.dataframe(demo_filtered.head(5), use_container_width=True)
    else:
        st.warning("⚠️ Aucun résultat")

st.markdown("""
**🔍 Techniques de filtrage avancées :**
- **Filtres combinés** : Intersection de plusieurs critères (AND logique)
- **Filtres hiérarchiques** : Dépendances entre filtres (département → poste)
- **Plages de valeurs** : Sliders pour valeurs numériques continues
- **Recherche textuelle** : Pattern matching insensible à la casse
- **Feedback immédiat** : Nombre de résultats et métriques en temps réel
- **Export conditionnel** : Sauvegarde des résultats filtrés
""")

st.divider()

# ================================
# 4. ÉDITION DE DONNÉES COLLABORATIVE
# ================================
st.subheader("4️⃣ Édition collaborative et gestion des données")

st.markdown("""
**📖 Description :**
`st.data_editor()` révolutionne la gestion des données en permettant l'édition 
collaborative en temps réel. Idéal pour les workflows collaboratifs, 
la validation de données, et la création d'interfaces de type "spreadsheet".

**🎯 Fonctionnalités d'édition :**
- Édition cellule par cellule en temps réel
- Ajout/suppression de lignes dynamique
- Validation de types et contraintes
- Historique des modifications
- Export des données modifiées

**💡 Cas d'usage professionnels :**
- Budgets collaboratifs et plannings
- Validation de datasets par les experts métier
- Configuration de paramètres d'applications
- Saisie de données terrain par équipes
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# SYSTÈME D'ÉDITION COLLABORATIVE
st.markdown("### ✏️ Éditeur de Données Collaboratif")

# 1. DONNÉES INITIALES avec différents types
if 'editable_products' not in st.session_state:
    st.session_state.editable_products = pd.DataFrame({
        'Produit': ['Laptop Pro', 'Smartphone X', 'Tablette Air'],
        'Catégorie': ['Informatique', 'Mobile', 'Informatique'],
        'Prix_Unitaire': [1299.99, 899.99, 599.99],
        'Stock_Actuel': [45, 120, 78],
        'Stock_Minimum': [10, 25, 15],
        'Actif': [True, True, False],
        'Date_Lancement': [
            datetime(2024, 1, 15),
            datetime(2024, 3, 10),
            datetime(2023, 11, 20)
        ],
        'Fournisseur': ['TechCorp', 'MobilePlus', 'TechCorp']
    })

# 2. CONFIGURATION AVANCÉE DE L'ÉDITEUR
st.markdown("**🛠️ Configuration de l'Éditeur**")
config_cols = st.columns(3)

with config_cols[0]:
    allow_add_rows = st.checkbox("➕ Ajouter lignes", value=True)
    
with config_cols[1]:
    allow_delete_rows = st.checkbox("🗑️ Supprimer lignes", value=True)
    
with config_cols[2]:
    show_toolbar = st.checkbox("🔧 Barre d'outils", value=True)

# 3. ÉDITEUR AVEC CONFIGURATION PERSONNALISÉE
num_rows_mode = "dynamic" if (allow_add_rows or allow_delete_rows) else "fixed"

edited_products = st.data_editor(
    st.session_state.editable_products,
    num_rows=num_rows_mode,
    use_container_width=True,
    height=400,
    hide_index=True,
    column_config={
        "Produit": st.column_config.TextColumn(
            "Nom du Produit",
            help="Nom commercial du produit",
            max_chars=50,
            validate="^[A-Za-z0-9 ]+$"  # Validation regex
        ),
        "Catégorie": st.column_config.SelectboxColumn(
            "Catégorie",
            help="Catégorie du produit",
            options=["Informatique", "Mobile", "Audio", "Gaming"],
            required=True
        ),
        "Prix_Unitaire": st.column_config.NumberColumn(
            "Prix (€)",
            help="Prix unitaire en euros",
            min_value=0.0,
            max_value=10000.0,
            step=0.01,
            format="€%.2f"
        ),
        "Stock_Actuel": st.column_config.NumberColumn(
            "Stock",
            help="Quantité en stock",
            min_value=0,
            step=1,
            format="%d unités"
        ),
        "Stock_Minimum": st.column_config.NumberColumn(
            "Min. Stock",
            help="Seuil d'alerte stock",
            min_value=0,
            step=1
        ),
        "Actif": st.column_config.CheckboxColumn(
            "Actif",
            help="Produit disponible à la vente"
        ),
        "Date_Lancement": st.column_config.DateColumn(
            "Lancement",
            help="Date de mise sur le marché",
            format="DD/MM/YYYY"
        ),
        "Fournisseur": st.column_config.SelectboxColumn(
            "Fournisseur",
            options=["TechCorp", "MobilePlus", "AudioMaster", "GameWorld"],
            required=True
        )
    },
    key="product_editor"
)

# 4. DÉTECTION ET AFFICHAGE DES MODIFICATIONS
st.markdown("### 📊 Analyse des Modifications")

if not edited_products.equals(st.session_state.editable_products):
    st.success("✅ Modifications détectées!")
    
    # Mise à jour du state pour persistance
    st.session_state.editable_products = edited_products.copy()
    
    # Analyse des changements
    changes_cols = st.columns(3)
    
    with changes_cols[0]:
        total_products = len(edited_products)
        st.metric("📦 Total Produits", total_products)
    
    with changes_cols[1]:
        active_products = len(edited_products[edited_products['Actif'] == True])
        st.metric("✅ Produits Actifs", active_products)
    
    with changes_cols[2]:
        total_value = (edited_products['Prix_Unitaire'] * edited_products['Stock_Actuel']).sum()
        st.metric("💰 Valeur Stock", f"€{total_value:,.2f}")
    
    # Alertes automatiques
    st.markdown("**🚨 Alertes Automatiques**")
    
    # Stock faible
    low_stock = edited_products[
        edited_products['Stock_Actuel'] <= edited_products['Stock_Minimum']
    ]
    if len(low_stock) > 0:
        st.warning(f"⚠️ {len(low_stock)} produit(s) en stock faible:")
        st.dataframe(low_stock[['Produit', 'Stock_Actuel', 'Stock_Minimum']], hide_index=True)
    
    # Produits inactifs avec stock
    inactive_with_stock = edited_products[
        (edited_products['Actif'] == False) & (edited_products['Stock_Actuel'] > 0)
    ]
    if len(inactive_with_stock) > 0:
        st.info(f"💡 {len(inactive_with_stock)} produit(s) inactif(s) avec stock:")
        st.dataframe(inactive_with_stock[['Produit', 'Stock_Actuel']], hide_index=True)

else:
    st.info("ℹ️ Aucune modification détectée. Commencez à éditer le tableau ci-dessus.")

# 5. ACTIONS SUR LES DONNÉES ÉDITÉES
st.markdown("### 💾 Actions sur les Données")

action_cols = st.columns(4)

with action_cols[0]:
    if st.button("📥 Exporter CSV", use_container_width=True):
        csv_data = edited_products.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📁 Télécharger",
            data=csv_data,
            file_name=f'produits_edited_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
            mime='text/csv'
        )

with action_cols[1]:
    if st.button("📄 Exporter JSON", use_container_width=True):
        json_data = edited_products.to_json(orient='records', date_format='iso').encode('utf-8')
        st.download_button(
            label="📁 Télécharger",
            data=json_data,
            file_name=f'produits_edited_{datetime.now().strftime("%Y%m%d_%H%M")}.json',
            mime='application/json'
        )

with action_cols[2]:
    if st.button("🔄 Réinitialiser", use_container_width=True):
        # Reset aux données originales
        st.session_state.editable_products = pd.DataFrame({
            'Produit': ['Laptop Pro', 'Smartphone X', 'Tablette Air'],
            'Catégorie': ['Informatique', 'Mobile', 'Informatique'],
            'Prix_Unitaire': [1299.99, 899.99, 599.99],
            'Stock_Actuel': [45, 120, 78],
            'Stock_Minimum': [10, 25, 15],
            'Actif': [True, True, False],
            'Date_Lancement': [
                datetime(2024, 1, 15),
                datetime(2024, 3, 10),
                datetime(2023, 11, 20)
            ],
            'Fournisseur': ['TechCorp', 'MobilePlus', 'TechCorp']
        })
        st.success("🔄 Données réinitialisées!")
        st.rerun()

with action_cols[3]:
    if st.button("📊 Rapport", use_container_width=True):
        # Génération d'un rapport automatique
        report = f'''
        📊 RAPPORT PRODUITS - {datetime.now().strftime("%d/%m/%Y %H:%M")}
        
        📦 Total produits: {len(edited_products)}
        ✅ Produits actifs: {len(edited_products[edited_products['Actif'] == True])}
        💰 Valeur totale du stock: €{(edited_products['Prix_Unitaire'] * edited_products['Stock_Actuel']).sum():,.2f}
        
        📈 Prix moyen: €{edited_products['Prix_Unitaire'].mean():.2f}
        📊 Stock moyen: {edited_products['Stock_Actuel'].mean():.1f} unités
        
        🏷️ Répartition par catégorie:
        {edited_products['Catégorie'].value_counts().to_string()}
        '''
        
        st.text_area("📄 Rapport Automatique", report, height=200)
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Version simplifiée pour la démo
    st.markdown("### ✏️ Éditeur Collaboratif")
    
    # Données de démo
    if 'demo_products' not in st.session_state:
        st.session_state.demo_products = pd.DataFrame({
            'Produit': ['Laptop Pro', 'Smartphone X'],
            'Prix': [1299.99, 899.99],
            'Stock': [45, 120],
            'Actif': [True, True],
        })
    
    # Éditeur simplifié
    demo_edited = st.data_editor(
        st.session_state.demo_products,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Prix": st.column_config.NumberColumn(
                "Prix (€)",
                min_value=0.0,
                format="€%.2f"
            ),
            "Stock": st.column_config.NumberColumn(
                "Stock",
                min_value=0,
                step=1
            ),
            "Actif": st.column_config.CheckboxColumn("Actif")
        },
        key="demo_editor"
    )
    
    # Détection des changements
    if not demo_edited.equals(st.session_state.demo_products):
        st.success("✅ Modifications détectées!")
        st.session_state.demo_products = demo_edited.copy()
        
        # Métriques simples
        metrics_cols = st.columns(2)
        with metrics_cols[0]:
            total_value = (demo_edited['Prix'] * demo_edited['Stock']).sum()
            st.metric("💰 Valeur", f"€{total_value:,.0f}")
        with metrics_cols[1]:
            active_count = len(demo_edited[demo_edited['Actif'] == True])
            st.metric("✅ Actifs", active_count)
    
    # Export simplifié
    if st.button("📥 Exporter", key="demo_export"):
        csv_demo = demo_edited.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📁 CSV",
            csv_demo,
            f'demo_products_{datetime.now().strftime("%H%M")}.csv',
            'text/csv',
            key="demo_download"
        )

st.markdown("""
**🔍 Fonctionnalités avancées de l'éditeur :**
- **Validation en temps réel** : Contraintes de types, plages, regex
- **Colonnes configurables** : Types spécialisés (dates, sélecteurs, checkboxes)
- **Gestion d'état** : Persistance des modifications via `session_state`
- **Alertes automatiques** : Règles métier et notifications
- **Export multi-format** : CSV, JSON, Excel selon les besoins
- **Rapports automatisés** : Génération de synthèses sur les données modifiées
""")

st.divider()

# ================================
# 5. JSON ET DONNÉES STRUCTURÉES
# ================================
st.subheader("5️⃣ JSON et données structurées avancées")

st.markdown("""
**📖 Description :**
Le format JSON est essentiel pour les APIs, configurations, et données hiérarchiques.
Streamlit excelle dans l'affichage et la manipulation de structures complexes
avec une interface claire et navigable.

**🎯 Types de données JSON :**
- Configurations d'applications
- Réponses d'APIs REST
- Métadonnées et schemas
- Logs structurés d'applications
- Paramètres de modèles ML

**💡 Avantages du JSON dans Streamlit :**
- Affichage hiérarchique interactif
- Syntaxe colorée et formatage automatique
- Navigation par collapse/expand
- Édition directe possible avec widgets
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code("""
# 1. CONFIGURATION D'APPLICATION COMPLEXE
app_config = {
    "application": {
        "name": "DataDashboard Pro",
        "version": "2.1.3",
        "environment": "production",
        "features": {
            "authentication": {
                "enabled": True,
                "providers": ["google", "microsoft", "local"],
                "session_timeout": 3600,
                "multi_factor": True
            },
            "analytics": {
                "enabled": True,
                "tracking_id": "GA-123456789",
                "events": ["page_view", "click", "download"],
                "retention_days": 90
            },
            "performance": {
                "cache_enabled": True,
                "cache_ttl": 600,
                "max_concurrent_users": 1000,
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 60
                }
            }
        },
        "database": {
            "type": "postgresql",
            "host": "db.company.com",
            "port": 5432,
            "ssl_enabled": True,
            "connection_pool": {
                "min_connections": 5,
                "max_connections": 20,
                "timeout": 30
            }
        },
        "notifications": {
            "email": {
                "smtp_host": "smtp.company.com",
                "port": 587,
                "templates": {
                    "welcome": "welcome_template.html",
                    "alert": "alert_template.html",
                    "report": "report_template.html"
                }
            },
            "slack": {
                "webhook_url": "https://hooks.slack.com/...",
                "channels": ["#alerts", "#reports", "#general"]
            }
        }
    },
    "last_updated": "2024-07-19T10:30:00Z",
    "updated_by": "admin@company.com"
}

st.markdown("**⚙️ Configuration d'Application**")
st.json(app_config)

# 2. SIMULATION DE RÉPONSE API avec métadonnées
api_response = {
    "status": "success",
    "timestamp": datetime.now().isoformat(),
    "request_id": "req_789abc123def",
    "data": {
        "users": [
            {
                "id": 1,
                "username": "alice_data",
                "email": "alice@company.com",
                "profile": {
                    "first_name": "Alice",
                    "last_name": "Anderson",
                    "department": "Data Science",
                    "skills": ["Python", "SQL", "Machine Learning"],
                    "certifications": [
                        {"name": "AWS Data Engineer", "date": "2024-03-15"},
                        {"name": "Google Analytics", "date": "2023-11-20"}
                    ]
                },
                "permissions": {
                    "read": ["all_data", "reports", "dashboards"],
                    "write": ["personal_data", "draft_reports"],
                    "admin": []
                }
            },
            {
                "id": 2,
                "username": "bob_analyst",
                "email": "bob@company.com",
                "profile": {
                    "first_name": "Bob",
                    "last_name": "Builder",
                    "department": "Business Intelligence",
                    "skills": ["Tableau", "Power BI", "Excel"],
                    "certifications": [
                        {"name": "Tableau Desktop Specialist", "date": "2024-01-10"}
                    ]
                },
                "permissions": {
                    "read": ["reports", "dashboards"],
                    "write": ["draft_reports"],
                    "admin": []
                }
            }
        ],
        "pagination": {
            "current_page": 1,
            "total_pages": 1,
            "total_items": 2,
            "items_per_page": 10
        }
    },
    "metadata": {
        "api_version": "v2.1",
        "response_time_ms": 87,
        "server": "api-server-03",
        "cache_hit": False
    }
}

st.markdown("**🌐 Réponse API Simulée**")
st.json(api_response)

# 3. LOGS STRUCTURÉS d'application
log_entries = [
    {
        "timestamp": "2024-07-19T10:25:33.123Z",
        "level": "INFO",
        "service": "auth-service",
        "message": "User login successful",
        "user_id": "alice_data",
        "session_id": "sess_abc123",
        "ip_address": "192.168.1.100",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "metadata": {
            "login_method": "google_oauth",
            "location": {"country": "France", "city": "Paris"},
            "device_type": "desktop"
        }
    },
    {
        "timestamp": "2024-07-19T10:26:15.456Z",
        "level": "WARNING",
        "service": "data-processor",
        "message": "High memory usage detected",
        "metrics": {
            "memory_usage_percent": 87.5,
            "cpu_usage_percent": 45.2,
            "disk_usage_gb": 23.7
        },
        "thresholds": {
            "memory_warning": 80,
            "memory_critical": 95
        },
        "action_taken": "scaling_initiated"
    },
    {
        "timestamp": "2024-07-19T10:27:02.789Z",
        "level": "ERROR",
        "service": "report-generator",
        "message": "Failed to generate monthly report",
        "error": {
            "type": "DatabaseConnectionError",
            "code": "DB_TIMEOUT",
            "details": "Connection timeout after 30 seconds",
            "stack_trace": [
                "at ReportService.generateReport()",
                "at DatabaseConnector.query()",
                "at ConnectionPool.getConnection()"
            ]
        },
        "retry_info": {
            "attempt": 3,
            "max_attempts": 5,
            "next_retry_in_seconds": 60
        }
    }
]

st.markdown("**📋 Logs d'Application Structurés**")
for i, log_entry in enumerate(log_entries):
    with st.expander(f"Log {i+1}: {log_entry['level']} - {log_entry['message'][:50]}..."):
        st.json(log_entry)

# 4. ÉDITION JSON INTERACTIVE
st.markdown("**✏️ Éditeur JSON Interactif**")

# Interface pour éditer la configuration
st.markdown("Modifiez la configuration ci-dessous :")

# Extraction de paramètres éditables
config_edit_cols = st.columns(2)

with config_edit_cols[0]:
    new_timeout = st.number_input(
        "Session Timeout (secondes):",
        value=app_config["application"]["features"]["authentication"]["session_timeout"],
        min_value=300,
        max_value=7200,
        step=300
    )
    
    new_cache_ttl = st.number_input(
        "Cache TTL (secondes):",
        value=app_config["application"]["features"]["performance"]["cache_ttl"],
        min_value=60,
        max_value=3600,
        step=60
    )

with config_edit_cols[1]:
    new_max_users = st.number_input(
        "Max Utilisateurs Concurrents:",
        value=app_config["application"]["features"]["performance"]["max_concurrent_users"],
        min_value=100,
        max_value=5000,
        step=100
    )
    
    new_rate_limit = st.number_input(
        "Limite Requêtes/Min:",
        value=app_config["application"]["features"]["performance"]["rate_limiting"]["requests_per_minute"],
        min_value=10,
        max_value=200,
        step=10
    )

# Mise à jour de la configuration
if st.button("🔄 Mettre à jour la configuration"):
    # Mise à jour des valeurs
    app_config["application"]["features"]["authentication"]["session_timeout"] = new_timeout
    app_config["application"]["features"]["performance"]["cache_ttl"] = new_cache_ttl
    app_config["application"]["features"]["performance"]["max_concurrent_users"] = new_max_users
    app_config["application"]["features"]["performance"]["rate_limiting"]["requests_per_minute"] = new_rate_limit
    app_config["last_updated"] = datetime.now().isoformat()
    
    st.success("✅ Configuration mise à jour!")
    st.json(app_config)
    
    # Export de la nouvelle configuration
    config_json = json.dumps(app_config, indent=2).encode('utf-8')
    st.download_button(
        "📥 Exporter configuration mise à jour",
        config_json,
        f'app_config_updated_{datetime.now().strftime("%Y%m%d_%H%M")}.json',
        'application/json'
    )
""", language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Version simplifiée pour la démo
    st.markdown("**⚙️ Configuration App**")
    
    demo_config = {
        "app": {
            "name": "DataDashboard",
            "version": "2.1.3",
            "features": {
                "auth": {
                    "enabled": True,
                    "timeout": 3600
                },
                "cache": {
                    "enabled": True,
                    "ttl": 600
                }
            }
        },
        "database": {
            "type": "postgresql",
            "host": "db.company.com",
            "ssl": True
        }
    }
    
    st.json(demo_config)
    
    # Réponse API simulée
    st.markdown("**🌐 API Response**")
    
    demo_api = {
        "status": "success",
        "data": {
            "users": [
                {
                    "id": 1,
                    "name": "Alice",
                    "department": "Data Science",
                    "skills": ["Python", "SQL"]
                }
            ]
        },
        "metadata": {
            "response_time_ms": 87,
            "api_version": "v2.1"
        }
    }
    
    st.json(demo_api)
    
    # Log simple
    st.markdown("**📋 Application Log**")
    
    demo_log = {
        "timestamp": "2024-07-19T10:25:33Z",
        "level": "INFO",
        "service": "auth-service",
        "message": "User login successful",
        "user_id": "alice_data",
        "metadata": {
            "login_method": "google_oauth",
            "location": "Paris"
        }
    }
    
    with st.expander("Voir détails du log"):
        st.json(demo_log)
    
    # Éditeur simple
    st.markdown("**✏️ Éditeur JSON**")
    
    demo_timeout = st.slider(
        "Session Timeout:",
        300, 7200, 3600, 300,
        key="demo_timeout"
    )
    
    if st.button("🔄 Mettre à jour", key="demo_update"):
        demo_config["app"]["features"]["auth"]["timeout"] = demo_timeout
        st.success("✅ Configuré!")
        st.json(demo_config)

st.markdown("""
**🔍 Avantages du JSON structuré :**
- **Lisibilité hiérarchique** : Navigation claire dans les structures complexes
- **Syntaxe colorée** : Identification rapide des types de données
- **Interactivité** : Expand/collapse pour explorer les niveaux
- **Édition guidée** : Widgets spécialisés pour modifier les valeurs
- **Validation** : Contrôle de la cohérence des données structurées
- **Export/Import** : Sauvegarde et partage faciles des configurations
""")

st.markdown("---")

st.success("🎉 **Félicitations !** Vous maîtrisez maintenant la gestion avancée des données avec Streamlit !")

st.markdown("""
**📚 Récapitulatif des techniques de gestion de données :**

📊 **DataFrames Interactifs** → Exploration intuitive avec tri, recherche et styling  
📈 **Analyse Statistique** → Insights automatisés et détection d'anomalies  
🔍 **Filtrage Avancé** → Segmentation dynamique et exploration guidée  
✏️ **Édition Collaborative** → Modification temps réel et validation métier  
🗂️ **JSON Structuré** → Gestion de configurations et données hiérarchiques  

**🚀 Prochaine étape :** Explorez le module T_05 sur la mise en page et l'organisation !
""")

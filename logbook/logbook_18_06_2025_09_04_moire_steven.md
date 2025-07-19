## ✅ Objectif

Mise en place d’une démonstration interactive avec Streamlit pour valoriser les données (cartographie, visualisations dynamiques, API publiques, etc.).

---

## 🛠️ Étapes réalisées

### 🔹 1. Vérification de l’environnement Python
- Création d’un notebook de contrôle `check_python_version.ipynb`.
- Confirmation de la version de Python utilisée dans l’environnement virtuel.

### 🔹 2. Adaptation de l’environnement
- Suite au changement de version de Python (vers 3.11), régénération du fichier `requirements.txt`.
- Mise à jour des dépendances, notamment ajout explicite de :
  - `streamlit`
- Création ou mise à jour de l’environnement virtuel avec les dépendances compatibles.

### 🔹 3. Implémentation de la démo Streamlit
- Création d’une nouvelle page dans le dossier `streamlit/pages/` :
  - Intégration d’un appel API à [public.opendatasoft.com](https://public.opendatasoft.com/explore/dataset/donnees-synop-essentielles-omm/).
  - Traitement des données météo (température, géolocalisation, timestamp).
  - Affichage d’un `pydeck_chart` dynamique en fonction de la date sélectionnée.
  - Ajout d’un slider de date, d’un DataFrame affichant les valeurs, et d’un système de fallback (`st.error`) en cas d’absence de données.

---

## 🎯 Résultat

- Application Streamlit opérationnelle, interactive et visuellement engageante.
- Affichage cartographique des relevés météo via `pydeck`, basé sur une source ouverte actualisée.
- Potentiel de réutilisation pour d'autres cas d'usage (cartes COVID, pollution, inondations, etc.).
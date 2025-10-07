## ✅ Objectifs atteints
- Mise en place d’un **script automatisé** (interface Tkinter) pour analyser les dossiers d’images et générer un **template de rapport exploratoire**.
- Ajout de l’export CSV dans `reports/explorationdata/` avec encodage UTF-8.
- **Log automatique** des anomalies (dimensions non conformes, metadata manquants…).
- **Refactorisation modulaire** de la fonction `run_analysis()` :
  - Séparation claire en fonctions : `get_dossiers`, `get_sous_dossiers`, `analyze_metadata`, `build_output_entry`.
  - Réduction de la complexité cognitive (conforme aux recommandations SonarQube).
- Réorganisation du projet selon une architecture MVC claire :
  - `model/` → traitement des données (statistiques, métadonnées…)
  - `view/gui.py` → interface utilisateur avec gestion des événements

---

## 📦 Fichiers impactés
- `model/metadata_loader.py` → parsing + chargement du fichier `.metadata.xlsx`
- `model/image_analyzer.py` → collecte des propriétés des images
- `model/analysis_runner.py` → exécution propre de l’analyse (refactorisée)
- `view/gui.py` → interface graphique
- `reports\explorationdata` (pour le template et les logs d erreurs)

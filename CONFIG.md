# 🔧 Configuration du Projet DS_COVID

## 📁 Fichiers de Configuration

### `.env` - Variables d'Environnement
Le fichier `.env` contient toutes les variables de configuration du projet :
- **Chemins** : Répertoires de données, modèles, résultats
- **Paramètres ML** : Hyperparamètres, tailles de batch, époques
- **Configuration GPU/CPU** : Optimisations performance
- **Visualisation** : Styles de graphiques, couleurs

### `src/config.py` - Gestionnaire de Configuration
Le module `config.py` fournit :
- **Chargement automatique** du fichier `.env`
- **Validation** des paramètres
- **Accès centralisé** à la configuration
- **Setup automatique** de l'environnement

## 🚀 Utilisation

### 1. Configuration Initiale
```bash
# Copiez le fichier exemple
cp .env.example .env

# Éditez selon votre environnement
nano .env
```

### 2. Dans les Notebooks
```python
# Import de la configuration
from src.config import config_manager, get_config, setup_environment

# Configuration automatique
setup_environment()

# Accès aux paramètres
BATCH_SIZE = get_config('training', 'batch_size')
IMG_SIZE = get_config('image', 'img_size')
DATA_DIR = get_config('paths', 'data_dir')
```

### 3. Dans les Scripts Python
```python
import sys
from pathlib import Path

# Ajout du chemin source
sys.path.append(str(Path(__file__).parent / "src"))

from config import get_config
```

## ⚙️ Variables Importantes

### Chemins
- `PROJECT_ROOT` : Racine du projet
- `DATA_DIR` : Données COVID-19
- `MODELS_DIR` : Modèles sauvegardés
- `RESULTS_DIR` : Résultats d'analyse

### Paramètres ML
- `BATCH_SIZE` : Taille des lots (32 par défaut)
- `LEARNING_RATE` : Taux d'apprentissage (0.001)
- `EPOCHS` : Nombre d'époques (50)
- `RANDOM_SEED` : Graine aléatoire (42)

### Classes COVID
- `CLASS_NAMES` : Liste des classes
- `NUM_CLASSES` : Nombre de classes (4)

## 🔒 Sécurité

### Fichiers Protégés
- `.env` est dans `.gitignore`
- Utilisez `.env.example` pour la documentation
- Ne commitez JAMAIS le fichier `.env` réel

### Bonnes Pratiques
- Adaptez les chemins à votre environnement
- Documentez les changements importants
- Testez la configuration avant les gros entraînements

## 🛠️ Dépannage

### Problèmes Courants

1. **Fichier .env non trouvé**
   ```bash
   cp .env.example .env
   ```

2. **Chemins incorrects**
   - Vérifiez `PROJECT_ROOT`
   - Adaptez `DATA_DIR`

3. **Import config.py échoue**
   ```python
   import sys
   sys.path.append("/chemin/vers/projet/src")
   ```

4. **GPU non détecté**
   - Vérifiez `CUDA_VISIBLE_DEVICES`
   - Configurez `TF_ENABLE_ONEDNN_OPTS`

## 📊 Exemple de Configuration

```env
# Configuration type pour développement local
PROJECT_ROOT=/home/user/DS_COVID
DATA_DIR=/home/user/DS_COVID/data/raw/COVID-19_Radiography_Dataset
BATCH_SIZE=16  # Réduit pour GPU limité
EPOCHS=20      # Tests rapides
VERBOSE=1      # Debug activé
```

## 🔄 Mise à Jour

Pour ajouter de nouveaux paramètres :

1. **Ajoutez dans `.env`**
   ```env
   NEW_PARAMETER=value
   ```

2. **Mettez à jour `config.py`**
   ```python
   'new_section': {
       'new_parameter': os.getenv('NEW_PARAMETER', 'default')
   }
   ```

3. **Documentez dans `.env.example`**

4. **Testez la configuration**
   ```python
   config_manager.print_summary()
   ```

---

**💡 Cette configuration centralisée facilite la collaboration et la reproductibilité !**
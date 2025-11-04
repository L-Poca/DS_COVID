# 🔄 Guide de Migration - Ancienne → Nouvelle API

Ce guide vous aide à migrer du code utilisant l'ancienne API vers la nouvelle.

## 📋 Changements de méthodes

### ❌ Ancienne API (obsolète)

```python
# Méthode 1: load_sample_images() - SUPPRIMÉE
images, labels = loader.load_sample_images(paths, labels, n_samples=10)

# Méthode 2: load_images_with_masks() - RENOMMÉE/DIVISÉE
images, labels, masks = loader.load_images_with_masks(
    paths, labels, apply_mask=True, n_samples=100
)

# Méthode 3: load_images() wrapper confus - REMPLACÉE
images, labels, masks = loader.load_images(
    paths, labels, n_samples=100, masked=True
)
```

### ✅ Nouvelle API (claire)

```python
# Méthode 1: Chargement simple → load_images()
images, labels = loader.load_images(paths, labels, n_samples=10)

# Méthode 2a: Masques appliqués → load_masked_images()
masked_images, labels = loader.load_masked_images(paths, labels, n_samples=100)

# Méthode 2b: Masques séparés → load_images_and_masks()
images, labels, masks = loader.load_images_and_masks(paths, labels, n_samples=100)

# Plus de paramètre masked=True confus !
```

---

## 🔄 Exemples de migration

### Cas 1: Chargement simple d'images

**Avant:**
```python
# Ancien code
images, labels = loader.load_sample_images(paths, labels, n_samples=50)
```

**Après:**
```python
# Nouveau code (identique en termes de résultat)
images, labels = loader.load_images(paths, labels, n_samples=50)
```

---

### Cas 2: Images avec masques appliqués

**Avant:**
```python
# Ancien code (ambigu)
images, labels, masks = loader.load_images_with_masks(
    paths, labels, 
    apply_mask=True,  # ← Pas clair ce que ça fait
    n_samples=100
)
# On récupère les images masquées dans `images`
# Mais on récupère aussi `masks` ? Pourquoi ?
```

**Après:**
```python
# Nouveau code (clair)
masked_images, labels = loader.load_masked_images(
    paths, labels, 
    n_samples=100
)
# Clair: images avec masques APPLIQUÉS
# Pas de confusion avec les masques séparés
```

---

### Cas 3: Visualisation avec masques séparés

**Avant:**
```python
# Ancien code
images, labels, masks = loader.load_images_with_masks(
    paths, labels,
    apply_mask=False,  # ← Il faut deviner que False = masques séparés
    n_samples=10
)
```

**Après:**
```python
# Nouveau code (explicite)
images, labels, masks = loader.load_images_and_masks(
    paths, labels,
    n_samples=10
)
# Nom explicite: on récupère images ET masques
```

---

### Cas 4: Wrapper load_images() avec masked=True

**Avant:**
```python
# Ancien code (très confus)
images, labels, masks = loader.load_images(
    paths, labels,
    n_samples=100,
    masked=True  # ← Que fait masked=True exactement ?
)
# Retourne quoi ? Images masquées ? Images + masques ?
```

**Après:**
```python
# Option A: Si vous vouliez des images masquées
masked_images, labels = loader.load_masked_images(
    paths, labels,
    n_samples=100
)

# Option B: Si vous vouliez images + masques séparés
images, labels, masks = loader.load_images_and_masks(
    paths, labels,
    n_samples=100
)
```

---

## 🔍 MaskProcessor - Changements

### ❌ Ancienne méthode

```python
# batch_process_masks() - RENOMMÉE
processed_images, masks = mask_processor.batch_process_masks(
    image_paths,
    apply_mask=True
)
```

### ✅ Nouvelle méthode

```python
# batch_process_with_masks() - NOM PLUS CLAIR
processed_images, masks = mask_processor.batch_process_with_masks(
    image_paths,
    apply_mask=True
)

# MAIS: Préférez utiliser les méthodes DataLoader à la place !
```

**Note**: En général, vous ne devriez PAS appeler `MaskProcessor` directement. Utilisez les méthodes du `DataLoader` à la place.

---

## 📝 Checklist de migration

- [ ] Remplacer `load_sample_images()` par `load_images()`
- [ ] Remplacer `load_images_with_masks(..., apply_mask=True)` par `load_masked_images()`
- [ ] Remplacer `load_images_with_masks(..., apply_mask=False)` par `load_images_and_masks()`
- [ ] Supprimer les appels à `load_images(..., masked=True/False)`
- [ ] Vérifier que vous n'appelez pas `MaskProcessor` directement (sauf cas avancés)
- [ ] Mettre à jour vos imports si nécessaire
- [ ] Tester avec un petit échantillon d'abord

---

## 🚀 Script de migration automatique

Si vous avez beaucoup de code à migrer, voici des patterns de recherche/remplacement :

### Pattern 1: load_sample_images
```bash
# Rechercher
load_sample_images\(

# Remplacer par
load_images(
```

### Pattern 2: load_images_with_masks avec apply_mask=True
```bash
# Rechercher
load_images_with_masks\((.*), apply_mask=True

# Remplacer par
load_masked_images(\1
```

### Pattern 3: load_images_with_masks avec apply_mask=False
```bash
# Rechercher
load_images_with_masks\((.*), apply_mask=False

# Remplacer par
load_images_and_masks(\1
```

### Pattern 4: load_images avec masked=True
```bash
# Rechercher
load_images\((.*), masked=True

# Remplacer par
load_masked_images(\1
```

### Pattern 5: load_images avec masked=False
```bash
# Rechercher
load_images\((.*), masked=False(.*)\)

# Remplacer par
load_images(\1)
# Supprimer masked=False, c'est le comportement par défaut
```

---

## ⚠️ Points d'attention

### 1. Retour de valeurs différent

**Anciennement:**
```python
# Retournait toujours 3 valeurs
images, labels, masks = loader.load_images(..., masked=False)
# masks = None
```

**Maintenant:**
```python
# 2 valeurs pour load_images() et load_masked_images()
images, labels = loader.load_images(...)
masked_images, labels = loader.load_masked_images(...)

# 3 valeurs SEULEMENT pour load_images_and_masks()
images, labels, masks = loader.load_images_and_masks(...)
```

### 2. Paramètre n_samples

Le paramètre `n_samples` fonctionne de la même manière dans toutes les nouvelles méthodes.

### 3. MaskProcessor.batch_process_masks

Si vous utilisiez directement `batch_process_masks()`, elle a été renommée `batch_process_with_masks()`.

**MAIS** il est recommandé d'utiliser les méthodes du `DataLoader` à la place:
- `load_masked_images()` au lieu de `batch_process_with_masks(..., apply_mask=True)`
- `load_images_and_masks()` au lieu de `batch_process_with_masks(..., apply_mask=False)`

---

## 🧪 Tester votre migration

```python
# Test simple après migration
from pathlib import Path
from src.features.raf.data_loading import DataLoader
from src.features.raf.utils.config import build_config

config = build_config(Path.cwd(), environment="local")
loader = DataLoader(config=config)

paths, labels, _ = loader.load_image_paths_and_labels()

# Test 1: Chargement simple
images, labels = loader.load_images(paths, labels, n_samples=5)
assert len(images) == 5
print("✅ Test 1: load_images() OK")

# Test 2: Chargement avec masques appliqués
masked_images, labels = loader.load_masked_images(paths, labels, n_samples=5)
assert len(masked_images) == 5
print("✅ Test 2: load_masked_images() OK")

# Test 3: Chargement images + masques séparés
images, labels, masks = loader.load_images_and_masks(paths, labels, n_samples=5)
assert len(images) == 5
assert len(masks) == 5
print("✅ Test 3: load_images_and_masks() OK")

print("\n✅ Migration réussie!")
```

---

## 📚 Ressources

- **Guide d'utilisation complet**: `USAGE_GUIDE.md`
- **Exemples pratiques**: `EXAMPLES.py`
- **Architecture**: `ARCHITECTURE.md`
- **Résumé de la refactorisation**: `REFACTORING_SUMMARY.md`

---

## ❓ Questions fréquentes

**Q: Pourquoi 3 méthodes au lieu d'une seule avec des paramètres ?**  
R: Pour la clarté. Chaque méthode a un nom explicite qui indique exactement ce qu'elle fait. Plus besoin de deviner ce que fait `masked=True` ou `apply_mask=False`.

**Q: Est-ce que l'ancienne API fonctionne encore ?**  
R: Non, les anciennes méthodes ont été supprimées. Ce guide vous aide à migrer.

**Q: Y a-t-il des différences de performance ?**  
R: Non, les performances sont identiques. Seule l'interface a changé pour plus de clarté.

**Q: Puis-je encore utiliser MaskProcessor directement ?**  
R: Oui, mais c'est déconseillé. Les méthodes du DataLoader sont plus simples et couvrent 99% des cas d'usage.

**Q: Comment choisir entre les 3 méthodes ?**  
R: Consultez le tableau comparatif dans `USAGE_GUIDE.md` ou `ARCHITECTURE.md`.

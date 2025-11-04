# Guide d'utilisation du DataLoader

## 🎯 Vue d'ensemble

Le `DataLoader` offre **3 méthodes principales** pour charger vos images, selon vos besoins :

### 1️⃣ `load_images()` - Chargement simple
**Utilisation** : Charger des images préprocessées **SANS masquage**

```python
from src.features.raf.data_loading import DataLoader
from src.features.raf.utils.config import build_config

# Initialisation
config = build_config(project_root, environment="local")
loader = DataLoader(config=config)

# Charger les chemins
paths, labels, counts = loader.load_image_paths_and_labels()

# Charger 100 images sans masquage
images, labels = loader.load_images(paths, labels, n_samples=100)

print(f"Images chargées: {len(images)}")
print(f"Forme d'une image: {images[0].shape}")  # (256, 256, 3)
```

**Quand l'utiliser** :
- Entraînement de modèles sur images complètes
- Visualisation basique
- Analyse exploratoire
- Baseline sans segmentation

---

### 2️⃣ `load_masked_images()` - Images avec masques appliqués
**Utilisation** : Charger des images avec **segmentation appliquée** (zones hors poumons = noir)

```python
# Charger 100 images avec masques appliqués
masked_images, labels = loader.load_masked_images(paths, labels, n_samples=100)

print(f"Images masquées: {len(masked_images)}")
# Les zones hors masque sont mises à 0 (noires)
```

**Quand l'utiliser** :
- Entraîner un modèle en se concentrant uniquement sur les poumons
- Réduire le bruit de fond
- Améliorer les performances sur les régions d'intérêt
- Comparaison masked vs non-masked

---

### 3️⃣ `load_images_and_masks()` - Images + masques séparés
**Utilisation** : Charger images ET masques **séparément** (pour visualisation/analyse)

```python
# Charger 10 images avec leurs masques séparés
images, labels, masks = loader.load_images_and_masks(paths, labels, n_samples=10)

print(f"Images: {len(images)}")
print(f"Masques disponibles: {sum(1 for m in masks if m is not None)}")

# Visualisation côte à côte
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    axes[0, i].imshow(images[i])
    axes[0, i].set_title(f"{labels[i]}")
    axes[0, i].axis('off')
    
    if masks[i] is not None:
        axes[1, i].imshow(masks[i], cmap='gray')
    axes[1, i].set_title("Masque")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()
```

**Quand l'utiliser** :
- Visualiser les masques de segmentation
- Vérifier la qualité des masques
- Créer des overlays colorés
- Analyser la disponibilité des masques
- Debugging

---

## 📊 Autres méthodes utiles

### Vérifier la disponibilité des masques

```python
# Statistiques sur les masques
stats = loader.check_masks_availability(paths)
print(f"Masques disponibles: {stats['available']}/{stats['total']}")
print(f"Taux de disponibilité: {stats['availability_rate']:.1%}")
```

### Créer un dataset équilibré

```python
# Équilibrer le dataset (500 images max par classe)
balanced_paths, balanced_labels = loader.create_balanced_subset(
    paths, labels, max_per_class=500
)

print(f"Dataset original: {len(paths)} images")
print(f"Dataset équilibré: {len(balanced_paths)} images")
```

### Obtenir un résumé complet

```python
# Résumé du dataset
summary = loader.get_dataset_summary()
print(f"Classes: {summary['class_counts']}")
```

---

## 🔧 Fonctionnalités avancées

### Créer un overlay du masque

```python
from src.features.raf.data_loading import MaskProcessor

mask_processor = MaskProcessor(config)

# Charger une image et son masque
images, labels, masks = loader.load_images_and_masks(paths, labels, n_samples=1)

if masks[0] is not None:
    # Créer une superposition rouge du masque
    overlay = mask_processor.create_overlay(
        (images[0] * 255).astype(np.uint8),
        masks[0],
        mask_color=(255, 0, 0),  # Rouge
        alpha=0.3
    )
    
    plt.imshow(overlay)
    plt.title("Image avec overlay du masque")
    plt.show()
```

### Statistiques sur un masque

```python
if masks[0] is not None:
    stats = mask_processor.get_mask_statistics(masks[0])
    print(f"Pixels masqués: {stats['masked_pixels']:,}")
    print(f"Ratio de masquage: {stats['mask_ratio']:.1%}")
```

---

## ⚙️ Workflow complet

```python
# 1. Initialisation
config = build_config(project_root, environment="local")
loader = DataLoader(config=config)

# 2. Chargement des chemins
paths, labels, counts = loader.load_image_paths_and_labels()
print(f"Total images: {len(paths)}")
print(f"Distribution: {counts}")

# 3. Équilibrage (optionnel)
balanced_paths, balanced_labels = loader.create_balanced_subset(
    paths, labels, max_per_class=500
)

# 4. Vérification des masques
mask_stats = loader.check_masks_availability(balanced_paths)
print(f"Masques: {mask_stats['availability_rate']:.1%}")

# 5. Chargement selon le besoin

# Option A: Sans masques (baseline)
images, labels = loader.load_images(balanced_paths, balanced_labels, n_samples=1000)

# Option B: Avec masques appliqués (segmentation)
masked_images, labels = loader.load_masked_images(balanced_paths, balanced_labels, n_samples=1000)

# Option C: Visualisation avec masques séparés
viz_images, viz_labels, viz_masks = loader.load_images_and_masks(
    balanced_paths, balanced_labels, n_samples=10
)
```

---

## 🆚 Comparaison des méthodes

| Méthode | Images retournées | Masques retournés | Use Case |
|---------|-------------------|-------------------|----------|
| `load_images()` | Originales préprocessées | ❌ Non | Entraînement baseline |
| `load_masked_images()` | Avec masques appliqués | ❌ Non | Entraînement avec segmentation |
| `load_images_and_masks()` | Originales préprocessées | ✅ Oui (séparés) | Visualisation, analyse |

---

## 💡 Conseils

1. **Pour l'entraînement** : Utilisez `load_images()` ou `load_masked_images()`
2. **Pour la visualisation** : Utilisez `load_images_and_masks()`
3. **Toujours équilibrer** votre dataset avec `create_balanced_subset()`
4. **Vérifier les masques** avant de les utiliser avec `check_masks_availability()`
5. **Échantillonner** pour les tests avec `n_samples` pour économiser la RAM

---

## ❓ FAQ

**Q: Quelle est la différence entre `load_masked_images()` et `load_images_and_masks()` ?**  
R: 
- `load_masked_images()` retourne des images avec le masque **déjà appliqué** (fond noir)
- `load_images_and_masks()` retourne images ET masques **séparément** (pour visualisation)

**Q: Que se passe-t-il si un masque n'existe pas ?**  
R: 
- `load_masked_images()` : retourne l'image originale sans masquage
- `load_images_and_masks()` : retourne `None` pour le masque correspondant

**Q: Comment choisir `n_samples` ?**  
R: 
- Pour tester/visualiser : 10-50 images
- Pour l'analyse : 100-500 images
- Pour l'entraînement : ne pas spécifier (charge tout) ou utiliser `create_balanced_subset()`

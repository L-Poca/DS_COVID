# Architecture du DataLoader Refactorisé

```
┌─────────────────────────────────────────────────────────────────┐
│                         DataLoader                              │
│                    (Point d'entrée principal)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ utilise
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MaskProcessor                              │
│              (Gestion bas-niveau des masques)                   │
└─────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════
                    3 MÉTHODES PRINCIPALES
═══════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────┐
│ 1️⃣  load_images()                                                │
│    ▪ Charge images préprocessées                                │
│    ▪ SANS masquage                                              │
│    ▪ Usage: Baseline, analyse exploratoire                      │
├──────────────────────────────────────────────────────────────────┤
│ Input:  image_paths, labels, n_samples                          │
│ Output: (images, labels)                                        │
│                                                                  │
│ images[i] = [0..1] RGB (256x256x3)  <-- Original préprocessée   │
└──────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────┐
│ 2️⃣  load_masked_images()                                         │
│    ▪ Charge images avec masques APPLIQUÉS                       │
│    ▪ Zones hors poumons = noir (0)                              │
│    ▪ Usage: Entraînement avec segmentation                      │
├──────────────────────────────────────────────────────────────────┤
│ Input:  image_paths, labels, n_samples                          │
│ Output: (images_masquées, labels)                               │
│                                                                  │
│ images[i] = [0..1] RGB (256x256x3)  <-- Poumons visibles        │
│             [0, 0, 0] pour le fond    <-- Fond noir              │
└──────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────┐
│ 3️⃣  load_images_and_masks()                                      │
│    ▪ Charge images ET masques SÉPARÉS                           │
│    ▪ Masques non appliqués                                      │
│    ▪ Usage: Visualisation, analyse, debugging                   │
├──────────────────────────────────────────────────────────────────┤
│ Input:  image_paths, labels, n_samples                          │
│ Output: (images, labels, masks)                                 │
│                                                                  │
│ images[i] = [0..1] RGB (256x256x3)  <-- Original                │
│ masks[i]  = [0,255] Grayscale (256x256) <-- Masque séparé       │
│             ou None si pas disponible                            │
└──────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════
                     WORKFLOW TYPIQUE
═══════════════════════════════════════════════════════════════════

    ┌─────────────────┐
    │ Initialisation  │
    │   DataLoader    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ load_image_     │
    │ paths_and_      │
    │ labels()        │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ create_         │
    │ balanced_       │
    │ subset()        │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ check_masks_    │
    │ availability()  │
    └────────┬────────┘
             │
             ├─────────────┬─────────────┬─────────────┐
             ▼             ▼             ▼             ▼
        ┌────────┐   ┌─────────┐  ┌─────────┐  ┌──────────┐
        │ load_  │   │ load_   │  │ load_   │  │Comparaison│
        │ images │   │ masked_ │  │ images_ │  │des 3      │
        │        │   │ images  │  │ and_    │  │méthodes   │
        │        │   │         │  │ masks   │  │           │
        └────────┘   └─────────┘  └─────────┘  └──────────┘
            │             │             │            │
            ▼             ▼             ▼            ▼
        Baseline    Segmentation  Visualisation  Analyse


═══════════════════════════════════════════════════════════════════
                   MaskProcessor (Méthodes)
═══════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────┐
│ load_mask(mask_path)                                             │
│   └─> Charge un masque binaire                                  │
│       Output: ndarray [0, 255] ou None                           │
├──────────────────────────────────────────────────────────────────┤
│ get_mask_path_from_image_path(image_path)                        │
│   └─> /path/images/img.png → /path/masks/img.png                │
│       Output: str ou None                                        │
├──────────────────────────────────────────────────────────────────┤
│ apply_mask_to_image(image, mask)                                 │
│   └─> Applique masque binaire sur image                         │
│       Output: image masquée                                      │
├──────────────────────────────────────────────────────────────────┤
│ create_overlay(image, mask, color, alpha)                        │
│   └─> Superposition colorée du masque                           │
│       Output: image avec overlay                                 │
├──────────────────────────────────────────────────────────────────┤
│ get_mask_statistics(mask)                                        │
│   └─> Calcule ratio masqué, pixels, etc.                        │
│       Output: dict avec stats                                    │
├──────────────────────────────────────────────────────────────────┤
│ batch_process_with_masks(paths, apply_mask)                      │
│   └─> Traite lot d'images avec masques                          │
│       Output: (images, masks)                                    │
│       Note: Utilisé en interne par DataLoader                    │
└──────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════
                  EXEMPLE VISUEL - Différences
═══════════════════════════════════════════════════════════════════

ORIGINAL                  MASQUE                 MASKED IMAGE
(load_images)          (load_images_          (load_masked_
                        and_masks)              images)

┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│             │        │             │        │             │
│   ██████    │        │   ██████    │        │   ██████    │
│  ████████   │        │  ████████   │        │  ████████   │
│  ████████   │   ×    │  ████████   │   =    │  ████████   │
│   ██████    │        │   ██████    │        │   ██████    │
│    ████     │        │    ████     │        │             │
│             │        │             │        │             │
└─────────────┘        └─────────────┘        └─────────────┘
  Full image           Binary mask           Segmented image
  RGB [0,1]            Grayscale [0,255]     RGB [0,1]
                                             (fond = 0)


═══════════════════════════════════════════════════════════════════
                    GUIDE DE CHOIX RAPIDE
═══════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────┐
│ Besoin                           │ Méthode à utiliser          │
├──────────────────────────────────┼─────────────────────────────┤
│ Entraîner modèle baseline        │ load_images()               │
│ Entraîner avec segmentation      │ load_masked_images()        │
│ Visualiser les masques           │ load_images_and_masks()     │
│ Comparer masked vs non-masked    │ Les 2 premières             │
│ Créer des overlays colorés       │ load_images_and_masks()     │
│ Analyser qualité des masques     │ load_images_and_masks()     │
│ Debugging de la segmentation     │ load_images_and_masks()     │
└────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════
                     HIÉRARCHIE DES APPELS
═══════════════════════════════════════════════════════════════════

DataLoader.load_images()
    └─> load_and_preprocess_image()
        └─> PIL.Image.open()
        └─> resize()
        └─> normalize [0,1]

DataLoader.load_masked_images()
    └─> MaskProcessor.batch_process_with_masks(apply_mask=True)
        └─> load_and_preprocess_image()
        └─> get_mask_path_from_image_path()
        └─> load_mask()
        └─> apply_mask_to_image()  <-- Applique ici

DataLoader.load_images_and_masks()
    └─> MaskProcessor.batch_process_with_masks(apply_mask=False)
        └─> load_and_preprocess_image()
        └─> get_mask_path_from_image_path()
        └─> load_mask()
        └─> PAS d'application  <-- Retourne séparément


═══════════════════════════════════════════════════════════════════
                     FICHIERS DISPONIBLES
═══════════════════════════════════════════════════════════════════

data_loading/
├── loader.py                    ← Code principal DataLoader
├── mask_processor.py            ← Code MaskProcessor
├── USAGE_GUIDE.md              ← 📖 Guide complet d'utilisation
├── EXAMPLES.py                 ← 🚀 Exemples exécutables
├── REFACTORING_SUMMARY.md      ← 📋 Résumé des changements
└── ARCHITECTURE.md             ← 📐 Ce fichier (architecture)

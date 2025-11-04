"""
Exemples d'utilisation du DataLoader refactorisé
Démontre les 3 méthodes principales et leurs cas d'usage
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Imports du framework
from src.features.raf.data_loading import DataLoader, MaskProcessor
from src.features.raf.utils.config import build_config


# =============================================================================
# EXEMPLE 1: Chargement simple d'images (sans masques)
# =============================================================================
def example_1_simple_loading():
    """Exemple: Charger des images preprocessées SANS masquage"""
    print("=" * 60)
    print("EXEMPLE 1: Chargement simple (sans masques)")
    print("=" * 60)
    
    # Configuration
    project_root = Path.cwd()
    config = build_config(project_root, environment="local")
    loader = DataLoader(config=config)
    
    # Chargement des chemins
    paths, labels, counts = loader.load_image_paths_and_labels()
    print(f"Total images trouvées: {len(paths)}")
    print(f"Distribution par classe: {counts}")
    
    # Chargement de 100 images sans masquage
    images, labels = loader.load_images(paths, labels, n_samples=100)
    
    print(f"\n✅ {len(images)} images chargées")
    print(f"Forme d'une image: {images[0].shape}")
    print(f"Type de données: {images[0].dtype}")
    print(f"Plage de valeurs: [{images[0].min():.2f}, {images[0].max():.2f}]")
    
    return images, labels


# =============================================================================
# EXEMPLE 2: Chargement avec masques appliqués (segmentation)
# =============================================================================
def example_2_masked_loading():
    """Exemple: Charger des images avec segmentation appliquée"""
    print("\n" + "=" * 60)
    print("EXEMPLE 2: Chargement avec masques appliqués")
    print("=" * 60)
    
    # Configuration
    project_root = Path.cwd()
    config = build_config(project_root, environment="local")
    loader = DataLoader(config=config)
    
    # Chargement des chemins
    paths, labels, _ = loader.load_image_paths_and_labels()
    
    # Équilibrage du dataset
    balanced_paths, balanced_labels = loader.create_balanced_subset(
        paths, labels, max_per_class=500
    )
    
    # Vérification de la disponibilité des masques
    mask_stats = loader.check_masks_availability(balanced_paths)
    print(f"Masques disponibles: {mask_stats['available']}/{mask_stats['total']}")
    print(f"Taux: {mask_stats['availability_rate']:.1%}")
    
    # Chargement avec masques appliqués
    masked_images, labels = loader.load_masked_images(
        balanced_paths, balanced_labels, n_samples=100
    )
    
    print(f"\n✅ {len(masked_images)} images masquées chargées")
    print("Les zones hors poumons sont noires (valeur 0)")
    
    return masked_images, labels


# =============================================================================
# EXEMPLE 3: Chargement images + masques séparés (visualisation)
# =============================================================================
def example_3_visualization():
    """Exemple: Charger images et masques pour visualisation"""
    print("\n" + "=" * 60)
    print("EXEMPLE 3: Visualisation images + masques")
    print("=" * 60)
    
    # Configuration
    project_root = Path.cwd()
    config = build_config(project_root, environment="local")
    loader = DataLoader(config=config)
    
    # Chargement des chemins
    paths, labels, _ = loader.load_image_paths_and_labels()
    
    # Chargement de 10 images avec masques séparés
    images, labels, masks = loader.load_images_and_masks(
        paths, labels, n_samples=10
    )
    
    print(f"✅ {len(images)} images chargées")
    masks_available = sum(1 for m in masks if m is not None)
    print(f"Masques disponibles: {masks_available}/{len(masks)}")
    
    # Visualisation
    fig, axes = plt.subplots(3, min(5, len(images)), figsize=(15, 9))
    
    for i in range(min(5, len(images))):
        # Image originale
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(f"{labels[i]}")
        axes[0, i].axis('off')
        
        # Masque
        if masks[i] is not None:
            axes[1, i].imshow(masks[i], cmap='gray')
            axes[1, i].set_title("Masque")
        else:
            axes[1, i].text(0.5, 0.5, "Pas de masque", 
                          ha='center', va='center')
        axes[1, i].axis('off')
        
        # Image avec masque appliqué
        if masks[i] is not None:
            mask_processor = MaskProcessor(config)
            masked_img = mask_processor.apply_mask_to_image(images[i], masks[i])
            axes[2, i].imshow(masked_img)
            axes[2, i].set_title("Masqué")
        else:
            axes[2, i].imshow(images[i])
            axes[2, i].set_title("Original")
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('visualization_masks.png', dpi=150)
    print("\n📊 Visualisation sauvegardée: visualization_masks.png")
    
    return images, labels, masks


# =============================================================================
# EXEMPLE 4: Comparaison masqué vs non-masqué
# =============================================================================
def example_4_comparison():
    """Exemple: Comparer images originales vs masquées"""
    print("\n" + "=" * 60)
    print("EXEMPLE 4: Comparaison original vs masqué")
    print("=" * 60)
    
    project_root = Path.cwd()
    config = build_config(project_root, environment="local")
    loader = DataLoader(config=config)
    
    paths, labels, _ = loader.load_image_paths_and_labels()
    
    # Sélectionner les mêmes images
    sample_paths = paths[:20]
    sample_labels = labels[:20]
    
    # Charger versions originales et masquées
    original_images, orig_labels = loader.load_images(
        sample_paths, sample_labels, n_samples=20
    )
    
    masked_images, mask_labels = loader.load_masked_images(
        sample_paths, sample_labels, n_samples=20
    )
    
    # Comparaison visuelle
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i in range(5):
        # Original
        axes[0, i].imshow(original_images[i])
        axes[0, i].set_title(f"Original: {orig_labels[i]}")
        axes[0, i].axis('off')
        
        # Masqué
        axes[1, i].imshow(masked_images[i])
        axes[1, i].set_title("Masqué")
        axes[1, i].axis('off')
    
    plt.suptitle("Comparaison: Images originales vs masquées", fontsize=14)
    plt.tight_layout()
    plt.savefig('comparison_masked_vs_original.png', dpi=150)
    print("\n📊 Comparaison sauvegardée: comparison_masked_vs_original.png")


# =============================================================================
# EXEMPLE 5: Overlay coloré du masque
# =============================================================================
def example_5_overlay():
    """Exemple: Créer un overlay coloré du masque sur l'image"""
    print("\n" + "=" * 60)
    print("EXEMPLE 5: Overlay coloré des masques")
    print("=" * 60)
    
    project_root = Path.cwd()
    config = build_config(project_root, environment="local")
    loader = DataLoader(config=config)
    mask_processor = MaskProcessor(config)
    
    paths, labels, _ = loader.load_image_paths_and_labels()
    
    # Charger images et masques
    images, labels, masks = loader.load_images_and_masks(
        paths, labels, n_samples=10
    )
    
    # Créer des overlays
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    overlay_count = 0
    for i in range(len(images)):
        if masks[i] is not None and overlay_count < 5:
            # Image originale
            col = overlay_count
            axes[0, col].imshow(images[i])
            axes[0, col].set_title(f"Original: {labels[i]}")
            axes[0, col].axis('off')
            
            # Overlay rouge
            overlay = mask_processor.create_overlay(
                (images[i] * 255).astype(np.uint8),
                masks[i],
                mask_color=(255, 0, 0),
                alpha=0.4
            )
            axes[1, col].imshow(overlay)
            axes[1, col].set_title("Overlay du masque")
            axes[1, col].axis('off')
            
            # Statistiques du masque
            stats = mask_processor.get_mask_statistics(masks[i])
            print(f"\nImage {i} ({labels[i]}):")
            print(f"  Ratio masqué: {stats['mask_ratio']:.1%}")
            
            overlay_count += 1
    
    plt.suptitle("Overlays colorés des masques", fontsize=14)
    plt.tight_layout()
    plt.savefig('overlay_masks.png', dpi=150)
    print("\n📊 Overlays sauvegardés: overlay_masks.png")


# =============================================================================
# WORKFLOW COMPLET
# =============================================================================
def complete_workflow():
    """Exemple complet d'un workflow typique"""
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLET")
    print("=" * 60)
    
    # 1. Initialisation
    project_root = Path.cwd()
    config = build_config(project_root, environment="local")
    loader = DataLoader(config=config)
    print("✅ 1. DataLoader initialisé")
    
    # 2. Chargement des chemins
    paths, labels, counts = loader.load_image_paths_and_labels()
    print(f"✅ 2. {len(paths)} images trouvées")
    print(f"   Distribution: {counts}")
    
    # 3. Équilibrage
    balanced_paths, balanced_labels = loader.create_balanced_subset(
        paths, labels, max_per_class=500
    )
    print(f"✅ 3. Dataset équilibré: {len(balanced_paths)} images")
    
    # 4. Vérification des masques
    mask_stats = loader.check_masks_availability(balanced_paths)
    print(f"✅ 4. Masques: {mask_stats['availability_rate']:.1%} disponibles")
    
    # 5. Chargement selon le besoin
    print("\n📦 Chargement des données...")
    
    # Option A: Sans masques (baseline)
    images_baseline, labels_baseline = loader.load_images(
        balanced_paths, balanced_labels, n_samples=100
    )
    print(f"  📁 Baseline: {len(images_baseline)} images")
    
    # Option B: Avec masques (segmentation)
    images_masked, labels_masked = loader.load_masked_images(
        balanced_paths, balanced_labels, n_samples=100
    )
    print(f"  🎭 Masquées: {len(images_masked)} images")
    
    # Option C: Visualisation
    viz_images, viz_labels, viz_masks = loader.load_images_and_masks(
        balanced_paths, balanced_labels, n_samples=10
    )
    print(f"  🖼️ Visualisation: {len(viz_images)} images + masques")
    
    print("\n✅ Workflow terminé!")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n🚀 Exemples d'utilisation du DataLoader refactorisé\n")
    
    # Décommenter les exemples à exécuter:
    
    # example_1_simple_loading()
    # example_2_masked_loading()
    # example_3_visualization()
    # example_4_comparison()
    # example_5_overlay()
    complete_workflow()
    
    print("\n✅ Tous les exemples terminés!")

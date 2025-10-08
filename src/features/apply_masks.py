#!/usr/bin/env python3
"""
Script pour appliquer les masques aux images de radiographies COVID-19
Auteur: Rafael CEPA
Date: Octobre 2025
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm


class MaskApplicator:
    """Classe pour appliquer les masques aux images de radiographies"""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.categories = ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia']
        
    def get_image_mask_pairs(self, category):
        """Récupère les paires image/masque pour une catégorie donnée"""
        image_dir = self.dataset_path / category / 'images'
        mask_dir = self.dataset_path / category / 'masks'
        
        if not image_dir.exists() or not mask_dir.exists():
            print(f"⚠️  Répertoires manquants pour {category}")
            return []
        
        image_files = list(image_dir.glob('*.png'))
        pairs = []
        
        for image_file in image_files:
            mask_file = mask_dir / image_file.name
            if mask_file.exists():
                pairs.append((image_file, mask_file))
            else:
                print(f"⚠️  Masque manquant pour {image_file.name}")
        
        return pairs
    
    def apply_mask(self, image_path, mask_path, method='overlay'):
        """
        Applique un masque à une image
        
        Args:
            image_path: Chemin vers l'image
            mask_path: Chemin vers le masque
            method: Méthode d'application ('overlay', 'multiply', 'extract')
        
        Returns:
            Tuple (image_originale, masque, image_avec_masque)
        """
        # Charger l'image et le masque
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise ValueError(f"Impossible de charger {image_path} ou {mask_path}")
        
        # Redimensionner le masque si nécessaire
        if image.shape != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Normaliser le masque (0-1)
        mask_norm = mask.astype(np.float32) / 255.0
        
        if method == 'overlay':
            # Superposer le masque en couleur sur l'image
            # Convertir l'image en RGB pour la couleur
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # Créer un masque coloré (rouge pour les zones d'intérêt)
            mask_colored = np.zeros_like(image_rgb)
            mask_colored[:, :, 0] = mask  # Canal rouge
            # Mélanger l'image et le masque
            alpha = 0.3  # Transparence du masque
            result = cv2.addWeighted(image_rgb, 1-alpha, mask_colored, alpha, 0)
            
        elif method == 'multiply':
            # Multiplier l'image par le masque
            result = (image.astype(np.float32) * mask_norm).astype(np.uint8)
            
        elif method == 'extract':
            # Extraire seulement les zones masquées
            result = np.where(mask_norm > 0.5, image, 0).astype(np.uint8)
        
        else:
            raise ValueError(f"Méthode inconnue: {method}")
        
        return image, mask, result
    
    def visualize_sample(self, category, num_samples=3, method='overlay'):
        """Visualise quelques échantillons avec masques appliqués"""
        pairs = self.get_image_mask_pairs(category)
        
        if not pairs:
            print(f"Aucune paire image/masque trouvée pour {category}")
            return
        
        # Prendre les premiers échantillons
        samples = pairs[:min(num_samples, len(pairs))]
        
        fig, axes = plt.subplots(len(samples), 3, figsize=(15, 5*len(samples)))
        if len(samples) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (image_path, mask_path) in enumerate(samples):
            try:
                image, mask, result = self.apply_mask(image_path, mask_path, method)
                
                # Image originale
                axes[i, 0].imshow(image, cmap='gray')
                axes[i, 0].set_title(f'Image originale\n{image_path.name}')
                axes[i, 0].axis('off')
                
                # Masque
                axes[i, 1].imshow(mask, cmap='gray')
                axes[i, 1].set_title(f'Masque\n{mask_path.name}')
                axes[i, 1].axis('off')
                
                # Résultat
                if method == 'overlay':
                    axes[i, 2].imshow(result)
                else:
                    axes[i, 2].imshow(result, cmap='gray')
                axes[i, 2].set_title(f'Avec masque ({method})')
                axes[i, 2].axis('off')
                
            except Exception as e:
                print(f"Erreur avec {image_path.name}: {e}")
        
        plt.suptitle(f'Échantillons de {category}', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def process_category(self, category, output_dir, method='overlay'):
        """Traite toutes les images d'une catégorie"""
        pairs = self.get_image_mask_pairs(category)
        
        if not pairs:
            print(f"Aucune paire pour {category}")
            return
        
        output_path = Path(output_dir) / category
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🔄 Traitement de {len(pairs)} images pour {category}...")
        
        for image_path, mask_path in tqdm(pairs, desc=f"Processing {category}"):
            try:
                _, _, result = self.apply_mask(image_path, mask_path, method)
                
                # Sauvegarder le résultat
                output_file = output_path / f"masked_{image_path.name}"
                cv2.imwrite(str(output_file), result)
                
            except Exception as e:
                print(f"Erreur avec {image_path.name}: {e}")
        
        print(f"✅ {category} terminé ! Images sauvées dans {output_path}")
    
    def get_statistics(self):
        """Affiche les statistiques du dataset"""
        print("📊 Statistiques du dataset:")
        print("-" * 50)
        
        total_images = 0
        total_masks = 0
        
        for category in self.categories:
            pairs = self.get_image_mask_pairs(category)
            image_dir = self.dataset_path / category / 'images'
            mask_dir = self.dataset_path / category / 'masks'
            
            num_images = len(list(image_dir.glob('*.png'))) if image_dir.exists() else 0
            num_masks = len(list(mask_dir.glob('*.png'))) if mask_dir.exists() else 0
            num_pairs = len(pairs)
            
            print(f"{category:15} | Images: {num_images:4d} | Masques: {num_masks:4d} | Paires: {num_pairs:4d}")
            
            total_images += num_images
            total_masks += num_masks
        
        print("-" * 50)
        print(f"{'TOTAL':15} | Images: {total_images:4d} | Masques: {total_masks:4d}")


def main():
    parser = argparse.ArgumentParser(description='Appliquer des masques aux images COVID-19')
    parser.add_argument('--dataset', default='data/raw/COVID-19_Radiography_Dataset', 
                       help='Chemin vers le dataset')
    parser.add_argument('--category', choices=['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia', 'all'],
                       default='all', help='Catégorie à traiter')
    parser.add_argument('--method', choices=['overlay', 'multiply', 'extract'],
                       default='overlay', help='Méthode d\'application du masque')
    parser.add_argument('--output', default='data/processed/masked_images',
                       help='Répertoire de sortie')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualiser quelques échantillons')
    parser.add_argument('--stats', action='store_true',
                       help='Afficher les statistiques')
    
    args = parser.parse_args()
    
    # Initialiser l'applicateur de masques
    applicator = MaskApplicator(args.dataset)
    
    # Afficher les statistiques si demandé
    if args.stats:
        applicator.get_statistics()
    
    # Visualiser si demandé
    if args.visualize:
        if args.category == 'all':
            for cat in applicator.categories:
                try:
                    applicator.visualize_sample(cat, num_samples=2, method=args.method)
                except Exception as e:
                    print(f"Erreur avec {cat}: {e}")
        else:
            applicator.visualize_sample(args.category, num_samples=3, method=args.method)
    
    # Traiter les images
    if not args.visualize or input("Continuer le traitement des images? (y/n): ").lower() == 'y':
        if args.category == 'all':
            for cat in applicator.categories:
                applicator.process_category(cat, args.output, args.method)
        else:
            applicator.process_category(args.category, args.output, args.method)


if __name__ == "__main__":
    main()
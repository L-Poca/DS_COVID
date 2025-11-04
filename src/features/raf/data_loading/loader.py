# =================================
# DATA LOADER
# =================================
"""
Module de chargement des données pour le projet DS_COVID
Gère le chargement, l'analyse et l'équilibrage du dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import warnings

from ..utils.config import get_config


class DataLoader:
    """Chargeur de données pour images COVID-19"""
    
    def __init__(self, data_dir: Optional[Path] = None, config=None):
        """Initialise le chargeur de données"""
        self.config = config or get_config()
        self.data_dir = data_dir or self.config.data_dir
        
        # Initialisation du processeur de masques
        # Import local pour éviter les imports circulaires
        from .mask_processor import MaskProcessor
        self.mask_processor = MaskProcessor(self.config)
        self.classes = self.config.classes
        print(f"📂 DataLoader initialisé: {self.data_dir}")
    
    def load_image_paths_and_labels(self) -> Tuple[List[str], List[str], Dict[str, int]]:
        """Charge les chemins des images et leurs labels"""
        print("🔍 Chargement des chemins d'images...")
        
        image_paths = []
        labels = []
        class_counts = {}
        
        for class_name in self.classes:
            possible_dirs = [
                self.data_dir / class_name / "images",
                self.data_dir / class_name,
            ]
            
            class_dir = None
            for dir_path in possible_dirs:
                if dir_path.exists():
                    class_dir = dir_path
                    break
            
            if not class_dir:
                print(f"⚠️ Dossier non trouvé: {class_name}")
                class_counts[class_name] = 0
                continue
            
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(list(class_dir.glob(ext)))
            
            class_counts[class_name] = len(image_files)
            
            for img_path in image_files:
                image_paths.append(str(img_path))
                labels.append(class_name)
            
            print(f"  {class_name}: {len(image_files)} images")
        
        print(f"📊 Total: {len(image_paths)} images")
        return image_paths, labels, class_counts
    
    def create_balanced_subset(self, image_paths: List[str], labels: List[str], 
                              max_per_class: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """Crée un sous-ensemble équilibré du dataset
        
        Args:
            image_paths: Liste des chemins d'images
            labels: Liste des labels
            max_per_class: Nombre maximum d'images par classe (config par défaut)
            
        Returns:
            tuple: (subset_paths, subset_labels)
        """
        max_per_class = max_per_class or self.config.max_images_per_class
        
        print(f"⚖️ Création d'un sous-ensemble équilibré ({max_per_class} images/classe max)...")
        
        df = pd.DataFrame({'path': image_paths, 'label': labels})
        
        # Échantillonnage équilibré
        balanced_df = df.groupby('label').apply(
            lambda x: x.sample(n=min(len(x), max_per_class), 
                             random_state=self.config.random_seed)
        ).reset_index(drop=True)
        
        subset_paths = balanced_df['path'].tolist()
        subset_labels = balanced_df['label'].tolist()
        
        # Statistiques
        new_distribution = Counter(subset_labels)
        print("📊 Nouveau dataset équilibré:")
        for cls, count in new_distribution.items():
            print(f"  {cls}: {count} images")
        
        return subset_paths, subset_labels
    
    def load_sample_images(self, image_paths: List[str], labels: List[str], 
                          n_samples: int = 10) -> Tuple[List[np.ndarray], List[str]]:
        """Charge un échantillon d'images pour visualisation
        
        Args:
            image_paths: Liste des chemins d'images
            labels: Liste des labels 
            n_samples: Nombre d'images à charger
            
        Returns:
            tuple: (images, corresponding_labels)
        """
        print(f"🖼️ Chargement de {n_samples} images d'exemple...")
        
        # Sélection aléatoire
        sample_indices = np.random.choice(len(image_paths), 
                                        min(n_samples, len(image_paths)), 
                                        replace=False)
        
        sample_images = []
        sample_labels = []
        
        for idx in sample_indices:
            try:
                img = self.load_and_preprocess_image(image_paths[idx])
                if img is not None:
                    sample_images.append(img)
                    sample_labels.append(labels[idx])
            except Exception as e:
                print(f"⚠️ Erreur chargement image {idx}: {e}")
        
        print(f"✅ {len(sample_images)} images chargées avec succès")
        return sample_images, sample_labels
    
    def load_and_preprocess_image(self, image_path: str, 
                                target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """Charge et préprocesse une image
        
        Args:
            image_path: Chemin vers l'image
            target_size: Taille cible (width, height)
            
        Returns:
            Image préprocessée ou None si erreur
        """
        target_size = target_size or (self.config.img_width, self.config.img_height)
        
        try:
            # Chargement avec PIL
            with Image.open(image_path) as img:
                # Conversion en RGB si nécessaire
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Redimensionnement
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Conversion en array numpy
                img_array = np.array(img, dtype=np.float32)
                
                # Normalisation [0, 1]
                img_array = img_array / 255.0
                
                return img_array
        
        except Exception as e:
            warnings.warn(f"Erreur lors du chargement de {image_path}: {e}")
            return None
    
    def get_dataset_summary(self) -> Dict:
        """Retourne un résumé complet du dataset"""
        image_paths, labels, class_counts = self.load_image_paths_and_labels()
        stats = self.analyze_dataset(image_paths, labels)
        
        return {
            'paths': image_paths,
            'labels': labels,
            'class_counts': class_counts
        }
    
    def load_images_with_masks(self, image_paths: List[str], labels: List[str], 
                              apply_mask: bool = True, n_samples: Optional[int] = None) -> Tuple[List[np.ndarray], List[str], List[Optional[np.ndarray]]]:
        """
        Charge les images avec leurs masques
        
        Args:
            image_paths: Liste des chemins d'images
            labels: Liste des labels correspondants
            apply_mask: Si True, applique les masques aux images
            n_samples: Nombre d'échantillons à charger (None = tous)
            
        Returns:
            Tuple (images, labels, masques)
        """
        if n_samples:
            indices = np.random.choice(len(image_paths), min(n_samples, len(image_paths)), replace=False)
            image_paths = [image_paths[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        print(f"🖼️ Chargement de {len(image_paths)} images avec masques...")
        
        processed_images, masks = self.mask_processor.batch_process_masks(image_paths, apply_mask)
        
        # Filtrage pour ne garder que les images réussies
        valid_indices = [i for i, img in enumerate(processed_images) if img is not None]
        valid_images = [processed_images[i] for i in valid_indices]
        valid_labels = [labels[i] for i in valid_indices]
        valid_masks = [masks[i] for i in valid_indices]
        
        print(f"✅ {len(valid_images)} images chargées avec succès")
        
        # Statistiques sur les masques
        mask_stats = {'with_mask': 0, 'without_mask': 0}
        for mask in valid_masks:
            if mask is not None:
                mask_stats['with_mask'] += 1
            else:
                mask_stats['without_mask'] += 1
        
        print(f"📊 Masques: {mask_stats['with_mask']} disponibles, {mask_stats['without_mask']} manquants")
        
        return valid_images, valid_labels, valid_masks
    
    def get_mask_path_for_image(self, image_path: str) -> Optional[str]:
        """Retourne le chemin du masque pour une image donnée"""
        return self.mask_processor.get_mask_path_from_image_path(image_path)
    
    def check_masks_availability(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Vérifie la disponibilité des masques pour une liste d'images
        
        Args:
            image_paths: Liste des chemins d'images
            
        Returns:
            Dictionnaire avec les statistiques de disponibilité
        """
        stats = {'available': 0, 'missing': 0, 'total': len(image_paths)}
        
        for image_path in image_paths:
            mask_path = self.mask_processor.get_mask_path_from_image_path(image_path)
            if mask_path and Path(mask_path).exists():
                stats['available'] += 1
            else:
                stats['missing'] += 1
        
        stats['availability_rate'] = stats['available'] / stats['total'] if stats['total'] > 0 else 0
        
        return stats
    
    def load_images(self, image_paths: List[str], labels: List[str], 
                   n_samples: Optional[int] = None, masked: bool = False) -> Tuple[List[np.ndarray], List[str], Optional[List[Optional[np.ndarray]]]]:
        """
        Charge des images avec ou sans masques selon l'argument 'masked'
        
        Args:
            image_paths: Liste des chemins d'images
            labels: Liste des labels correspondants
            n_samples: Nombre d'échantillons à charger (None = tous)
            masked: Si True, utilise load_images_with_masks, sinon load_sample_images
            
        Returns:
            Tuple (images, labels, masks) où masks=None si masked=False
        """
        if masked:
            # Utiliser la méthode avec masques
            images, labels, masks = self.load_images_with_masks(
                image_paths=image_paths,
                labels=labels,
                apply_mask=True,
                n_samples=n_samples
            )
            return images, labels, masks
        else:
            # Utiliser la méthode simple sans masques
            images, labels_out = self.load_sample_images(
                image_paths=image_paths,
                labels=labels,
                n_samples=n_samples or 10
            )
            return images, labels_out, None
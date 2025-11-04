# =================================
# DATA LOADER
# =================================
"""
Module de chargement des données pour le projet DS_COVID
Gère le chargement, l'analyse et l'équilibrage du dataset
"""

import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image



class DataLoader:
    """Chargeur de données pour images COVID-19"""

    def __init__(self, data_dir: Optional[Path] = None, config=None):
        """Initialise le chargeur de données"""
        self.config = config
        self.data_dir = data_dir or self.config.data_dir

        # Initialisation du processeur de masques
        # Import local pour éviter les imports circulaires
        from .mask_processor import MaskProcessor

        self.mask_processor = MaskProcessor(self.config)
        self.classes = self.config.classes
        print(f"📂 DataLoader initialisé: {self.data_dir}")

    def load_image_paths_and_labels(
        self,
    ) -> Tuple[List[str], List[str], Dict[str, int]]:
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
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                image_files.extend(list(class_dir.glob(ext)))

            class_counts[class_name] = len(image_files)

            for img_path in image_files:
                image_paths.append(str(img_path))
                labels.append(class_name)

            print(f"  {class_name}: {len(image_files)} images")

        print(f"📊 Total: {len(image_paths)} images")
        return image_paths, labels, class_counts

    def create_balanced_subset(
        self,
        image_paths: List[str],
        labels: List[str],
        max_per_class: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """Crée un sous-ensemble équilibré du dataset

        Args:
            image_paths: Liste des chemins d'images
            labels: Liste des labels
            max_per_class: Nombre maximum d'images par classe (config par défaut)

        Returns:
            tuple: (subset_paths, subset_labels)
        """
        max_per_class = max_per_class or self.config.max_images_per_class

        print(
            f"⚖️ Création d'un sous-ensemble équilibré ({max_per_class} images/classe max)..."
        )

        df = pd.DataFrame({"path": image_paths, "label": labels})

        # Échantillonnage équilibré
        balanced_df = (
            df.groupby("label")
            .apply(
                lambda x: x.sample(
                    n=min(len(x), max_per_class), random_state=self.config.random_seed
                )
            )
            .reset_index(drop=True)
        )

        subset_paths = balanced_df["path"].tolist()
        subset_labels = balanced_df["label"].tolist()

        # Statistiques
        new_distribution = Counter(subset_labels)
        print("📊 Nouveau dataset équilibré:")
        for cls, count in new_distribution.items():
            print(f"  {cls}: {count} images")

        return subset_paths, subset_labels

    def load_images(
        self, 
        image_paths: List[str], 
        labels: List[str], 
        n_samples: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Charge et préprocesse des images (SANS masquage)
        
        Usage typique: chargement simple pour entraînement ou visualisation
        
        Args:
            image_paths: Liste des chemins d'images
            labels: Liste des labels correspondants
            n_samples: Nombre d'images à charger (None = toutes)
            
        Returns:
            tuple: (images_preprocessées, labels_correspondants)
            
        Example:
            >>> loader = DataLoader(config)
            >>> paths, labels, _ = loader.load_image_paths_and_labels()
            >>> images, labels = loader.load_images(paths, labels, n_samples=100)
        """
        # Échantillonnage si demandé
        if n_samples and n_samples < len(image_paths):
            indices = np.random.choice(len(image_paths), n_samples, replace=False)
            image_paths = [image_paths[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        print(f"🖼️ Chargement de {len(image_paths)} images...")

        loaded_images = []
        loaded_labels = []

        for img_path, label in zip(image_paths, labels):
            try:
                img = self.load_and_preprocess_image(img_path)
                if img is not None:
                    loaded_images.append(img)
                    loaded_labels.append(label)
            except Exception as e:
                print(f"⚠️ Erreur chargement {img_path}: {e}")

        print(f"✅ {len(loaded_images)}/{len(image_paths)} images chargées avec succès")
        return loaded_images, loaded_labels

    def load_and_preprocess_image(
        self, image_path: str, target_size: Optional[Tuple[int, int]] = None
    ) -> Optional[np.ndarray]:
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
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Redimensionnement
                img = img.resize(target_size, Image.Resampling.LANCZOS)

                # Conversion en array numpy
                img_array = np.array(img, dtype=np.float32)

                # Normalisation [0, 1]
                img_array = img_array / 255.0

                return img_array

        except Exception as e:
            warnings.warn(f"Erreur lors du chargement de {image_path}: {e}", stacklevel=2)
            return None

    def get_dataset_summary(self) -> Dict:
        """Retourne un résumé complet du dataset"""
        image_paths, labels, class_counts = self.load_image_paths_and_labels()

        return {"paths": image_paths, "labels": labels, "class_counts": class_counts}

    def load_masked_images(
        self,
        image_paths: List[str],
        labels: List[str],
        n_samples: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Charge des images préprocessées AVEC masques appliqués (segmentation)
        
        Usage typique: pour entraîner un modèle sur des images segmentées
        Les zones hors masque sont mises à 0 (noir)
        
        Args:
            image_paths: Liste des chemins d'images
            labels: Liste des labels correspondants
            n_samples: Nombre d'images à charger (None = toutes)
            
        Returns:
            tuple: (images_masquées, labels_correspondants)
            
        Example:
            >>> loader = DataLoader(config)
            >>> paths, labels, _ = loader.load_image_paths_and_labels()
            >>> masked_imgs, labels = loader.load_masked_images(paths, labels, n_samples=100)
        """
        # Échantillonnage si demandé
        if n_samples and n_samples < len(image_paths):
            indices = np.random.choice(len(image_paths), n_samples, replace=False)
            image_paths = [image_paths[i] for i in indices]
            labels = [labels[i] for i in indices]

        print(f"🎭 Chargement de {len(image_paths)} images avec masques appliqués...")

        processed_images, _ = self.mask_processor.batch_process_with_masks(
            image_paths, apply_mask=True
        )

        # Filtrage pour ne garder que les images réussies
        valid_indices = [i for i, img in enumerate(processed_images) if img is not None]
        valid_images = [processed_images[i] for i in valid_indices]
        valid_labels = [labels[i] for i in valid_indices]

        print(f"✅ {len(valid_images)}/{len(image_paths)} images masquées chargées")

        return valid_images, valid_labels

    def load_images_and_masks(
        self,
        image_paths: List[str],
        labels: List[str],
        n_samples: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[str], List[Optional[np.ndarray]]]:
        """
        Charge des images ET leurs masques séparément (pour visualisation/analyse)
        
        Usage typique: pour visualiser les masques ou analyser la segmentation
        Les masques sont retournés séparément, non appliqués aux images
        
        Args:
            image_paths: Liste des chemins d'images
            labels: Liste des labels correspondants
            n_samples: Nombre d'images à charger (None = toutes)
            
        Returns:
            tuple: (images, labels, masques)
            - images: images originales préprocessées
            - labels: labels correspondants
            - masques: masques binaires (ou None si pas disponible)
            
        Example:
            >>> loader = DataLoader(config)
            >>> paths, labels, _ = loader.load_image_paths_and_labels()
            >>> imgs, labs, masks = loader.load_images_and_masks(paths, labels, n_samples=10)
            >>> # Visualiser image et masque côte à côte
        """
        # Échantillonnage si demandé
        if n_samples and n_samples < len(image_paths):
            indices = np.random.choice(len(image_paths), n_samples, replace=False)
            image_paths = [image_paths[i] for i in indices]
            labels = [labels[i] for i in indices]

        print(f"🖼️ Chargement de {len(image_paths)} images + masques séparés...")

        processed_images, masks = self.mask_processor.batch_process_with_masks(
            image_paths, apply_mask=False
        )

        # Filtrage pour ne garder que les images réussies
        valid_indices = [i for i, img in enumerate(processed_images) if img is not None]
        valid_images = [processed_images[i] for i in valid_indices]
        valid_labels = [labels[i] for i in valid_indices]
        valid_masks = [masks[i] for i in valid_indices]

        print(f"✅ {len(valid_images)}/{len(image_paths)} images chargées")

        # Statistiques sur les masques
        masks_available = sum(1 for m in valid_masks if m is not None)
        print(f"📊 Masques: {masks_available} disponibles, {len(valid_masks) - masks_available} manquants")

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
        stats: Dict[str, Any] = {"available": 0, "missing": 0, "total": len(image_paths)}

        for image_path in image_paths:
            mask_path = self.mask_processor.get_mask_path_from_image_path(image_path)
            if mask_path and Path(mask_path).exists():
                stats["available"] += 1
            else:
                stats["missing"] += 1

        stats["availability_rate"] = (
            stats["available"] / stats["total"] if stats["total"] > 0 else 0.0
        )

        return stats

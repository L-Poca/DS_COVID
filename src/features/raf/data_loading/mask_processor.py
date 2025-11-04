"""
Gestion des masques pour le framework RAF
Gère le chargement et l'application des masques de segmentation
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image
import warnings

from ..utils.config import Config


class MaskProcessor:
    """Processeur de masques pour la segmentation des radiographies pulmonaires"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def load_mask(self, mask_path: str, target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """
        Charge un masque de segmentation
        
        Args:
            mask_path: Chemin vers le masque
            target_size: Taille cible (width, height)
            
        Returns:
            Masque sous forme d'array numpy ou None si erreur
        """
        target_size = target_size or (self.config.img_width, self.config.img_height)
        
        try:
            # Chargement avec PIL
            with Image.open(mask_path) as mask:
                # Conversion en niveaux de gris si nécessaire
                if mask.mode != 'L':
                    mask = mask.convert('L')
                
                # Redimensionnement
                mask = mask.resize(target_size, Image.Resampling.NEAREST)
                
                # Conversion en array numpy
                mask_array = np.array(mask, dtype=np.uint8)
                
                # Normalisation binaire (0 ou 255)
                mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
                
                return mask_array
                
        except Exception as e:
            warnings.warn(f"Erreur lors du chargement du masque {mask_path}: {e}", stacklevel=2)
            return None    def get_mask_path_from_image_path(self, image_path: str) -> Optional[str]:
        """
        Déduit le chemin du masque à partir du chemin de l'image
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Chemin vers le masque correspondant ou None
        """
        try:
            path_obj = Path(image_path)
            
            # Remplace 'images' par 'masks' dans le chemin
            mask_path = str(path_obj).replace('/images/', '/masks/')
            
            # Vérification que le fichier existe
            if Path(mask_path).exists():
                return mask_path
            else:
                return None
                
        except Exception as e:
            warnings.warn(f"Erreur génération chemin masque pour {image_path}: {e}", stacklevel=2)
            return None
    
    def apply_mask_to_image(self, image: np.ndarray, mask: np.ndarray, 
                           background_value: float = 0.0) -> np.ndarray:
        """
        Applique un masque à une image
        
        Args:
            image: Image en array numpy (H, W, C) ou (H, W)
            mask: Masque binaire (H, W)
            background_value: Valeur pour les pixels masqués
            
        Returns:
            Image avec masque appliqué
        """
        # Normalisation du masque (0 ou 1)
        mask_normalized = (mask > 127).astype(np.float32)
        
        # Application du masque
        if len(image.shape) == 3:  # Image couleur
            # Expansion du masque pour les 3 canaux
            mask_3d = np.expand_dims(mask_normalized, axis=2)
            masked_image = image * mask_3d + background_value * (1 - mask_3d)
        else:  # Image en niveaux de gris
            masked_image = image * mask_normalized + background_value * (1 - mask_normalized)
        
        return masked_image.astype(image.dtype)
    
    def create_overlay(self, image: np.ndarray, mask: np.ndarray, 
                      mask_color: Tuple[int, int, int] = (255, 0, 0),
                      alpha: float = 0.3) -> np.ndarray:
        """
        Crée une superposition colorée du masque sur l'image
        
        Args:
            image: Image originale
            mask: Masque binaire
            mask_color: Couleur du masque (R, G, B)
            alpha: Transparence du masque
            
        Returns:
            Image avec superposition du masque
        """
        # Conversion de l'image en couleur si nécessaire
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()
        
        # Création du masque coloré
        mask_colored = np.zeros_like(image_rgb)
        mask_binary = mask > 127
        
        for i, color in enumerate(mask_color):
            mask_colored[:, :, i] = mask_binary * color
        
        # Superposition avec transparence
        overlay = cv2.addWeighted(image_rgb, 1 - alpha, mask_colored, alpha, 0)
        
        return overlay
    
    def get_mask_statistics(self, mask: np.ndarray) -> dict:
        """
        Calcule des statistiques sur le masque
        
        Args:
            mask: Masque binaire
            
        Returns:
            Dictionnaire avec les statistiques
        """
        mask_binary = mask > 127
        total_pixels = mask.size
        masked_pixels = np.sum(mask_binary)
        
        return {
            'total_pixels': total_pixels,
            'masked_pixels': int(masked_pixels),
            'mask_ratio': float(masked_pixels / total_pixels),
            'background_pixels': int(total_pixels - masked_pixels),
            'background_ratio': float((total_pixels - masked_pixels) / total_pixels)
        }
    
    def batch_process_with_masks(
        self, 
        image_paths: List[str], 
        apply_mask: bool = False
    ) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """
        Traite un lot d'images avec leurs masques
        
        Args:
            image_paths: Liste des chemins d'images
            apply_mask: 
                - Si False: retourne images originales + masques séparés
                - Si True: retourne images avec masques APPLIQUÉS + masques
            
        Returns:
            Tuple (images_processées, masques)
            - Si apply_mask=False: (images_originales, masques)
            - Si apply_mask=True: (images_masquées, masques)
            
        Note:
            Préférez utiliser les méthodes du DataLoader:
            - load_images() : images sans masques
            - load_masked_images() : images avec masques appliqués
            - load_images_and_masks() : images + masques séparés
        """
        processed_images = []
        masks = []
        
        for image_path in image_paths:
            # Chargement de l'image
            try:
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((self.config.img_width, self.config.img_height), Image.Resampling.LANCZOS)
                    image_array = np.array(img, dtype=np.float32) / 255.0
            except Exception as e:
                warnings.warn(f"Erreur chargement image {image_path}: {e}", stacklevel=2)
                processed_images.append(None)
                masks.append(None)
                continue
            
            # Chargement du masque
            mask_path = self.get_mask_path_from_image_path(image_path)
            if mask_path:
                mask = self.load_mask(mask_path)
                if mask is not None and apply_mask:
                    # Application du masque à l'image
                    image_array = self.apply_mask_to_image(image_array, mask)
            else:
                mask = None
            
            processed_images.append(image_array)
            masks.append(mask)
        
        return processed_images, masks
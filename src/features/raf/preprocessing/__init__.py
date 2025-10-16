"""
Module de préprocessing pour le framework RAF
Gère le preprocessing des images et la préparation des données
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
from typing import Tuple, Optional, Dict, Any, List

from ..utils.config import Config


class ImagePreprocessor:
    """
    Classe pour le préprocessing d'images médicales
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialise le préprocesseur d'images
        
        Args:
            target_size: Taille cible pour le redimensionnement
        """
        self.target_size = target_size
        self.scaler = None
        
    def resize_image(self, image: np.ndarray, maintain_aspect_ratio: bool = False) -> np.ndarray:
        """
        Redimensionne une image
        
        Args:
            image: Image à redimensionner
            maintain_aspect_ratio: Conserver le ratio d'aspect
            
        Returns:
            Image redimensionnée
        """
        if maintain_aspect_ratio:
            h, w = image.shape[:2]
            target_h, target_w = self.target_size
            
            # Calculer le ratio pour maintenir les proportions
            ratio = min(target_w/w, target_h/h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            
            # Redimensionner
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Padding pour atteindre la taille cible
            delta_w = target_w - new_w
            delta_h = target_h - new_h
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)
            
            # Padding avec la couleur moyenne
            color = [int(image.mean())] * image.shape[2] if len(image.shape) == 3 else int(image.mean())
            resized = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                       cv2.BORDER_CONSTANT, value=color)
        else:
            resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            
        return resized
    
    def normalize_image(self, image: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalise une image
        
        Args:
            image: Image à normaliser
            method: Méthode de normalisation ('minmax', 'zscore', 'clahe')
            
        Returns:
            Image normalisée
        """
        if method == 'minmax':
            # Normalisation Min-Max [0, 1]
            image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
        elif method == 'zscore':
            # Normalisation Z-score
            image_norm = (image - image.mean()) / (image.std() + 1e-8)
            
        elif method == 'clahe':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if len(image.shape) == 3:
                # Image couleur
                lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                image_norm = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                image_norm = image_norm.astype(np.float32) / 255.0
            else:
                # Image en niveaux de gris
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                image_norm = clahe.apply(image.astype(np.uint8))
                image_norm = image_norm.astype(np.float32) / 255.0
        else:
            warnings.warn(f"Méthode {method} non reconnue, utilisation de minmax")
            image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
        return image_norm
    
    def apply_filters(self, image: np.ndarray, filters: List[str] = None) -> np.ndarray:
        """
        Applique des filtres de préprocessing
        
        Args:
            image: Image source
            filters: Liste des filtres à appliquer
            
        Returns:
            Image filtrée
        """
        if filters is None:
            filters = ['gaussian_blur']
            
        filtered_image = image.copy()
        
        for filter_name in filters:
            if filter_name == 'gaussian_blur':
                filtered_image = cv2.GaussianBlur(filtered_image, (3, 3), 0)
                
            elif filter_name == 'median_blur':
                filtered_image = cv2.medianBlur(filtered_image.astype(np.uint8), 3)
                
            elif filter_name == 'bilateral':
                filtered_image = cv2.bilateralFilter(filtered_image.astype(np.uint8), 9, 75, 75)
                
            elif filter_name == 'sharpen':
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                filtered_image = cv2.filter2D(filtered_image, -1, kernel)
                
            elif filter_name == 'edge_enhance':
                if len(filtered_image.shape) == 3:
                    gray = cv2.cvtColor(filtered_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    gray = filtered_image.astype(np.uint8)
                edges = cv2.Canny(gray, 50, 150)
                if len(filtered_image.shape) == 3:
                    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                filtered_image = cv2.addWeighted(filtered_image.astype(np.uint8), 0.8, edges, 0.2, 0)
                
        return filtered_image.astype(np.float32)
    
    def preprocess_batch(self, images: np.ndarray, normalize_method: str = 'minmax',
                        apply_filters_flag: bool = True, filters: List[str] = None) -> np.ndarray:
        """
        Préprocess un batch d'images
        
        Args:
            images: Batch d'images
            normalize_method: Méthode de normalisation
            apply_filters_flag: Appliquer les filtres
            filters: Liste des filtres
            
        Returns:
            Batch d'images préprocessées
        """
        processed_images = []
        
        print(f"🔄 Preprocessing de {len(images)} images...")
        
        for i, img in enumerate(images):
            try:
                # Redimensionnement
                resized = self.resize_image(img)
                
                # Filtrage
                if apply_filters_flag:
                    filtered = self.apply_filters(resized, filters)
                else:
                    filtered = resized
                
                # Normalisation
                normalized = self.normalize_image(filtered, normalize_method)
                
                processed_images.append(normalized)
                
            except Exception as e:
                warnings.warn(f"Erreur preprocessing image {i}: {e}")
                # Image par défaut en cas d'erreur
                default_img = np.zeros((*self.target_size, 3))
                processed_images.append(default_img)
        
        processed_array = np.array(processed_images)
        print(f"✅ Preprocessing terminé: {processed_array.shape}")
        
        return processed_array


def prepare_data_splits(X: np.ndarray, y: np.ndarray, 
                       test_size: float = 0.2, val_size: float = 0.2,
                       random_state: int = 42, stratify: bool = True) -> Tuple:
    """
    Divise les données en ensembles train/val/test
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion pour test
        val_size: Proportion pour validation (sur le train restant)
        random_state: Graine aléatoire
        stratify: Stratification des splits
        
    Returns:
        Tuple (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print(f"📊 Division des données:")
    print(f"   Dataset original: {X.shape[0]} échantillons")
    
    # Stratification si demandée
    stratify_param = y if stratify else None
    
    # Premier split: train+val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=stratify_param
    )
    
    # Deuxième split: train / val
    val_size_adjusted = val_size / (1 - test_size)  # Ajuster la proportion
    stratify_temp = y_temp if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=stratify_temp
    )
    
    print(f"   Train: {X_train.shape[0]} échantillons ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"   Val:   {X_val.shape[0]} échantillons ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"   Test:  {X_test.shape[0]} échantillons ({X_test.shape[0]/X.shape[0]*100:.1f}%)")
    
    # Vérification de la distribution des classes
    unique_classes = np.unique(y)
    print(f"\n📈 Distribution des classes:")
    
    for split_name, split_y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"   {split_name}:")
        for class_label in unique_classes:
            count = np.sum(split_y == class_label)
            percentage = count / len(split_y) * 100
            print(f"     Classe {class_label}: {count} ({percentage:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, 
                  method: str = 'standard') -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Normalise les features avec ajustement sur train seulement
    
    Args:
        X_train, X_val, X_test: Ensembles de données
        method: Méthode de normalisation ('standard', 'minmax')
        
    Returns:
        Tuple (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    print(f"🔧 Normalisation des features (méthode: {method})")
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Méthode {method} non supportée")
    
    # Ajustement sur train seulement
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   Train: {X_train_scaled.shape}")
    print(f"   Val:   {X_val_scaled.shape}")
    print(f"   Test:  {X_test_scaled.shape}")
    
    # Statistiques après normalisation
    print(f"   Moyennes train: min={X_train_scaled.mean(axis=0).min():.3f}, max={X_train_scaled.mean(axis=0).max():.3f}")
    print(f"   Std train: min={X_train_scaled.std(axis=0).min():.3f}, max={X_train_scaled.std(axis=0).max():.3f}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def encode_labels(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Tuple:
    """
    Encode les labels en format numérique
    
    Args:
        y_train, y_val, y_test: Labels des différents splits
        
    Returns:
        Tuple (y_train_encoded, y_val_encoded, y_test_encoded, label_encoder, class_names)
    """
    print("🏷️ Encodage des labels")
    
    label_encoder = LabelEncoder()
    
    # Ajustement sur train seulement
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    class_names = label_encoder.classes_
    print(f"   Classes détectées: {list(class_names)}")
    print(f"   Nombre de classes: {len(class_names)}")
    
    # Mapping des classes
    print("   Mapping classe -> numéro:")
    for i, class_name in enumerate(class_names):
        print(f"     {class_name} -> {i}")
    
    return y_train_encoded, y_val_encoded, y_test_encoded, label_encoder, class_names


def remove_outliers(X: np.ndarray, y: np.ndarray, method: str = 'iqr',
                   contamination: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Supprime les outliers du dataset
    
    Args:
        X: Features
        y: Labels
        method: Méthode de détection ('iqr', 'isolation_forest', 'zscore')
        contamination: Proportion d'outliers attendue
        
    Returns:
        Tuple (X_clean, y_clean, outlier_mask)
    """
    print(f"🧹 Détection d'outliers (méthode: {method})")
    n_samples_original = X.shape[0]
    
    if method == 'iqr':
        # Méthode IQR sur chaque feature
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Un échantillon est outlier s'il dépasse les bornes sur au moins une feature
        outlier_mask = np.any((X < lower_bound) | (X > upper_bound), axis=1)
        
    elif method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_predictions = iso_forest.fit_predict(X)
        outlier_mask = outlier_predictions == -1
        
    elif method == 'zscore':
        # Z-score avec seuil à 3
        z_scores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8))
        outlier_mask = np.any(z_scores > 3, axis=1)
        
    else:
        raise ValueError(f"Méthode {method} non supportée")
    
    # Suppression des outliers
    X_clean = X[~outlier_mask]
    y_clean = y[~outlier_mask]
    
    n_outliers = np.sum(outlier_mask)
    n_samples_clean = X_clean.shape[0]
    
    print(f"   Échantillons originaux: {n_samples_original}")
    print(f"   Outliers détectés: {n_outliers} ({n_outliers/n_samples_original*100:.1f}%)")
    print(f"   Échantillons gardés: {n_samples_clean} ({n_samples_clean/n_samples_original*100:.1f}%)")
    
    return X_clean, y_clean, outlier_mask
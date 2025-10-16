# =================================
# GRADCAM EXPLAINER - RAF
# =================================
"""
Module GradCAM pour l'explication des prédictions de modèles CNN
Implémente Grad-CAM, Grad-CAM++, et d'autres techniques de visualisation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, Dict, List, Tuple, Any
import cv2
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow non installé. Utilisez: pip install tensorflow")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL non installé. Utilisez: pip install Pillow")


class GradCAMExplainer:
    """
    Classe principale pour les explications GradCAM
    """
    
    def __init__(self, model: Any, layer_name: Optional[str] = None):
        """
        Initialise l'explainer GradCAM
        
        Args:
            model: Modèle CNN entraîné (TensorFlow/Keras)
            layer_name: Nom de la couche pour GradCAM (auto-détection si None)
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow n'est pas installé")
            
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        self.grad_model = None
        self._build_grad_model()
        
    def _find_last_conv_layer(self) -> str:
        """Trouve automatiquement la dernière couche convolutionnelle"""
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:  # Conv2D layer
                return layer.name
        
        # Si aucune couche conv trouvée, prendre la dernière avant les dense
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower() or 'block' in layer.name.lower():
                return layer.name
                
        raise ValueError("Aucune couche convolutionnelle trouvée dans le modèle")
    
    def _build_grad_model(self):
        """Construit le modèle de gradient pour GradCAM"""
        try:
            # Trouver la couche cible
            target_layer = None
            for layer in self.model.layers:
                if layer.name == self.layer_name:
                    target_layer = layer
                    break
            
            if target_layer is None:
                raise ValueError(f"Couche '{self.layer_name}' non trouvée dans le modèle")
            
            # Créer le modèle de gradient
            self.grad_model = keras.models.Model(
                inputs=self.model.inputs,
                outputs=[target_layer.output, self.model.output]
            )
            
        except Exception as e:
            print(f"Erreur lors de la construction du modèle de gradient: {e}")
            raise
    
    def generate_gradcam(self, img: np.ndarray, class_idx: Optional[int] = None,
                        alpha: float = 0.4, colormap: str = 'jet') -> Dict[str, np.ndarray]:
        """
        Génère une carte GradCAM pour une image
        
        Args:
            img: Image d'entrée (format modèle)
            class_idx: Index de la classe cible (classe prédite si None)
            alpha: Transparence de la superposition
            colormap: Colormap pour la heatmap
            
        Returns:
            Dictionnaire contenant les cartes générées
        """
        # S'assurer que l'image a la bonne forme
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)
        
        # Calcul des gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img)
            
            if class_idx is None:
                class_idx = np.argmax(predictions[0])
            
            loss = predictions[:, class_idx]
        
        # Gradients par rapport à la couche convolutionnelle
        grads = tape.gradient(loss, conv_outputs)
        
        # Pooling global des gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiplication des activations par les gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normalisation
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Redimensionnement à la taille de l'image originale
        original_img = img[0]
        if len(original_img.shape) == 3:
            img_height, img_width = original_img.shape[:2]
        else:
            img_height, img_width = original_img.shape
            
        heatmap = cv2.resize(heatmap, (img_width, img_height))
        
        # Application du colormap
        heatmap_colored = plt.cm.get_cmap(colormap)(heatmap)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Préparation de l'image originale pour superposition
        if len(original_img.shape) == 3 and original_img.shape[-1] == 3:
            # Image couleur
            original_for_overlay = (original_img * 255).astype(np.uint8)
        elif len(original_img.shape) == 3 and original_img.shape[-1] == 1:
            # Image en niveaux de gris avec channel
            original_for_overlay = np.repeat((original_img[:, :, 0] * 255).astype(np.uint8)[:, :, np.newaxis], 3, axis=2)
        else:
            # Image en niveaux de gris
            original_for_overlay = np.repeat((original_img * 255).astype(np.uint8)[:, :, np.newaxis], 3, axis=2)
        
        # Superposition
        overlay = cv2.addWeighted(original_for_overlay, 1-alpha, heatmap_colored, alpha, 0)
        
        return {
            'heatmap': heatmap,
            'heatmap_colored': heatmap_colored,
            'original': original_for_overlay,
            'overlay': overlay,
            'predicted_class': class_idx,
            'prediction_confidence': float(predictions[0][class_idx])
        }
    
    def generate_gradcam_plus_plus(self, img: np.ndarray, class_idx: Optional[int] = None,
                                  alpha: float = 0.4, colormap: str = 'jet') -> Dict[str, np.ndarray]:
        """
        Génère une carte GradCAM++ (version améliorée)
        
        Args:
            img: Image d'entrée
            class_idx: Index de la classe cible
            alpha: Transparence de la superposition
            colormap: Colormap pour la heatmap
            
        Returns:
            Dictionnaire contenant les cartes générées
        """
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)
        
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape3:
                    conv_outputs, predictions = self.grad_model(img)
                    if class_idx is None:
                        class_idx = np.argmax(predictions[0])
                    loss = predictions[:, class_idx]
                
                # Gradients de premier ordre
                grads_1 = tape3.gradient(loss, conv_outputs)
            # Gradients de second ordre
            grads_2 = tape2.gradient(grads_1, conv_outputs)
        # Gradients de troisième ordre
        grads_3 = tape1.gradient(grads_2, conv_outputs)
        
        # Calcul des poids alpha pour GradCAM++
        global_sum = tf.reduce_sum(conv_outputs, axis=(1, 2), keepdims=True)
        alpha_num = grads_2
        alpha_denom = 2.0 * grads_2 + tf.reduce_sum(grads_3 * global_sum, axis=(1, 2), keepdims=True)
        alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
        alphas = alpha_num / alpha_denom
        
        # Calcul des poids
        weights = tf.reduce_sum(alphas * tf.nn.relu(grads_1), axis=(1, 2))
        
        # Génération de la heatmap
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Redimensionnement et application du colormap
        original_img = img[0]
        if len(original_img.shape) == 3:
            img_height, img_width = original_img.shape[:2]
        else:
            img_height, img_width = original_img.shape
            
        heatmap = cv2.resize(heatmap, (img_width, img_height))
        heatmap_colored = plt.cm.get_cmap(colormap)(heatmap)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Préparation de l'image pour superposition
        if len(original_img.shape) == 3 and original_img.shape[-1] == 3:
            original_for_overlay = (original_img * 255).astype(np.uint8)
        elif len(original_img.shape) == 3 and original_img.shape[-1] == 1:
            original_for_overlay = np.repeat((original_img[:, :, 0] * 255).astype(np.uint8)[:, :, np.newaxis], 3, axis=2)
        else:
            original_for_overlay = np.repeat((original_img * 255).astype(np.uint8)[:, :, np.newaxis], 3, axis=2)
        
        overlay = cv2.addWeighted(original_for_overlay, 1-alpha, heatmap_colored, alpha, 0)
        
        return {
            'heatmap': heatmap,
            'heatmap_colored': heatmap_colored,
            'original': original_for_overlay,
            'overlay': overlay,
            'predicted_class': class_idx,
            'prediction_confidence': float(predictions[0][class_idx])
        }
    
    def plot_gradcam(self, results: Dict[str, np.ndarray], title: str = "GradCAM Analysis",
                    class_names: Optional[List[str]] = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Affiche les résultats GradCAM
        
        Args:
            results: Résultats de generate_gradcam
            title: Titre du graphique
            class_names: Noms des classes
            save_path: Chemin de sauvegarde
            
        Returns:
            Figure matplotlib
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Image originale
        axes[0].imshow(results['original'])
        axes[0].set_title('Image Originale')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(results['heatmap'], cmap='jet')
        axes[1].set_title('Heatmap GradCAM')
        axes[1].axis('off')
        
        # Heatmap colorée
        axes[2].imshow(results['heatmap_colored'])
        axes[2].set_title('Heatmap Colorée')
        axes[2].axis('off')
        
        # Superposition
        axes[3].imshow(results['overlay'])
        predicted_class = results['predicted_class']
        confidence = results['prediction_confidence']
        
        if class_names and predicted_class < len(class_names):
            class_name = class_names[predicted_class]
        else:
            class_name = f"Classe {predicted_class}"
            
        axes[3].set_title(f'Overlay\n{class_name} ({confidence:.2%})')
        axes[3].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_multiple_layers(self, img: np.ndarray, layer_names: List[str],
                               class_idx: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Analyse GradCAM sur plusieurs couches
        
        Args:
            img: Image d'entrée
            layer_names: Liste des noms de couches
            class_idx: Index de la classe cible
            
        Returns:
            Dictionnaire des résultats par couche
        """
        results = {}
        original_layer = self.layer_name
        
        for layer_name in layer_names:
            try:
                self.layer_name = layer_name
                self._build_grad_model()
                results[layer_name] = self.generate_gradcam(img, class_idx)
            except Exception as e:
                print(f"Erreur avec la couche {layer_name}: {e}")
                continue
        
        # Restaurer la couche originale
        self.layer_name = original_layer
        self._build_grad_model()
        
        return results
    
    def compare_classes(self, img: np.ndarray, class_indices: List[int],
                       class_names: Optional[List[str]] = None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Compare les GradCAM pour différentes classes
        
        Args:
            img: Image d'entrée
            class_indices: Liste des indices de classes
            class_names: Noms des classes
            
        Returns:
            Dictionnaire des résultats par classe
        """
        results = {}
        
        for class_idx in class_indices:
            try:
                results[class_idx] = self.generate_gradcam(img, class_idx)
            except Exception as e:
                print(f"Erreur avec la classe {class_idx}: {e}")
                continue
        
        return results


def generate_gradcam(model: Any, img: np.ndarray, layer_name: Optional[str] = None,
                    class_idx: Optional[int] = None, method: str = 'gradcam') -> Dict[str, np.ndarray]:
    """
    Fonction helper pour générer rapidement une carte GradCAM
    
    Args:
        model: Modèle CNN
        img: Image d'entrée
        layer_name: Nom de la couche
        class_idx: Index de la classe
        method: Méthode ('gradcam' ou 'gradcam++')
        
    Returns:
        Résultats GradCAM
    """
    explainer = GradCAMExplainer(model, layer_name)
    
    if method == 'gradcam++':
        return explainer.generate_gradcam_plus_plus(img, class_idx)
    else:
        return explainer.generate_gradcam(img, class_idx)


def visualize_gradcam_comparison(results: Dict[str, Dict[str, np.ndarray]], 
                               title: str = "GradCAM Comparison",
                               class_names: Optional[List[str]] = None) -> plt.Figure:
    """
    Visualise une comparaison de plusieurs résultats GradCAM
    
    Args:
        results: Dictionnaire des résultats {nom: résultat_gradcam}
        title: Titre du graphique
        class_names: Noms des classes
        
    Returns:
        Figure matplotlib
    """
    n_results = len(results)
    fig, axes = plt.subplots(n_results, 4, figsize=(16, 4 * n_results))
    
    if n_results == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, result) in enumerate(results.items()):
        # Image originale
        axes[idx, 0].imshow(result['original'])
        axes[idx, 0].set_title(f'{name} - Original')
        axes[idx, 0].axis('off')
        
        # Heatmap
        axes[idx, 1].imshow(result['heatmap'], cmap='jet')
        axes[idx, 1].set_title(f'{name} - Heatmap')
        axes[idx, 1].axis('off')
        
        # Heatmap colorée
        axes[idx, 2].imshow(result['heatmap_colored'])
        axes[idx, 2].set_title(f'{name} - Colorée')
        axes[idx, 2].axis('off')
        
        # Superposition
        axes[idx, 3].imshow(result['overlay'])
        predicted_class = result['predicted_class']
        confidence = result['prediction_confidence']
        
        if class_names and predicted_class < len(class_names):
            class_name = class_names[predicted_class]
        else:
            class_name = f"Classe {predicted_class}"
            
        axes[idx, 3].set_title(f'{name} - Overlay\n{class_name} ({confidence:.2%})')
        axes[idx, 3].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def analyze_gradcam_regions(heatmap: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Analyse les régions importantes dans une heatmap GradCAM
    
    Args:
        heatmap: Heatmap GradCAM normalisée
        threshold: Seuil pour définir les régions importantes
        
    Returns:
        Statistiques des régions
    """
    # Binarisation de la heatmap
    binary_mask = (heatmap > threshold).astype(np.uint8)
    
    # Calcul des statistiques
    total_pixels = heatmap.size
    important_pixels = np.sum(binary_mask)
    importance_ratio = important_pixels / total_pixels
    
    # Intensité moyenne des régions importantes
    if important_pixels > 0:
        avg_importance_intensity = np.mean(heatmap[binary_mask == 1])
        max_importance_intensity = np.max(heatmap[binary_mask == 1])
    else:
        avg_importance_intensity = 0
        max_importance_intensity = 0
    
    # Centroïde des régions importantes
    if important_pixels > 0:
        y_coords, x_coords = np.where(binary_mask == 1)
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
    else:
        centroid_y, centroid_x = heatmap.shape[0] // 2, heatmap.shape[1] // 2
    
    return {
        'total_pixels': total_pixels,
        'important_pixels': important_pixels,
        'importance_ratio': importance_ratio,
        'avg_importance_intensity': avg_importance_intensity,
        'max_importance_intensity': max_importance_intensity,
        'centroid': (centroid_x, centroid_y),
        'threshold': threshold
    }


def extract_gradcam_features(heatmap: np.ndarray) -> Dict[str, float]:
    """
    Extrait des features quantitatives d'une heatmap GradCAM
    
    Args:
        heatmap: Heatmap GradCAM normalisée
        
    Returns:
        Features extraites
    """
    # Statistiques de base
    mean_activation = np.mean(heatmap)
    std_activation = np.std(heatmap)
    max_activation = np.max(heatmap)
    min_activation = np.min(heatmap)
    
    # Percentiles
    p25 = np.percentile(heatmap, 25)
    p50 = np.percentile(heatmap, 50)
    p75 = np.percentile(heatmap, 75)
    p90 = np.percentile(heatmap, 90)
    p95 = np.percentile(heatmap, 95)
    
    # Entropie
    hist, _ = np.histogram(heatmap, bins=50, range=(0, 1))
    hist = hist / np.sum(hist)  # Normalisation
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    
    # Concentration (pourcentage de pixels au-dessus de différents seuils)
    conc_50 = np.mean(heatmap > 0.5)
    conc_75 = np.mean(heatmap > 0.75)
    conc_90 = np.mean(heatmap > 0.9)
    
    return {
        'mean_activation': mean_activation,
        'std_activation': std_activation,
        'max_activation': max_activation,
        'min_activation': min_activation,
        'p25': p25,
        'p50': p50,
        'p75': p75,
        'p90': p90,
        'p95': p95,
        'entropy': entropy,
        'concentration_50': conc_50,
        'concentration_75': conc_75,
        'concentration_90': conc_90
    }
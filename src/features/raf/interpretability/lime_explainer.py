# =================================
# LIME EXPLAINER MODULE
# =================================
"""
Module LIME pour expliquer les prédictions de modèles image
LIME (Local Interpretable Model-agnostic Explanations)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
import cv2
from PIL import Image

try:
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    LIME_AVAILABLE = True
except ImportError:
    print("⚠️ LIME non disponible. Installer avec: pip install lime")
    LIME_AVAILABLE = False

class LIMEExplainer:
    """
    Explainer LIME pour modèles de classification d'images
    """
    
    def __init__(self, model, preprocess_fn=None):
        """
        Initialise l'explainer LIME
        
        Args:
            model: Modèle à expliquer (doit avoir une méthode predict ou __call__)
            preprocess_fn: Fonction de préprocessing des images
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME n'est pas disponible. Installer avec: pip install lime")
            
        self.model = model
        self.preprocess_fn = preprocess_fn or self._default_preprocess
        
        # Configuration LIME
        self.explainer = lime_image.LimeImageExplainer()
        
        print("✅ LIME explainer initialisé")
    
    def _default_preprocess(self, images):
        """Préprocessing par défaut"""
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        return images.astype(np.float32) / 255.0
    
    def _predict_fn(self, images):
        """
        Fonction de prédiction adaptée pour LIME
        
        Args:
            images: Batch d'images à prédire
            
        Returns:
            Array des probabilités de classe
        """
        processed_images = self.preprocess_fn(images)
        
        # Gérer différents types de modèles
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(processed_images, verbose=0)
        else:
            predictions = self.model(processed_images)
        
        # Convertir en numpy si nécessaire
        if hasattr(predictions, 'numpy'):
            predictions = predictions.numpy()
        
        return predictions
    
    def explain_image(self, 
                     image: np.ndarray,
                     top_labels: int = 5,
                     num_features: int = 100000,
                     num_samples: int = 1000,
                     segmentation_fn=None) -> Dict[str, Any]:
        """
        Génère une explication LIME pour une image
        
        Args:
            image: Image à expliquer (H, W, C)
            top_labels: Nombre de classes à expliquer
            num_features: Nombre de features/superpixels
            num_samples: Nombre d'échantillons pour l'approximation
            segmentation_fn: Fonction de segmentation personnalisée
            
        Returns:
            Dict contenant les résultats de l'explication
        """
        if image.shape[-1] == 1:
            # Convertir grayscale en RGB pour LIME
            image = np.repeat(image, 3, axis=-1)
        
        # Prédiction originale
        original_pred = self._predict_fn(np.expand_dims(image, axis=0))[0]
        predicted_class = np.argmax(original_pred)
        confidence = original_pred[predicted_class]
        
        print(f"🔍 Explication LIME...")
        print(f"• Classe prédite: {predicted_class}")
        print(f"• Confiance: {confidence:.2%}")
        print(f"• Échantillons LIME: {num_samples}")
        
        # Configuration de la segmentation
        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                   max_dist=200, ratio=0.2,
                                                   random_seed=42)
        
        # Génération de l'explication
        explanation = self.explainer.explain_instance(
            image.astype('double'),
            self._predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=segmentation_fn
        )
        
        results = {
            'explanation': explanation,
            'original_image': image,
            'predicted_class': predicted_class,
            'prediction_confidence': confidence,
            'prediction_probabilities': original_pred,
            'segments': explanation.segments
        }
        
        return results
    
    def plot_lime_explanation(self, 
                             results: Dict[str, Any],
                             positive_only: bool = True,
                             negative_only: bool = False,
                             hide_rest: bool = False,
                             num_features: int = 5,
                             min_weight: float = 0.01,
                             title: Optional[str] = None,
                             class_names: Optional[List[str]] = None) -> plt.Figure:
        """
        Visualise l'explication LIME
        
        Args:
            results: Résultats de explain_image
            positive_only: Afficher seulement les features positives
            negative_only: Afficher seulement les features négatives
            hide_rest: Masquer le reste de l'image
            num_features: Nombre de features à montrer
            min_weight: Poids minimum des features
            title: Titre du graphique
            class_names: Noms des classes
            
        Returns:
            Figure matplotlib
        """
        explanation = results['explanation']
        original_image = results['original_image']
        predicted_class = results['predicted_class']
        confidence = results['prediction_confidence']
        
        # Titre par défaut
        if title is None:
            class_name = class_names[predicted_class] if class_names else f"Classe {predicted_class}"
            title = f"Explication LIME - {class_name} ({confidence:.1%})"
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Image originale
        axes[0].imshow(original_image.squeeze(), cmap='gray' if original_image.shape[-1] == 1 else None)
        axes[0].set_title('Image Originale')
        axes[0].axis('off')
        
        # Explication avec features positives et négatives
        temp, mask = explanation.get_image_and_mask(
            predicted_class, 
            positive_only=False, 
            negative_only=False,
            hide_rest=hide_rest,
            num_features=num_features,
            min_weight=min_weight
        )
        axes[1].imshow(temp)
        axes[1].set_title(f'Features Importantes\n(Top {num_features})')
        axes[1].axis('off')
        
        # Explication avec seulement features positives
        temp_pos, mask_pos = explanation.get_image_and_mask(
            predicted_class,
            positive_only=True,
            negative_only=False,
            hide_rest=hide_rest,
            num_features=num_features,
            min_weight=min_weight
        )
        axes[2].imshow(temp_pos)
        axes[2].set_title(f'Features Positives\n(Soutiennent la prédiction)')
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def analyze_lime_features(self, results: Dict[str, Any], 
                             class_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyse quantitative des features LIME
        
        Args:
            results: Résultats de explain_image
            class_idx: Index de la classe à analyser (par défaut: classe prédite)
            
        Returns:
            Dict avec les statistiques des features
        """
        explanation = results['explanation']
        
        if class_idx is None:
            class_idx = results['predicted_class']
        
        # Récupérer les features et leurs poids
        features_weights = explanation.local_exp[class_idx]
        features_weights.sort(key=lambda x: abs(x[1]), reverse=True)
        
        positive_features = [fw for fw in features_weights if fw[1] > 0]
        negative_features = [fw for fw in features_weights if fw[1] < 0]
        
        total_positive_weight = sum(abs(fw[1]) for fw in positive_features)
        total_negative_weight = sum(abs(fw[1]) for fw in negative_features)
        total_weight = total_positive_weight + total_negative_weight
        
        analysis = {
            'num_features': len(features_weights),
            'positive_features': len(positive_features),
            'negative_features': len(negative_features),
            'total_positive_weight': total_positive_weight,
            'total_negative_weight': total_negative_weight,
            'total_weight': total_weight,
            'positive_ratio': total_positive_weight / total_weight if total_weight > 0 else 0,
            'top_positive_features': positive_features[:5],
            'top_negative_features': negative_features[:5],
            'feature_weights': features_weights
        }
        
        return analysis
    
    def compare_lime_explanations(self, 
                                 images: List[np.ndarray],
                                 labels: Optional[List[str]] = None,
                                 num_samples: int = 1000) -> Dict[str, Any]:
        """
        Compare les explications LIME pour plusieurs images
        
        Args:
            images: Liste d'images à comparer
            labels: Labels optionnels pour les images
            num_samples: Nombre d'échantillons LIME
            
        Returns:
            Dict avec les résultats comparatifs
        """
        print(f"🔍 Comparaison LIME de {len(images)} images...")
        
        explanations = []
        analyses = []
        
        for i, image in enumerate(images):
            try:
                # Explication LIME
                result = self.explain_image(image, num_samples=num_samples)
                analysis = self.analyze_lime_features(result)
                
                explanations.append(result)
                analyses.append(analysis)
                
                label = labels[i] if labels else f"Image {i+1}"
                print(f"✅ {label}: {analysis['positive_features']} features positives, "
                      f"{analysis['negative_features']} négatives")
                
            except Exception as e:
                print(f"❌ Erreur image {i}: {e}")
                explanations.append(None)
                analyses.append(None)
        
        return {
            'explanations': explanations,
            'analyses': analyses,
            'labels': labels or [f"Image {i+1}" for i in range(len(images))]
        }


def create_lime_explainer(model, preprocess_fn=None):
    """
    Factory function pour créer un explainer LIME
    
    Args:
        model: Modèle à expliquer
        preprocess_fn: Fonction de préprocessing
        
    Returns:
        LIMEExplainer instance
    """
    return LIMEExplainer(model, preprocess_fn)


def visualize_lime_comparison(comparison_results: Dict[str, Any], 
                             class_names: Optional[List[str]] = None) -> plt.Figure:
    """
    Visualise une comparaison LIME entre plusieurs images
    
    Args:
        comparison_results: Résultats de compare_lime_explanations
        class_names: Noms des classes
        
    Returns:
        Figure matplotlib
    """
    explanations = comparison_results['explanations']
    labels = comparison_results['labels']
    
    valid_explanations = [(exp, label) for exp, label in zip(explanations, labels) if exp is not None]
    
    if not valid_explanations:
        print("❌ Aucune explication valide pour la visualisation")
        return None
    
    n_images = len(valid_explanations)
    fig, axes = plt.subplots(2, n_images, figsize=(6*n_images, 12))
    
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    for i, (explanation, label) in enumerate(valid_explanations):
        predicted_class = explanation['predicted_class']
        confidence = explanation['prediction_confidence']
        
        # Image originale
        axes[0, i].imshow(explanation['original_image'].squeeze(), 
                         cmap='gray' if explanation['original_image'].shape[-1] == 1 else None)
        class_name = class_names[predicted_class] if class_names else f"Classe {predicted_class}"
        axes[0, i].set_title(f'{label}\n{class_name} ({confidence:.1%})')
        axes[0, i].axis('off')
        
        # Explication LIME
        lime_exp = explanation['explanation']
        temp, mask = lime_exp.get_image_and_mask(
            predicted_class,
            positive_only=False,
            negative_only=False,
            hide_rest=False,
            num_features=10
        )
        axes[1, i].imshow(temp)
        axes[1, i].set_title('Features LIME')
        axes[1, i].axis('off')
    
    plt.suptitle('Comparaison des Explications LIME', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig
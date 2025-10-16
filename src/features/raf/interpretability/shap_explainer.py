# =================================
# SHAP EXPLAINER - RAF
# =================================
"""
Module SHAP pour l'explication des prédictions de modèles
Supporte les modèles de machine learning classiques et les réseaux de neurones
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP non installé. Utilisez: pip install shap")

from sklearn.base import BaseEstimator
import tensorflow as tf
from tensorflow import keras


class SHAPExplainer:
    """
    Classe principale pour les explications SHAP
    """
    
    def __init__(self, model: Any, model_type: str = 'auto'):
        """
        Initialise l'explainer SHAP
        
        Args:
            model: Modèle entraîné (sklearn, tensorflow, etc.)
            model_type: Type de modèle ('tree', 'linear', 'deep', 'kernel', 'auto')
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP n'est pas installé")
            
        self.model = model
        self.model_type = self._detect_model_type(model) if model_type == 'auto' else model_type
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        
    def _detect_model_type(self, model) -> str:
        """Détecte automatiquement le type de modèle"""
        model_name = type(model).__name__.lower()
        
        if any(tree_type in model_name for tree_type in ['tree', 'forest', 'gradient', 'xgb', 'lgb']):
            return 'tree'
        elif any(linear_type in model_name for linear_type in ['linear', 'logistic', 'svm']):
            return 'linear'
        elif hasattr(model, 'layers') or 'neural' in model_name or 'mlp' in model_name:
            return 'deep'
        else:
            return 'kernel'
    
    def fit_explainer(self, X_background: np.ndarray, **kwargs):
        """
        Ajuste l'explainer SHAP sur les données d'arrière-plan
        
        Args:
            X_background: Données d'arrière-plan pour l'explainer
            **kwargs: Arguments additionnels pour l'explainer
        """
        print(f"🔧 Initialisation de l'explainer SHAP ({self.model_type})")
        
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model, **kwargs)
        elif self.model_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, X_background, **kwargs)
        elif self.model_type == 'deep':
            self.explainer = shap.DeepExplainer(self.model, X_background, **kwargs)
        else:  # kernel
            self.explainer = shap.KernelExplainer(self.model.predict, X_background, **kwargs)
            
        print("✅ Explainer SHAP initialisé")
    
    def explain(self, X: np.ndarray, max_evals: int = 100) -> np.ndarray:
        """
        Calcule les valeurs SHAP pour les échantillons donnés
        
        Args:
            X: Échantillons à expliquer
            max_evals: Nombre maximum d'évaluations (pour KernelExplainer)
            
        Returns:
            Valeurs SHAP
        """
        if self.explainer is None:
            raise ValueError("L'explainer doit être ajusté avec fit_explainer() d'abord")
        
        print(f"🔍 Calcul des valeurs SHAP pour {X.shape[0]} échantillons")
        
        if self.model_type == 'kernel':
            self.shap_values = self.explainer.shap_values(X, nsamples=max_evals)
        else:
            self.shap_values = self.explainer.shap_values(X)
            
        # Récupération de la valeur attendue
        if hasattr(self.explainer, 'expected_value'):
            self.expected_value = self.explainer.expected_value
        
        print("✅ Valeurs SHAP calculées")
        return self.shap_values
    
    def plot_waterfall(self, instance_idx: int = 0, class_idx: Optional[int] = None, 
                      feature_names: Optional[List[str]] = None, show: bool = True) -> plt.Figure:
        """
        Crée un graphique en cascade (waterfall) pour une prédiction
        
        Args:
            instance_idx: Index de l'instance à expliquer
            class_idx: Index de la classe (pour classification multi-classe)
            feature_names: Noms des features
            show: Afficher le graphique
            
        Returns:
            Figure matplotlib
        """
        if self.shap_values is None:
            raise ValueError("Les valeurs SHAP doivent être calculées d'abord")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Gestion des modèles multi-classes
        if isinstance(self.shap_values, list) and class_idx is not None:
            shap_vals = self.shap_values[class_idx][instance_idx]
            expected_val = self.expected_value[class_idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
        elif isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0][instance_idx]
            expected_val = self.expected_value[0] if isinstance(self.expected_value, np.ndarray) else self.expected_value
        else:
            shap_vals = self.shap_values[instance_idx]
            expected_val = self.expected_value
        
        try:
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_vals,
                    base_values=expected_val,
                    feature_names=feature_names
                ),
                show=show
            )
        except:
            # Fallback vers l'ancienne API
            shap.waterfall_plot(
                expected_val,
                shap_vals,
                feature_names=feature_names,
                show=show
            )
        
        if not show:
            plt.close()
        
        return fig
    
    def plot_summary(self, class_idx: Optional[int] = None, feature_names: Optional[List[str]] = None,
                    plot_type: str = 'dot', max_display: int = 20, show: bool = True) -> plt.Figure:
        """
        Crée un graphique de résumé des valeurs SHAP
        
        Args:
            class_idx: Index de la classe
            feature_names: Noms des features
            plot_type: Type de graphique ('dot', 'bar', 'violin')
            max_display: Nombre maximum de features à afficher
            show: Afficher le graphique
            
        Returns:
            Figure matplotlib
        """
        if self.shap_values is None:
            raise ValueError("Les valeurs SHAP doivent être calculées d'abord")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Gestion des modèles multi-classes
        if isinstance(self.shap_values, list) and class_idx is not None:
            shap_vals = self.shap_values[class_idx]
        elif isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        shap.summary_plot(
            shap_vals,
            feature_names=feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=show
        )
        
        if not show:
            plt.close()
        
        return fig
    
    def plot_force(self, instance_idx: int = 0, class_idx: Optional[int] = None,
                  feature_names: Optional[List[str]] = None, matplotlib: bool = True) -> Any:
        """
        Crée un graphique de force pour une prédiction
        
        Args:
            instance_idx: Index de l'instance
            class_idx: Index de la classe
            feature_names: Noms des features
            matplotlib: Utiliser matplotlib au lieu de l'affichage interactif
            
        Returns:
            Graphique de force
        """
        if self.shap_values is None:
            raise ValueError("Les valeurs SHAP doivent être calculées d'abord")
        
        # Gestion des modèles multi-classes
        if isinstance(self.shap_values, list) and class_idx is not None:
            shap_vals = self.shap_values[class_idx][instance_idx]
            expected_val = self.expected_value[class_idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
        elif isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0][instance_idx]
            expected_val = self.expected_value[0] if isinstance(self.expected_value, np.ndarray) else self.expected_value
        else:
            shap_vals = self.shap_values[instance_idx]
            expected_val = self.expected_value
        
        if matplotlib:
            return shap.force_plot(
                expected_val,
                shap_vals,
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
        else:
            return shap.force_plot(
                expected_val,
                shap_vals,
                feature_names=feature_names
            )
    
    def get_feature_importance(self, class_idx: Optional[int] = None) -> pd.DataFrame:
        """
        Calcule l'importance moyenne des features
        
        Args:
            class_idx: Index de la classe
            
        Returns:
            DataFrame avec l'importance des features
        """
        if self.shap_values is None:
            raise ValueError("Les valeurs SHAP doivent être calculées d'abord")
        
        # Gestion des modèles multi-classes
        if isinstance(self.shap_values, list) and class_idx is not None:
            shap_vals = self.shap_values[class_idx]
        elif isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        # Calcul de l'importance moyenne (valeur absolue)
        importance = np.mean(np.abs(shap_vals), axis=0)
        
        feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df_importance


def explain_prediction(model: Any, X_background: np.ndarray, X_explain: np.ndarray,
                      model_type: str = 'auto', feature_names: Optional[List[str]] = None,
                      instance_idx: int = 0, max_evals: int = 100) -> Dict[str, Any]:
    """
    Fonction helper pour expliquer rapidement une prédiction
    
    Args:
        model: Modèle entraîné
        X_background: Données d'arrière-plan
        X_explain: Données à expliquer
        model_type: Type de modèle
        feature_names: Noms des features
        instance_idx: Index de l'instance à expliquer
        max_evals: Nombre maximum d'évaluations
        
    Returns:
        Dictionnaire contenant les explications
    """
    explainer = SHAPExplainer(model, model_type)
    explainer.fit_explainer(X_background)
    shap_values = explainer.explain(X_explain, max_evals)
    
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'waterfall_fig': explainer.plot_waterfall(instance_idx, feature_names=feature_names, show=False),
        'summary_fig': explainer.plot_summary(feature_names=feature_names, show=False),
        'feature_importance': explainer.get_feature_importance()
    }


def visualize_shap_values(shap_values: np.ndarray, feature_names: Optional[List[str]] = None,
                         instance_idx: int = 0, plot_type: str = 'all') -> Dict[str, plt.Figure]:
    """
    Visualise les valeurs SHAP avec différents types de graphiques
    
    Args:
        shap_values: Valeurs SHAP calculées
        feature_names: Noms des features
        instance_idx: Index de l'instance
        plot_type: Type de graphique ('waterfall', 'summary', 'force', 'all')
        
    Returns:
        Dictionnaire des figures
    """
    figures = {}
    
    if plot_type in ['waterfall', 'all']:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Implémentation simplifiée du waterfall
        if len(shap_values.shape) == 2:
            values = shap_values[instance_idx]
        else:
            values = shap_values
            
        # Tri par importance
        indices = np.argsort(np.abs(values))[::-1][:15]  # Top 15
        
        colors = ['red' if v > 0 else 'blue' for v in values[indices]]
        names = [feature_names[i] if feature_names else f'Feature_{i}' for i in indices]
        
        plt.barh(range(len(indices)), values[indices], color=colors, alpha=0.7)
        plt.yticks(range(len(indices)), names)
        plt.xlabel('Valeur SHAP')
        plt.title('Contribution des features à la prédiction')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        figures['waterfall'] = fig
        plt.close(fig)
    
    if plot_type in ['summary', 'all']:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Summary plot simplifié
        if len(shap_values.shape) == 2:
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        else:
            mean_abs_shap = np.abs(shap_values)
            
        indices = np.argsort(mean_abs_shap)[::-1][:20]  # Top 20
        names = [feature_names[i] if feature_names else f'Feature_{i}' for i in indices]
        
        plt.barh(range(len(indices)), mean_abs_shap[indices], alpha=0.7)
        plt.yticks(range(len(indices)), names)
        plt.xlabel('Importance moyenne (|SHAP|)')
        plt.title('Importance globale des features')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        figures['summary'] = fig
        plt.close(fig)
    
    return figures


def compare_explanations(models: Dict[str, Any], X_background: np.ndarray, 
                        X_explain: np.ndarray, instance_idx: int = 0,
                        feature_names: Optional[List[str]] = None) -> plt.Figure:
    """
    Compare les explications SHAP de plusieurs modèles
    
    Args:
        models: Dictionnaire des modèles {nom: modèle}
        X_background: Données d'arrière-plan
        X_explain: Données à expliquer
        instance_idx: Index de l'instance
        feature_names: Noms des features
        
    Returns:
        Figure de comparaison
    """
    explanations = {}
    
    for name, model in models.items():
        try:
            explainer = SHAPExplainer(model)
            explainer.fit_explainer(X_background)
            shap_values = explainer.explain(X_explain)
            
            if len(shap_values.shape) == 2:
                explanations[name] = shap_values[instance_idx]
            else:
                explanations[name] = shap_values
                
        except Exception as e:
            print(f"Erreur avec le modèle {name}: {e}")
            continue
    
    if not explanations:
        raise ValueError("Aucune explication n'a pu être générée")
    
    # Création du graphique de comparaison
    n_models = len(explanations)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, shap_vals) in enumerate(explanations.items()):
        ax = axes[idx]
        
        # Top features par importance
        indices = np.argsort(np.abs(shap_vals))[::-1][:15]
        colors = ['red' if v > 0 else 'blue' for v in shap_vals[indices]]
        names = [feature_names[i] if feature_names else f'Feature_{i}' for i in indices]
        
        ax.barh(range(len(indices)), shap_vals[indices], color=colors, alpha=0.7)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Valeur SHAP')
        ax.set_title(f'Explications SHAP - {name}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
# =================================
# INTERPRETABILITY UTILITIES - RAF
# =================================
"""
Utilitaires pour l'interprétabilité
Fonctions helpers et visualisations combinées
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from .shap_explainer import SHAPExplainer, explain_prediction, visualize_shap_values
from .gradcam_explainer import GradCAMExplainer, generate_gradcam, visualize_gradcam_comparison


class InterpretabilityAnalyzer:
    """
    Classe principale pour l'analyse d'interprétabilité combinant SHAP et GradCAM
    """
    
    def __init__(self, cnn_model: Any = None, ml_model: Any = None, 
                 layer_name: Optional[str] = None):
        """
        Initialise l'analyseur d'interprétabilité
        
        Args:
            cnn_model: Modèle CNN pour GradCAM
            ml_model: Modèle ML pour SHAP
            layer_name: Couche pour GradCAM
        """
        self.cnn_model = cnn_model
        self.ml_model = ml_model
        
        # Initialisation des explainers
        self.shap_explainer = None
        self.gradcam_explainer = None
        
        if cnn_model is not None:
            try:
                self.gradcam_explainer = GradCAMExplainer(cnn_model, layer_name)
                print("✅ GradCAM explainer initialisé")
            except Exception as e:
                print(f"⚠️ Erreur initialisation GradCAM: {e}")
        
        if ml_model is not None:
            try:
                self.shap_explainer = SHAPExplainer(ml_model)
                print("✅ SHAP explainer initialisé")
            except Exception as e:
                print(f"⚠️ Erreur initialisation SHAP: {e}")
    
    def analyze_cnn_prediction(self, img: np.ndarray, class_idx: Optional[int] = None,
                              class_names: Optional[List[str]] = None,
                              methods: List[str] = ['gradcam', 'gradcam++']) -> Dict[str, Any]:
        """
        Analyse complète d'une prédiction CNN
        
        Args:
            img: Image à analyser
            class_idx: Index de la classe cible
            class_names: Noms des classes
            methods: Méthodes à utiliser
            
        Returns:
            Résultats d'analyse
        """
        if self.gradcam_explainer is None:
            raise ValueError("Aucun modèle CNN configuré")
        
        results = {}
        
        for method in methods:
            try:
                if method == 'gradcam':
                    results[method] = self.gradcam_explainer.generate_gradcam(img, class_idx)
                elif method == 'gradcam++':
                    results[method] = self.gradcam_explainer.generate_gradcam_plus_plus(img, class_idx)
                print(f"✅ {method} terminé")
            except Exception as e:
                print(f"❌ Erreur {method}: {e}")
                continue
        
        # Génération du graphique comparatif
        if results:
            comparison_fig = visualize_gradcam_comparison(results, 
                                                        "Comparaison des méthodes GradCAM",
                                                        class_names)
            results['comparison_figure'] = comparison_fig
        
        return results
    
    def analyze_ml_prediction(self, X_background: np.ndarray, X_explain: np.ndarray,
                             feature_names: Optional[List[str]] = None,
                             instance_idx: int = 0) -> Dict[str, Any]:
        """
        Analyse complète d'une prédiction ML avec SHAP
        
        Args:
            X_background: Données d'arrière-plan
            X_explain: Données à expliquer
            feature_names: Noms des features
            instance_idx: Index de l'instance
            
        Returns:
            Résultats d'analyse SHAP
        """
        if self.shap_explainer is None:
            raise ValueError("Aucun modèle ML configuré")
        
        try:
            # Ajustement de l'explainer
            self.shap_explainer.fit_explainer(X_background)
            
            # Calcul des valeurs SHAP
            shap_values = self.shap_explainer.explain(X_explain)
            
            # Génération des visualisations
            results = {
                'shap_values': shap_values,
                'waterfall_fig': self.shap_explainer.plot_waterfall(instance_idx, 
                                                                   feature_names=feature_names, 
                                                                   show=False),
                'summary_fig': self.shap_explainer.plot_summary(feature_names=feature_names, 
                                                               show=False),
                'feature_importance': self.shap_explainer.get_feature_importance()
            }
            
            print("✅ Analyse SHAP terminée")
            return results
            
        except Exception as e:
            print(f"❌ Erreur analyse SHAP: {e}")
            return {}
    
    def compare_predictions(self, img: np.ndarray, X_flat: np.ndarray,
                           X_background: np.ndarray, class_names: Optional[List[str]] = None,
                           feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare les explications CNN (GradCAM) et ML (SHAP) pour la même image
        
        Args:
            img: Image pour CNN
            X_flat: Image aplatie pour ML
            X_background: Données d'arrière-plan pour SHAP
            class_names: Noms des classes
            feature_names: Noms des features
            
        Returns:
            Comparaison des explications
        """
        results = {}
        
        # Analyse CNN
        if self.gradcam_explainer is not None:
            try:
                cnn_results = self.analyze_cnn_prediction(img, class_names=class_names)
                results['cnn'] = cnn_results
            except Exception as e:
                print(f"❌ Erreur analyse CNN: {e}")
        
        # Analyse ML
        if self.shap_explainer is not None:
            try:
                X_explain = X_flat.reshape(1, -1) if len(X_flat.shape) == 1 else X_flat
                ml_results = self.analyze_ml_prediction(X_background, X_explain, feature_names)
                results['ml'] = ml_results
            except Exception as e:
                print(f"❌ Erreur analyse ML: {e}")
        
        # Génération d'un rapport comparatif
        if 'cnn' in results and 'ml' in results:
            results['comparison_report'] = self._generate_comparison_report(results)
        
        return results
    
    def _generate_comparison_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère un rapport de comparaison entre CNN et ML
        
        Args:
            results: Résultats des analyses
            
        Returns:
            Rapport de comparaison
        """
        report = {
            'cnn_confidence': None,
            'ml_confidence': None,
            'cnn_predicted_class': None,
            'ml_predicted_class': None,
            'agreement': False,
            'summary': ""
        }
        
        # Extraction des informations CNN
        if 'cnn' in results and 'gradcam' in results['cnn']:
            gradcam_result = results['cnn']['gradcam']
            report['cnn_confidence'] = gradcam_result.get('prediction_confidence')
            report['cnn_predicted_class'] = gradcam_result.get('predicted_class')
        
        # Pour le ML, nous aurions besoin de faire une prédiction
        # Ceci nécessiterait d'adapter selon le modèle ML utilisé
        
        # Génération du résumé
        if report['cnn_confidence'] is not None:
            confidence_level = "élevée" if report['cnn_confidence'] > 0.8 else \
                             "moyenne" if report['cnn_confidence'] > 0.6 else "faible"
            report['summary'] = f"Confiance CNN: {confidence_level} ({report['cnn_confidence']:.2%})"
        
        return report


def create_interpretability_dashboard(analyzer: InterpretabilityAnalyzer,
                                    img: np.ndarray, X_flat: np.ndarray,
                                    X_background: np.ndarray,
                                    class_names: Optional[List[str]] = None,
                                    feature_names: Optional[List[str]] = None,
                                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Crée un tableau de bord complet d'interprétabilité
    
    Args:
        analyzer: Analyseur d'interprétabilité
        img: Image pour CNN
        X_flat: Image aplatie pour ML
        X_background: Données d'arrière-plan
        class_names: Noms des classes
        feature_names: Noms des features
        save_path: Chemin de sauvegarde
        
    Returns:
        Figure du tableau de bord
    """
    # Analyse complète
    results = analyzer.compare_predictions(img, X_flat, X_background, class_names, feature_names)
    
    # Création du tableau de bord
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Titre principal
    fig.suptitle('Tableau de Bord d\'Interprétabilité - COVID-19', fontsize=16, fontweight='bold')
    
    # Section GradCAM (si disponible)
    if 'cnn' in results and 'gradcam' in results['cnn']:
        gradcam_result = results['cnn']['gradcam']
        
        # Image originale
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(gradcam_result['original'])
        ax1.set_title('Image Originale')
        ax1.axis('off')
        
        # Heatmap GradCAM
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(gradcam_result['heatmap'], cmap='jet')
        ax2.set_title('Heatmap GradCAM')
        ax2.axis('off')
        
        # Superposition
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(gradcam_result['overlay'])
        
        predicted_class = gradcam_result['predicted_class']
        confidence = gradcam_result['prediction_confidence']
        
        if class_names and predicted_class < len(class_names):
            class_name = class_names[predicted_class]
        else:
            class_name = f"Classe {predicted_class}"
            
        ax3.set_title(f'Superposition\n{class_name} ({confidence:.2%})')
        ax3.axis('off')
        
        # Statistiques GradCAM
        ax4 = fig.add_subplot(gs[0, 3])
        from .gradcam_explainer import extract_gradcam_features
        features = extract_gradcam_features(gradcam_result['heatmap'])
        
        stats_text = f"""Statistiques GradCAM:
        
Activation moyenne: {features['mean_activation']:.3f}
Activation max: {features['max_activation']:.3f}
Écart-type: {features['std_activation']:.3f}
Entropie: {features['entropy']:.3f}

Concentration:
• >50%: {features['concentration_50']:.1%}
• >75%: {features['concentration_75']:.1%}
• >90%: {features['concentration_90']:.1%}"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax4.axis('off')
    
    # Section SHAP (si disponible)
    if 'ml' in results and 'shap_values' in results['ml']:
        # Graphique de features importance
        ax5 = fig.add_subplot(gs[1, :2])
        importance_df = results['ml']['feature_importance'].head(15)
        
        bars = ax5.barh(range(len(importance_df)), importance_df['importance'], alpha=0.7)
        ax5.set_yticks(range(len(importance_df)))
        ax5.set_yticklabels([f'Feature_{i}' for i in range(len(importance_df))])
        ax5.set_xlabel('Importance SHAP')
        ax5.set_title('Top 15 Features - Importance SHAP')
        ax5.grid(True, alpha=0.3)
        
        # Statistiques SHAP
        ax6 = fig.add_subplot(gs[1, 2:])
        shap_values = results['ml']['shap_values']
        
        if isinstance(shap_values, list):
            shap_array = shap_values[0]
        else:
            shap_array = shap_values
            
        if len(shap_array.shape) > 1:
            shap_stats = shap_array[0]  # Premier échantillon
        else:
            shap_stats = shap_array
        
        stats_text = f"""Statistiques SHAP:
        
Nombre de features: {len(shap_stats)}
Valeur SHAP moyenne: {np.mean(shap_stats):.4f}
Valeur SHAP max: {np.max(shap_stats):.4f}
Valeur SHAP min: {np.min(shap_stats):.4f}
Écart-type: {np.std(shap_stats):.4f}

Features positives: {np.sum(shap_stats > 0)}
Features négatives: {np.sum(shap_stats < 0)}
Impact total: {np.sum(np.abs(shap_stats)):.4f}"""
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax6.axis('off')
    
    # Section de comparaison
    ax7 = fig.add_subplot(gs[2, :])
    
    if 'comparison_report' in results:
        report = results['comparison_report']
        comparison_text = f"""RAPPORT DE COMPARAISON

Modèle CNN (GradCAM):
• Classe prédite: {report.get('cnn_predicted_class', 'N/A')}
• Confiance: {report.get('cnn_confidence', 0):.2%}

Résumé: {report.get('summary', 'Analyse en cours...')}

RECOMMANDATIONS:
• La région d'attention GradCAM indique les zones importantes pour la classification
• Les valeurs SHAP montrent l'impact de chaque pixel/feature sur la décision
• Une cohérence entre les deux méthodes renforce la confiance dans la prédiction"""
    else:
        comparison_text = """ANALYSE D'INTERPRÉTABILITÉ EN COURS...
        
Veuillez vous assurer que les deux modèles (CNN et ML) sont correctement configurés
pour obtenir une analyse comparative complète."""
    
    ax7.text(0.05, 0.95, comparison_text, transform=ax7.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    ax7.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_interpretability_report(analyzer: InterpretabilityAnalyzer,
                                   images: List[np.ndarray],
                                   X_background: np.ndarray,
                                   class_names: Optional[List[str]] = None,
                                   sample_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Génère un rapport d'interprétabilité pour plusieurs échantillons
    
    Args:
        analyzer: Analyseur d'interprétabilité
        images: Liste d'images à analyser
        X_background: Données d'arrière-plan
        class_names: Noms des classes
        sample_names: Noms des échantillons
        
    Returns:
        DataFrame avec le rapport
    """
    report_data = []
    
    for idx, img in enumerate(images):
        sample_name = sample_names[idx] if sample_names else f"Sample_{idx+1}"
        
        try:
            # Analyse GradCAM
            if analyzer.gradcam_explainer is not None:
                gradcam_result = analyzer.gradcam_explainer.generate_gradcam(img)
                
                # Extraction des features GradCAM
                from .gradcam_explainer import extract_gradcam_features
                gradcam_features = extract_gradcam_features(gradcam_result['heatmap'])
                
                # Ajout au rapport
                row_data = {
                    'sample': sample_name,
                    'predicted_class': gradcam_result['predicted_class'],
                    'confidence': gradcam_result['prediction_confidence'],
                    'gradcam_mean_activation': gradcam_features['mean_activation'],
                    'gradcam_max_activation': gradcam_features['max_activation'],
                    'gradcam_entropy': gradcam_features['entropy'],
                    'gradcam_concentration_75': gradcam_features['concentration_75']
                }
                
                if class_names and gradcam_result['predicted_class'] < len(class_names):
                    row_data['predicted_class_name'] = class_names[gradcam_result['predicted_class']]
                
                report_data.append(row_data)
                
        except Exception as e:
            print(f"Erreur avec l'échantillon {sample_name}: {e}")
            continue
    
    return pd.DataFrame(report_data)


# Fonctions utilitaires additionnelles
def plot_interpretability_summary(gradcam_results: List[Dict], 
                                class_names: Optional[List[str]] = None) -> plt.Figure:
    """
    Crée un résumé visuel des résultats d'interprétabilité
    """
    n_samples = len(gradcam_results)
    fig, axes = plt.subplots(2, min(n_samples, 4), figsize=(16, 8))
    
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    elif n_samples < 4:
        axes = axes[:, :n_samples]
    
    for idx, result in enumerate(gradcam_results[:4]):  # Limite à 4 échantillons
        # Image originale
        axes[0, idx].imshow(result['original'])
        axes[0, idx].set_title(f'Échantillon {idx+1}')
        axes[0, idx].axis('off')
        
        # Superposition GradCAM
        axes[1, idx].imshow(result['overlay'])
        
        predicted_class = result['predicted_class']
        confidence = result['prediction_confidence']
        
        if class_names and predicted_class < len(class_names):
            class_name = class_names[predicted_class]
        else:
            class_name = f"Classe {predicted_class}"
            
        axes[1, idx].set_title(f'{class_name}\n({confidence:.2%})')
        axes[1, idx].axis('off')
    
    plt.suptitle('Résumé d\'Interprétabilité GradCAM', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig
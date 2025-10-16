# =================================
# INTERPRETABILITY MODULE - RAF
# =================================
"""
Module d'interprétabilité pour le framework RAF
Contient les implémentations SHAP, GradCAM et LIME pour expliquer les prédictions
"""

from .shap_explainer import (
    SHAPExplainer, 
    explain_prediction, 
    visualize_shap_values, 
    compare_explanations
)
from .gradcam_explainer import (
    GradCAMExplainer, 
    generate_gradcam, 
    visualize_gradcam_comparison,
    analyze_gradcam_regions,
    extract_gradcam_features
)
from .lime_explainer import (
    LIMEExplainer,
    create_lime_explainer,
    visualize_lime_comparison
)
from .utils import (
    InterpretabilityAnalyzer,
    create_interpretability_dashboard,
    generate_interpretability_report,
    plot_interpretability_summary
)

__all__ = [
    # SHAP
    'SHAPExplainer',
    'explain_prediction',
    'visualize_shap_values',
    'compare_explanations',
    
    # GradCAM
    'GradCAMExplainer',
    'generate_gradcam',
    'visualize_gradcam_comparison',
    'analyze_gradcam_regions',
    'extract_gradcam_features',
    
    # LIME
    'LIMEExplainer',
    'create_lime_explainer',
    'visualize_lime_comparison',
    
    # Utils
    'InterpretabilityAnalyzer',
    'create_interpretability_dashboard',
    'generate_interpretability_report',
    'plot_interpretability_summary'
]
# ===============================================
# EXEMPLE D'UTILISATION DU FICHIER .env
# ===============================================
"""
Ce script démontre comment utiliser le fichier .env 
pour configurer l'interprétabilité SHAP et GradCAM
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Chargement du fichier .env
project_root = Path(__file__).parent
env_path = project_root / '.env'

if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Configuration chargée depuis: {env_path}")
else:
    print(f"⚠️ Fichier .env non trouvé: {env_path}")

# ===============================================
# RÉCUPÉRATION DES VARIABLES D'ENVIRONNEMENT
# ===============================================

# Configuration des images
IMG_WIDTH = int(os.getenv('IMG_WIDTH', 256))
IMG_HEIGHT = int(os.getenv('IMG_HEIGHT', 256))
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# Configuration des classes
NUM_CLASSES = int(os.getenv('NUM_CLASSES', 4))
CLASS_NAMES = os.getenv('CLASS_NAMES', 'COVID,Normal,Viral,Lung_Opacity').split(',')

# Configuration d'interprétabilité
GRADCAM_ALPHA = float(os.getenv('GRADCAM_ALPHA', 0.4))
GRADCAM_COLORMAP = os.getenv('GRADCAM_COLORMAP', 'jet')
SHAP_MAX_EVALS = int(os.getenv('SHAP_MAX_EVALS', 100))
SHAP_BACKGROUND_SIZE = int(os.getenv('SHAP_BACKGROUND_SIZE', 50))

# Seuils de confiance
CONFIDENCE_HIGH = float(os.getenv('CONFIDENCE_HIGH_THRESHOLD', 0.8))
CONFIDENCE_MEDIUM = float(os.getenv('CONFIDENCE_MEDIUM_THRESHOLD', 0.6))

print("\n📊 Configuration chargée:")
print(f"• Taille d'image: {IMG_SIZE}")
print(f"• Nombre de classes: {NUM_CLASSES}")
print(f"• Classes: {CLASS_NAMES}")
print(f"• GradCAM Alpha: {GRADCAM_ALPHA}")
print(f"• GradCAM Colormap: {GRADCAM_COLORMAP}")
print(f"• SHAP Max Evals: {SHAP_MAX_EVALS}")
print(f"• SHAP Background Size: {SHAP_BACKGROUND_SIZE}")
print(f"• Seuil confiance élevée: {CONFIDENCE_HIGH}")
print(f"• Seuil confiance moyenne: {CONFIDENCE_MEDIUM}")

# ===============================================
# EXEMPLE D'UTILISATION AVEC LES MODULES
# ===============================================

def example_gradcam_usage():
    """Exemple d'utilisation de GradCAM avec configuration .env"""
    try:
        from src.features.raf.interpretability import GradCAMExplainer
        
        # Utilisation des paramètres du .env
        print(f"\n🎯 Exemple GradCAM avec config .env:")
        print(f"• Alpha: {GRADCAM_ALPHA}")
        print(f"• Colormap: {GRADCAM_COLORMAP}")
        
        # Le code suivant utiliserait un vrai modèle:
        # gradcam_explainer = GradCAMExplainer(model)
        # results = gradcam_explainer.generate_gradcam(
        #     image, 
        #     alpha=GRADCAM_ALPHA, 
        #     colormap=GRADCAM_COLORMAP
        # )
        
        print("✅ Configuration GradCAM prête")
        
    except ImportError as e:
        print(f"⚠️ Module GradCAM non disponible: {e}")

def example_shap_usage():
    """Exemple d'utilisation de SHAP avec configuration .env"""
    try:
        from src.features.raf.interpretability import SHAPExplainer
        
        # Utilisation des paramètres du .env
        print(f"\n📊 Exemple SHAP avec config .env:")
        print(f"• Max évaluations: {SHAP_MAX_EVALS}")
        print(f"• Taille background: {SHAP_BACKGROUND_SIZE}")
        
        # Le code suivant utiliserait un vrai modèle:
        # shap_explainer = SHAPExplainer(model)
        # shap_explainer.fit_explainer(X_background[:SHAP_BACKGROUND_SIZE])
        # shap_values = shap_explainer.explain(X_test, max_evals=SHAP_MAX_EVALS)
        
        print("✅ Configuration SHAP prête")
        
    except ImportError as e:
        print(f"⚠️ Module SHAP non disponible: {e}")

def example_confidence_evaluation(confidence_score):
    """Exemple d'évaluation de confiance avec seuils .env"""
    print(f"\n⚖️ Évaluation de confiance: {confidence_score:.2%}")
    
    if confidence_score > CONFIDENCE_HIGH:
        level = "Élevée"
        icon = "🔥"
    elif confidence_score > CONFIDENCE_MEDIUM:
        level = "Moyenne"
        icon = "⚡"
    else:
        level = "Faible"
        icon = "⚠️"
    
    print(f"{icon} Niveau de confiance: {level}")
    return level

if __name__ == "__main__":
    print("\n" + "="*50)
    print("DÉMONSTRATION CONFIGURATION .env")
    print("="*50)
    
    # Exemples d'utilisation
    example_gradcam_usage()
    example_shap_usage()
    
    # Test des seuils de confiance
    print("\n🧪 Test des seuils de confiance:")
    for confidence in [0.95, 0.75, 0.45]:
        example_confidence_evaluation(confidence)
    
    print(f"\n✨ Configuration .env opérationnelle !")
    print(f"📁 Fichier .env: {env_path}")
    print(f"🔧 Prêt pour l'interprétabilité SHAP & GradCAM")
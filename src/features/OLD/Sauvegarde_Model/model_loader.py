import os
import json
import joblib
import pickle
from pathlib import Path
import glob

def load_model_simple(model_path):
    """
    Charge un modèle depuis un fichier.
    
    Args:
        model_path (str): Chemin vers le fichier du modèle
    
    Returns:
        object: Modèle chargé
    """
    try:
        if model_path.endswith('.joblib'):
            model = joblib.load(model_path)
        elif model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"Format non supporté : {model_path}")
        
        print(f"[INFO] Modèle chargé : {model_path}")
        return model
    
    except Exception as e:
        print(f"[ERROR] Échec chargement : {e}")
        return None


def load_model_with_metadata(model_dir):
    """
    Charge un modèle avec ses métadonnées depuis un répertoire.
    
    Args:
        model_dir (str): Répertoire contenant le modèle et métadonnées
    
    Returns:
        dict: Dictionnaire avec model, metadata, performance
    """
    result = {'model': None, 'metadata': None, 'performance': None, 'report': None}
    
    try:
        # Charger le modèle
        model_path = os.path.join(model_dir, "model.joblib")
        if os.path.exists(model_path):
            result['model'] = joblib.load(model_path)
            print(f"[INFO] Modèle chargé depuis : {model_path}")
        else:
            print(f"[ERROR] Fichier modèle non trouvé : {model_path}")
            return result
        
        # Charger les métadonnées
        metadata_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                result['metadata'] = json.load(f)
            print(f"[INFO] Métadonnées chargées")
        
        # Charger les performances
        performance_path = os.path.join(model_dir, "performance.json")
        if os.path.exists(performance_path):
            with open(performance_path, 'r', encoding='utf-8') as f:
                result['performance'] = json.load(f)
            print(f"[INFO] Performances chargées")
        
        # Charger le rapport de classification
        report_path = os.path.join(model_dir, "classification_report.json")
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                result['report'] = json.load(f)
            print(f"[INFO] Rapport de classification chargé")
    
    except Exception as e:
        print(f"[ERROR] Échec chargement complet : {e}")
    
    return result


def find_latest_model(base_dir="models", model_name_pattern="*"):
    """
    Trouve le modèle le plus récent selon le timestamp dans le nom.
    
    Args:
        base_dir (str): Répertoire de base
        model_name_pattern (str): Pattern pour filtrer les modèles
    
    Returns:
        str: Chemin vers le répertoire du modèle le plus récent
    """
    pattern = os.path.join(base_dir, f"{model_name_pattern}_*")
    model_dirs = glob.glob(pattern)
    
    if not model_dirs:
        print(f"[WARNING] Aucun modèle trouvé avec le pattern : {pattern}")
        return None
    
    # Trier par nom (le timestamp est dans le nom)
    latest_dir = sorted(model_dirs)[-1]
    print(f"[INFO] Modèle le plus récent : {latest_dir}")
    return latest_dir


def list_available_models(base_dir="models"):
    """
    Liste tous les modèles disponibles avec leurs informations.
    
    Args:
        base_dir (str): Répertoire de base
    
    Returns:
        list: Liste des informations sur les modèles
    """
    if not os.path.exists(base_dir):
        print(f"[WARNING] Répertoire non trouvé : {base_dir}")
        return []
    
    models_info = []
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        if os.path.isdir(item_path):
            # Répertoire de modèle complet
            metadata_path = os.path.join(item_path, "metadata.json")
            model_path = os.path.join(item_path, "model.joblib")
            
            if os.path.exists(model_path):
                info = {
                    'type': 'directory',
                    'name': item,
                    'path': item_path,
                    'has_metadata': os.path.exists(metadata_path)
                }
                
                if info['has_metadata']:
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        info.update({
                            'model_name': metadata.get('model_name', 'Unknown'),
                            'timestamp': metadata.get('timestamp', 'Unknown'),
                            'model_type': metadata.get('model_type', 'Unknown')
                        })
                    except Exception as e:
                        print(f"[WARNING] Erreur lecture métadonnées {metadata_path}: {e}")
                
                models_info.append(info)
        
        elif item.endswith(('.joblib', '.pkl')):
            # Fichier modèle simple
            info = {
                'type': 'file',
                'name': item,
                'path': item_path,
                'format': 'joblib' if item.endswith('.joblib') else 'pickle'
            }
            models_info.append(info)
    
    return models_info


def predict_with_saved_model(model_path, X_new, return_proba=False):
    """
    Fait des prédictions avec un modèle sauvegardé.
    
    Args:
        model_path (str): Chemin vers le modèle ou répertoire
        X_new: Nouvelles données à prédire
        return_proba (bool): Retourner les probabilités si possible
    
    Returns:
        tuple: (predictions, probabilities) si return_proba=True, sinon predictions
    """
    # Déterminer si c'est un fichier ou un répertoire
    if os.path.isdir(model_path):
        model_file = os.path.join(model_path, "model.joblib")
    else:
        model_file = model_path
    
    # Charger le modèle
    model = load_model_simple(model_file)
    if model is None:
        return None
    
    try:
        # Prédictions
        predictions = model.predict(X_new)
        
        if return_proba:
            # Probabilités si disponibles
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_new)
                return predictions, probabilities
            elif hasattr(model, 'decision_function'):
                scores = model.decision_function(X_new)
                return predictions, scores
            else:
                return predictions, None
        
        return predictions
    
    except Exception as e:
        print(f"[ERROR] Échec prédiction : {e}")
        return None
import os
import json
import joblib
import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def get_model_filename(model_name="default", extension="joblib"):
    """
    Génère un nom de fichier avec format : model_name_YYYYMMDD_HHMMSS.extension
    
    Args:
        model_name (str): Nom du modèle (par défaut: "default")
        extension (str): Extension du fichier (par défaut: "joblib")
    
    Returns:
        str: Nom de fichier formaté
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = model_name.replace(' ', '_').replace('-', '_').lower()
    return f"model_{clean_name}_{timestamp}.{extension}"


def create_model_directory(base_dir="models", model_name="default"):
    """
    Crée un répertoire pour sauvegarder un modèle avec timestamp.
    
    Args:
        base_dir (str): Répertoire de base (par défaut: "models")
        model_name (str): Nom du modèle (par défaut: "default")
    
    Returns:
        str: Chemin du répertoire créé
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = model_name.replace(' ', '_').replace('-', '_').lower()
    model_dir = os.path.join(base_dir, f"{clean_name}_{timestamp}")
    
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    return model_dir


def save_model_simple(model, model_name="default", output_dir="models"):
    """
    Sauvegarde simple d'un modèle avec nom automatique.
    
    Args:
        model: Modèle scikit-learn entraîné
        model_name (str): Nom du modèle (par défaut: "default")
        output_dir (str): Répertoire de sortie (par défaut: "models")
    
    Returns:
        str: Chemin du fichier sauvegardé
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    filename = get_model_filename(model_name, "joblib")
    model_path = os.path.join(output_dir, filename)
    
    try:
        joblib.dump(model, model_path)
        print(f"[INFO] Modèle sauvegardé : {model_path}")
        return model_path
    except Exception as e:
        print(f"[ERROR] Échec sauvegarde : {e}")
        return None


def save_model_with_metadata(model, model_name="default", 
                           X_train=None, y_train=None, 
                           X_test=None, y_test=None, y_pred=None,
                           output_dir="models", classes=None):
    """
    Sauvegarde complète d'un modèle avec métadonnées et performances.
    
    Args:
        model: Modèle scikit-learn entraîné
        model_name (str): Nom du modèle (par défaut: "default")
        X_train, y_train: Données d'entraînement (optionnel)
        X_test, y_test, y_pred: Données de test et prédictions (optionnel)
        output_dir (str): Répertoire de sortie (par défaut: "models")
        classes (list): Liste des noms de classes (optionnel)
    
    Returns:
        dict: Informations sur les fichiers sauvegardés
    """
    # Créer le répertoire du modèle
    model_dir = create_model_directory(output_dir, model_name)
    
    saved_files = {}
    timestamp = datetime.now().isoformat()
    
    # 1. Sauvegarder le modèle
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    saved_files['model'] = model_path
    
    # 2. Sauvegarder les métadonnées
    metadata = {
        "model_name": model_name,
        "timestamp": timestamp,
        "model_type": type(model).__name__,
        "model_params": model.get_params() if hasattr(model, 'get_params') else {}
    }
    
    # Ajouter infos sur les données si disponibles
    if X_train is not None:
        metadata.update({
            "train_samples": X_train.shape[0],
            "features": X_train.shape[1] if len(X_train.shape) > 1 else len(X_train[0])
        })
    
    if X_test is not None:
        metadata["test_samples"] = X_test.shape[0]
    
    if classes is not None:
        metadata["classes"] = classes
        metadata["num_classes"] = len(classes)
    
    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    saved_files['metadata'] = metadata_path
    
    # 3. Sauvegarder les performances si disponibles
    if y_test is not None and y_pred is not None:
        performance = calculate_performance_metrics(y_test, y_pred, classes)
        
        performance_path = os.path.join(model_dir, "performance.json")
        with open(performance_path, 'w', encoding='utf-8') as f:
            json.dump(performance, f, indent=2, ensure_ascii=False)
        saved_files['performance'] = performance_path
        
        # 4. Sauvegarder le rapport de classification
        if classes:
            report = classification_report(y_test, y_pred, 
                                         target_names=classes, 
                                         output_dict=True)
            report_path = os.path.join(model_dir, "classification_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            saved_files['report'] = report_path
    
    print(f"[INFO] Modèle complet sauvegardé dans : {model_dir}")
    print(f"[INFO] Fichiers créés : {list(saved_files.keys())}")
    
    return {
        'model_dir': model_dir,
        'files': saved_files,
        'timestamp': timestamp
    }


def calculate_performance_metrics(y_test, y_pred, classes=None):
    """
    Calcule les métriques de performance détaillées.
    
    Args:
        y_test: Vraies étiquettes
        y_pred: Prédictions
        classes (list): Noms des classes (optionnel)
    
    Returns:
        dict: Métriques de performance
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    
    # Métriques globales
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    performance = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision_avg),
        "recall_weighted": float(recall_avg),
        "f1_weighted": float(f1_avg),
        "num_test_samples": len(y_test)
    }
    
    # Métriques par classe
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    
    if classes and len(classes) >= len(unique_labels):
        class_names = classes
    else:
        class_names = [f"Class_{i}" for i in unique_labels]
    
    performance["by_class"] = {}
    for i, label in enumerate(unique_labels):
        if i < len(precision):
            performance["by_class"][class_names[i]] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i])
            }
    
    return performance


def save_model_backup(model, model_name="default", output_dir="models", format="both"):
    """
    Sauvegarde de sécurité d'un modèle en multiple formats.
    
    Args:
        model: Modèle à sauvegarder
        model_name (str): Nom du modèle (par défaut: "default")
        output_dir (str): Répertoire de sortie
        format (str): Format ("joblib", "pickle", "both")
    
    Returns:
        list: Liste des fichiers créés
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_files = []
    
    base_filename = get_model_filename(model_name, "").rstrip('.')
    
    try:
        if format in ["joblib", "both"]:
            joblib_path = os.path.join(output_dir, f"{base_filename}.joblib")
            joblib.dump(model, joblib_path)
            saved_files.append(joblib_path)
            print(f"[INFO] Sauvegarde joblib : {joblib_path}")
        
        if format in ["pickle", "both"]:
            pickle_path = os.path.join(output_dir, f"{base_filename}.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(model, f)
            saved_files.append(pickle_path)
            print(f"[INFO] Sauvegarde pickle : {pickle_path}")
    
    except Exception as e:
        print(f"[ERROR] Échec sauvegarde backup : {e}")
    
    return saved_files
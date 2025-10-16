"""
Module d'évaluation et de modélisation pour le framework RAF
Gère l'entraînement, l'évaluation et l'analyse des modèles ML/DL
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, roc_curve, auc)
import warnings
from typing import Dict, List, Tuple, Any, Optional

from ..utils.config import Config


def evaluate_model(model, X_train: np.ndarray, X_val: np.ndarray, 
                  y_train: np.ndarray, y_val: np.ndarray, model_name: str):
    """
    Évalue un modèle de machine learning de manière complète
    
    Args:
        model: Modèle entraîné
        X_train, X_val: Données d'entraînement et de validation
        y_train, y_val: Labels d'entraînement et de validation
        model_name: Nom du modèle pour l'affichage
        
    Returns:
        Dict contenant les métriques d'évaluation
    """
    print(f"\n{'='*60}")
    print(f"🔍 ÉVALUATION DU MODÈLE: {model_name}")
    print(f"{'='*60}")
    
    # Prédictions
    try:
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Probabilités (si disponibles)
        try:
            y_train_proba = model.predict_proba(X_train)
            y_val_proba = model.predict_proba(X_val)
            has_proba = True
        except:
            has_proba = False
            
    except Exception as e:
        print(f"❌ Erreur lors des prédictions: {e}")
        return {}
    
    # Métriques de base
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    print(f"📊 Accuracy:")
    print(f"   Train: {train_acc:.3f}")
    print(f"   Val:   {val_acc:.3f}")
    print(f"   Écart: {abs(train_acc - val_acc):.3f}")
    
    print(f"\n📊 F1-Score (weighted):")
    print(f"   Train: {train_f1:.3f}")
    print(f"   Val:   {val_f1:.3f}")
    
    # Métriques détaillées pour la validation
    val_precision = precision_score(y_val, y_val_pred, average='weighted')
    val_recall = recall_score(y_val, y_val_pred, average='weighted')
    
    print(f"\n📊 Métriques de validation:")
    print(f"   Precision: {val_precision:.3f}")
    print(f"   Recall:    {val_recall:.3f}")
    
    # AUC si probabilités disponibles
    if has_proba and len(np.unique(y_val)) == 2:  # Binaire seulement
        try:
            val_auc = roc_auc_score(y_val, y_val_proba[:, 1])
            print(f"   AUC:       {val_auc:.3f}")
        except:
            val_auc = None
    else:
        val_auc = None
    
    # Rapport de classification
    print(f"\n📋 Rapport de classification (Validation):")
    print(classification_report(y_val, y_val_pred))
    
    # Matrice de confusion
    cm = confusion_matrix(y_val, y_val_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de confusion - {model_name}')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.show()
    
    # Détection du surapprentissage
    overfitting_score = train_acc - val_acc
    if overfitting_score > 0.1:
        print(f"⚠️ Surapprentissage détecté (écart: {overfitting_score:.3f})")
    elif overfitting_score < -0.05:
        print(f"⚠️ Sous-apprentissage possible (écart: {overfitting_score:.3f})")
    else:
        print(f"✅ Bon équilibre biais-variance")
    
    return {
        'model_name': model_name,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'val_auc': val_auc,
        'overfitting_score': overfitting_score,
        'confusion_matrix': cm
    }


def analyze_feature_importance(model, feature_names: List[str], 
                             model_name: str, top_n: int = 20):
    """
    Analyse l'importance des features pour les modèles qui le supportent
    
    Args:
        model: Modèle entraîné
        feature_names: Noms des features
        model_name: Nom du modèle
        top_n: Nombre de features les plus importantes à afficher
    """
    try:
        # Récupération de l'importance des features
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print(f"⚠️ {model_name} ne supporte pas l'analyse d'importance des features")
            return None
        
        # Création du DataFrame pour tri
        import pandas as pd
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Affichage des top features
        print(f"\n🎯 TOP {top_n} FEATURES - {model_name}")
        print("-" * 50)
        for i, (_, row) in enumerate(feature_df.head(top_n).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<30} {row['importance']:.4f}")
        
        # Visualisation
        plt.figure(figsize=(10, 8))
        top_features = feature_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Importance des features - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return feature_df
        
    except Exception as e:
        print(f"❌ Erreur analyse importance: {e}")
        return None


def extract_traditional_features(images: np.ndarray) -> np.ndarray:
    """
    Extrait des features traditionnelles d'images (histogrammes, textures, etc.)
    
    Args:
        images: Array d'images (N, H, W, C) ou (N, H, W)
        
    Returns:
        Array de features extraites
    """
    features_list = []
    
    print(f"🔍 Extraction de features sur {len(images)} images...")
    
    for i, img in enumerate(images):
        try:
            # Conversion en niveaux de gris si nécessaire
            if len(img.shape) == 3:
                gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = img
            
            # Normalisation [0, 255]
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
            
            # Features statistiques de base
            features = []
            
            # 1. Statistiques de l'histogramme
            hist = np.histogram(gray, bins=64, range=(0, 256))[0]
            hist = hist / np.sum(hist)  # Normalisation
            features.extend(hist)
            
            # 2. Moments statistiques
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            skew_val = np.mean(((gray - mean_val) / std_val) ** 3) if std_val > 0 else 0
            kurtosis_val = np.mean(((gray - mean_val) / std_val) ** 4) if std_val > 0 else 0
            
            features.extend([mean_val, std_val, skew_val, kurtosis_val])
            
            # 3. Energie et entropie
            hist_normalized = hist + 1e-10  # Éviter log(0)
            energy = np.sum(hist_normalized ** 2)
            entropy = -np.sum(hist_normalized * np.log2(hist_normalized))
            
            features.extend([energy, entropy])
            
            # 4. Features de texture (matrices de co-occurrence simplifiées)
            # Gradients horizontaux et verticaux
            grad_x = np.abs(np.diff(gray, axis=1)).mean()
            grad_y = np.abs(np.diff(gray, axis=0)).mean()
            
            features.extend([grad_x, grad_y])
            
            features_list.append(features)
            
        except Exception as e:
            warnings.warn(f"Erreur extraction features image {i}: {e}")
            # Features par défaut en cas d'erreur
            features_list.append([0] * 70)  # 64 (hist) + 4 (moments) + 2 (énergie/entropie) + 2 (gradients)
    
    features_array = np.array(features_list)
    print(f"✅ Features extraites: {features_array.shape}")
    
    return features_array


def create_ensemble_comparison() -> Dict[str, Any]:
    """
    Crée une comparaison des différentes méthodes d'ensemble
    
    Returns:
        Dictionnaire avec les configurations d'ensemble
    """
    from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                                 BaggingClassifier, AdaBoostClassifier,
                                 GradientBoostingClassifier, ExtraTreesClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    
    ensembles = {
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'type': 'bagging',
            'description': 'Moyenne de multiples arbres avec bootstrap'
        },
        
        'Extra Trees': {
            'model': ExtraTreesClassifier(n_estimators=100, random_state=42),
            'type': 'bagging', 
            'description': 'Arbres avec splits aléatoires'
        },
        
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'type': 'boosting',
            'description': 'Correction séquentielle des erreurs'
        },
        
        'AdaBoost': {
            'model': AdaBoostClassifier(n_estimators=100, random_state=42),
            'type': 'boosting',
            'description': 'Pondération adaptative des échantillons'
        },
        
        'Voting Classifier': {
            'model': VotingClassifier([
                ('lr', LogisticRegression(random_state=42)),
                ('dt', DecisionTreeClassifier(random_state=42)),
                ('svc', SVC(probability=True, random_state=42))
            ], voting='soft'),
            'type': 'voting',
            'description': 'Vote pondéré de modèles diversifiés'
        },
        
        'Bagging': {
            'model': BaggingClassifier(
                base_estimator=DecisionTreeClassifier(random_state=42),
                n_estimators=100, 
                random_state=42
            ),
            'type': 'bagging',
            'description': 'Bootstrap aggregating avec arbres de décision'
        }
    }
    
    return ensembles


def get_model_type(model_name: str) -> str:
    """
    Détermine le type d'un modèle à partir de son nom
    
    Args:
        model_name: Nom du modèle
        
    Returns:
        Type du modèle ('tree', 'linear', 'ensemble', 'deep', 'other')
    """
    model_name_lower = model_name.lower()
    
    if any(keyword in model_name_lower for keyword in ['forest', 'tree', 'xgb', 'lgb', 'catb', 'boost']):
        return 'ensemble'
    elif any(keyword in model_name_lower for keyword in ['linear', 'logistic', 'ridge', 'lasso']):
        return 'linear'
    elif any(keyword in model_name_lower for keyword in ['svm', 'svc']):
        return 'svm'
    elif any(keyword in model_name_lower for keyword in ['neural', 'cnn', 'deep', 'resnet', 'vgg']):
        return 'deep'
    elif any(keyword in model_name_lower for keyword in ['naive', 'bayes']):
        return 'probabilistic'
    else:
        return 'other'
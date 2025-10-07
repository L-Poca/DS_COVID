# 02_modele_ml_gridsearch.py

import numpy as np
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from tqdm import tqdm
import random
import word_generator


# === IMPORTER LES OUTILS V1 ===
from _01_modele_ml_baseline import (
    SEED, IMG_SIZE, SAMPLES_PER_CLASS,
    classes, class_labels,
    class_paths_images, class_paths_masks,
    load_images_flat, load_masks_flat, masking,
    visualize_prediction_errors, analyze_error_patterns
)


def train_with_gridsearch(model, model_name, params, X_train, X_test, y_train, y_test, results_list):
    """
    Entraîne un modèle avec GridSearchCV et enregistre les résultats.
    """
    print(f"\n{'='*60}")
    print(f"🔎 GRID SEARCH pour {model_name}")
    print(f"{'='*60}")

    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=3,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f"[INFO] Meilleurs paramètres pour {model_name} : {grid.best_params_}")
    print(f"[INFO] Meilleure accuracy CV : {grid.best_score_:.4f}")

    # Évaluation finale
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=classes)
    print(f"\n[RESULTS] Rapport de classification (meilleur {model_name}) :\n")
    print(report)

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    cm_filename = f"cm_grid_{model_name.replace(' ', '_').lower()}.png"
    cm_path = os.path.join("rapports", cm_filename)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Classe prédite')
    plt.ylabel('Classe réelle')
    plt.title(f'Matrice de confusion - {model_name} (GridSearch)')
    plt.tight_layout()
    os.makedirs("rapports", exist_ok=True)
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Erreurs
    errors_path = visualize_prediction_errors(best_model, model_name, X_test, y_test, y_pred)

    # Analyse erreurs
    analyze_error_patterns(y_test, y_pred, model_name)

    # Ajout rapport
    results_list.append({
        "name": f"{model_name} (GridSearch)",
        "best_params": grid.best_params_,
        "cv_score": grid.best_score_,
        "report": report,
        "confusion_matrix_path": cm_path,
        "errors_visualization_path": errors_path
    })


if __name__ == "__main__":
    print("[INFO] Démarrage v2 avec GridSearch...")

    # Chargement données
    data_img, labels_img = [], []
    for label, path in class_paths_images.items():
        d, l = load_images_flat(path, label, IMG_SIZE, max_images=SAMPLES_PER_CLASS)
        data_img += d
        labels_img += l

    data_masks, labels_masks = [], []
    for label, path in class_paths_masks.items():
        d, l = load_masks_flat(path, label, IMG_SIZE, max_masks=SAMPLES_PER_CLASS)
        data_masks += d
        labels_masks += l

    assert len(data_img) == len(data_masks)
    assert len(labels_img) == len(labels_masks)

    print(f"[INFO] Application du masquage sur {len(data_img)} images...")

    data_img_2d = [img.reshape(IMG_SIZE) for img in data_img]
    data_masks_2d = [mask.reshape(IMG_SIZE) for mask in data_masks]
    data_masked = [masking(img, mask) for img, mask in zip(data_img_2d, data_masks_2d)]
    X = np.array([img.flatten() for img in data_masked])
    y = np.array(labels_img)

    print(f"[INFO] Données prêtes : {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Charger JSON paramètres
    with open(r"notebooks\modelebaseline\grid_search_params.json", "r") as f:
        grid_params = json.load(f)

    # Résultats
    models_results = []

    # Random Forest
    train_with_gridsearch(
        RandomForestClassifier(random_state=SEED),
        "Random Forest",
        grid_params["RandomForest"],
        X_train, X_test, y_train, y_test,
        models_results
    )

    # Linear SVM
    train_with_gridsearch(
        LinearSVC(random_state=SEED),
        "Linear SVM",
        grid_params["LinearSVM"],
        X_train, X_test, y_train, y_test,
        models_results
    )

    # Rapport Word
    try:
        word_generator.generate_word_report(models_results)
    except ImportError:
        print("[WARNING] 'word_generator.py' manquant.")
    except Exception as e:
        print(f"[ERROR] Rapport Word échoué : {e}")

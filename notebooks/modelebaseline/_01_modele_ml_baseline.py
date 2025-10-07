# 01_modele_ml_baseline.py

import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from tqdm import tqdm
import random


# Configuration
SEED = 42
IMG_SIZE = (128, 128)
SAMPLES_PER_CLASS = 1000

# Définition des classes
classes = ['COVID', 'NORMAL', 'VIRAL', 'LUNG']
class_labels = {0: 'COVID', 1: 'NORMAL', 2: 'VIRAL', 3: 'LUNG'}

# Chemins vers les images
class_paths_images = {
    0: r"data\raw\COVID-19_Radiography_Dataset\COVID-19_Radiography_Dataset\COVID\images",
    1: r"data\raw\COVID-19_Radiography_Dataset\COVID-19_Radiography_Dataset\Normal\images",
    2: r"data\raw\COVID-19_Radiography_Dataset\COVID-19_Radiography_Dataset\Viral Pneumonia\images",
    3: r"data\raw\COVID-19_Radiography_Dataset\COVID-19_Radiography_Dataset\Lung_Opacity\images",
}

# Chemins vers les masques
class_paths_masks = {
    0: r"data\raw\COVID-19_Radiography_Dataset\COVID-19_Radiography_Dataset\COVID\masks",
    1: r"data\raw\COVID-19_Radiography_Dataset\COVID-19_Radiography_Dataset\Normal\masks",
    2: r"data\raw\COVID-19_Radiography_Dataset\COVID-19_Radiography_Dataset\Viral Pneumonia\masks",
    3: r"data\raw\COVID-19_Radiography_Dataset\COVID-19_Radiography_Dataset\Lung_Opacity\masks",
}


def load_images_flat(folder_path, label, img_size, max_images=None):
    """Charge les images en niveaux de gris, redimensionne et aplatit."""
    data, labels = [], []
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    random.seed(SEED)
    random.shuffle(files)

    if max_images:
        files = files[:max_images]

    print(f"[INFO] Chargement de {len(files)} images pour la classe {class_labels[label]}")

    for file in tqdm(files, desc=f"Images - {class_labels[label]}", leave=False):
        try:
            img = Image.open(os.path.join(folder_path, file)).convert('L')
            img = img.resize(img_size)
            data.append(np.array(img).flatten())
            labels.append(label)
        except Exception as e:
            print(f"[WARNING] Erreur avec {file} : {e}")

    return data, labels


def load_masks_flat(folder_path, label, mask_size, max_masks=None):
    """Charge les masques en niveaux de gris, redimensionne et aplatit."""
    data, labels = [], []
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    random.seed(SEED)
    random.shuffle(files)

    if max_masks:
        files = files[:max_masks]

    print(f"[INFO] Chargement de {len(files)} masques pour la classe {class_labels[label]}")

    for file in tqdm(files, desc=f"Masques - {class_labels[label]}", leave=False):
        try:
            mask = Image.open(os.path.join(folder_path, file)).convert('L')
            mask = mask.resize(mask_size)
            data.append(np.array(mask).flatten())
            labels.append(label)
        except Exception as e:
            print(f"[WARNING] Erreur avec {file} : {e}")

    return data, labels


def masking(img: np.array, mask: np.array) -> np.array:
    """
    Applique un masque binaire à une image.
    - Normalise l'image entre 0 et 1
    - Binarise le masque
    - Applique le masque
    """
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)  # Normalisation
    mask_binary = (mask > 0).astype(np.float32)  # Masque binaire
    return img * mask_binary


def visualize_prediction_errors(model, model_name, X_test, y_test, y_pred, max_errors=6):
    """
    Visualise les erreurs de prédiction d'un modèle en affichant les images mal classifiées.
    
    Args:
        model: Le modèle entraîné
        model_name: Nom du modèle pour le titre
        X_test: Images de test (aplaties)
        y_test: Vraies étiquettes
        y_pred: Prédictions du modèle
        max_errors: Nombre maximum d'erreurs à afficher
    """
    # Identifier les erreurs
    error_indices = np.where(y_test != y_pred)[0]
    
    if len(error_indices) == 0:
        print(f"[INFO] {model_name} : Aucune erreur trouvée !")
        return None
    
    print(f"[INFO] {model_name} : {len(error_indices)} erreurs détectées")
    
    # Limiter le nombre d'erreurs à afficher
    num_errors_to_show = min(max_errors, len(error_indices))
    selected_errors = np.random.choice(error_indices, num_errors_to_show, replace=False)
    
    # Configuration de la grille d'affichage
    cols = 3
    rows = (num_errors_to_show + cols - 1) // cols
    
    plt.figure(figsize=(15, 5 * rows))
    plt.suptitle(f'Erreurs de prédiction - {model_name}', fontsize=16, fontweight='bold')
    
    for i, error_idx in enumerate(selected_errors):
        # Récupérer l'image (la reformater en 2D)
        img = X_test[error_idx].reshape(IMG_SIZE)
        true_label = y_test[error_idx]
        pred_label = y_pred[error_idx]
        
        # Obtenir les probabilités de prédiction si possible
        try:
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X_test[error_idx].reshape(1, -1))[0]
                confidence = probas[pred_label]
            elif hasattr(model, 'decision_function'):
                scores = model.decision_function(X_test[error_idx].reshape(1, -1))[0]
                if len(scores.shape) > 0 and len(scores) > 1:
                    # Multi-classe
                    confidence = scores[pred_label] / np.max(np.abs(scores))
                else:
                    # Binaire
                    confidence = abs(scores) if isinstance(scores, (int, float)) else abs(scores[0])
            else:
                confidence = "N/A"
        except:
            confidence = "N/A"
        
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'Vraie: {class_labels[true_label]}\n'
                 f'Prédite: {class_labels[pred_label]}\n'
                 f'Confiance: {confidence:.3f}' if confidence != "N/A" else 
                 f'Vraie: {class_labels[true_label]}\nPrédite: {class_labels[pred_label]}',
                 fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Sauvegarder l'image des erreurs
    errors_filename = f"erreurs_{model_name.replace(' ', '_').lower()}.png"
    errors_path = os.path.join("rapports", errors_filename)
    
    try:
        plt.savefig(errors_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Erreurs de prédiction sauvegardées : {errors_path}")
    except Exception as e:
        print(f"[ERROR] Impossible de sauvegarder les erreurs : {e}")
    finally:
        plt.close()
    
    return errors_path


def analyze_error_patterns(y_test, y_pred, model_name):
    """
    Analyse les patterns d'erreurs pour identifier les confusions les plus fréquentes.
    """
    print(f"\n[ANALYSIS] Analyse des erreurs pour {model_name}:")
    
    # Matrice de confusion pour analyse détaillée
    cm = confusion_matrix(y_test, y_pred)
    
    print("Confusions les plus fréquentes:")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i][j] > 0:
                print(f"  {class_labels[i]} → {class_labels[j]} : {cm[i][j]} fois")
    
    # Calculer les taux d'erreur par classe
    print("\nTaux d'erreur par classe:")
    for i in range(len(classes)):
        total_class = np.sum(cm[i, :])
        correct_class = cm[i, i]
        error_rate = (total_class - correct_class) / total_class * 100 if total_class > 0 else 0
        print(f"  {class_labels[i]} : {error_rate:.2f}% d'erreurs")


def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, results_list):
    """
    Entraîne le modèle, évalue, sauvegarde la matrice de confusion,
    visualise les erreurs et ajoute les résultats au rapport.
    """
    print(f"\n" + "="*60)
    print(f"🚀 ENTRAÎNEMENT DU MODÈLE : {model_name}")
    print("="*60)

    # Entraînement
    model.fit(X_train, y_train)

    # Prédiction
    y_pred = model.predict(X_test)

    # Rapport de classification
    report = classification_report(y_test, y_pred, target_names=classes)
    print(f"\n[RESULTS] Rapport de classification pour {model_name} :\n")
    print(report)

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    cm_filename = f"cm_{model_name.replace(' ', '_').lower()}.png"
    cm_path = os.path.join("rapports", cm_filename)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Classe prédite')
    plt.ylabel('Classe réelle')
    plt.title(f'Matrice de confusion - {model_name}')
    plt.tight_layout()

    # Créer le dossier
    os.makedirs("rapports", exist_ok=True)

    # Sauvegarder l'image
    try:
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        size = os.path.getsize(cm_path)
        if size < 1000:
            print(f"[WARNING] Image très petite ({size} octets) : {cm_path}")
        else:
            print(f"[INFO] Matrice de confusion sauvegardée ({size} octets) : {cm_path}")
    except Exception as e:
        print(f"[ERROR] Impossible de sauvegarder l'image : {e}")
    finally:
        plt.close()

    # === NOUVEAU : Visualisation des erreurs ===
    errors_path = visualize_prediction_errors(model, model_name, X_test, y_test, y_pred)
    
    # === NOUVEAU : Analyse des patterns d'erreurs ===
    analyze_error_patterns(y_test, y_pred, model_name)

    # Ajout au rapport
    results_list.append({
        'name': model_name,
        'report': report,
        'confusion_matrix_path': cm_path,
        'errors_visualization_path': errors_path  # Nouveau champ
    })

    # Exemples de prédictions
    print(f"[INFO] Exemples de prédictions avec {model_name} :")
    for i, pred in enumerate(y_pred[:5]):
        print(f"Image {i + 1} : {class_labels[pred]} détectée")


if __name__ == "__main__":
    print("[INFO] Démarrage du chargement des données...")

    # === Chargement des images ===
    data_img, labels_img = [], []
    for label, path in class_paths_images.items():
        d, l = load_images_flat(path, label, IMG_SIZE, max_images=SAMPLES_PER_CLASS)
        data_img += d
        labels_img += l

    # === Chargement des masques ===
    data_masks, labels_masks = [], []
    for label, path in class_paths_masks.items():
        d, l = load_masks_flat(path, label, IMG_SIZE, max_masks=SAMPLES_PER_CLASS)
        data_masks += d
        labels_masks += l

    # === Vérification ===
    assert len(data_img) == len(data_masks), "[ERROR] Nombre d'images ≠ nombre de masques"
    assert len(labels_img) == len(labels_masks), "[ERROR] Labels images ≠ labels masques"

    print(f"[INFO] Application du masquage sur {len(data_img)} images...")

    # === Appliquer le masquage ===
    try:
        # Reformater les images et masques en 2D pour le masquage
        data_img_2d = [img.reshape(IMG_SIZE) for img in data_img]
        data_masks_2d = [mask.reshape(IMG_SIZE) for mask in data_masks]

        # Appliquer le masquage
        data_masked = [masking(img, mask) for img, mask in zip(data_img_2d, data_masks_2d)]

        # Ré-aplatir pour le modèle
        X = np.array([img.flatten() for img in data_masked])
        y = np.array(labels_img)

        print(f"[INFO] Masquage terminé. Dimensions finales : {X.shape}")
    except Exception as e:
        print(f"[ERROR] Échec du masquage : {e}")
        exit(1)

    # === Normalisation (déjà faite dans masking, mais on peut renormaliser) ===
    # X = X / 255.0  # Optionnel si tu veux renormaliser

    # === Split train/test ===
    print("[INFO] Découpage des données en train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # === Liste pour le rapport ===
    models_results = []

    # === Entraînement des modèles ===
    train_and_evaluate_model(
        RandomForestClassifier(n_estimators=100, random_state=SEED),
        "Random Forest",
        X_train, X_test, y_train, y_test,
        models_results
    )

    train_and_evaluate_model(
        LinearSVC(max_iter=5000, random_state=SEED, verbose=2),
        "Linear SVM",
        X_train, X_test, y_train, y_test,
        models_results
    )

    train_and_evaluate_model(
        KNeighborsClassifier(n_neighbors=10),
        "K-Nearest Neighbors",
        X_train, X_test, y_train, y_test,
        models_results
    )

    train_and_evaluate_model(
        GradientBoostingClassifier(n_estimators=100, random_state=SEED, verbose=2),
        "Gradient Boosting",
        X_train, X_test, y_train, y_test,
        models_results
    )

    # === Génération du rapport Word ===
    try:
        import word_generator
        word_generator.generate_word_report(models_results)
    except ImportError:
        print("[WARNING] 'word_generator.py' non trouvé. Crée-le pour générer un rapport Word.")
    except Exception as e:
        print(f"[ERROR] Échec de la génération du rapport : {e}")
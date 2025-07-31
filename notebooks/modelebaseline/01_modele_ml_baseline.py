import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from tqdm import tqdm
import random

print("test raf 3")
# Configuration
SEED = 42
IMG_SIZE = (299, 299)
SAMPLES_PER_CLASS = 1000

# Définition des classes
classes = ['COVID', 'NORMAL', 'VIRAL', 'LUNG']
class_labels = {0: 'COVID', 1: 'NORMAL', 2: 'VIRAL', 3: 'LUNG'}

# Définition des chemins par classe
class_paths = {
    0: r"data\raw\COVID-19_Radiography_Dataset\COVID-19_Radiography_Dataset\COVID\images",
    1: r"data\raw\COVID-19_Radiography_Dataset\COVID-19_Radiography_Dataset\Normal\images",
    2: r"data\raw\COVID-19_Radiography_Dataset\COVID-19_Radiography_Dataset\Viral Pneumonia\images",
    3: r"data\raw\COVID-19_Radiography_Dataset\COVID-19_Radiography_Dataset\Lung_Opacity\images",
}


def load_images_flat(folder_path, label, img_size=(299, 299), max_images=None):
    """Charge les images, les convertit en niveaux de gris, redimensionne et aplatit."""
    data, labels = [], []
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    random.seed(SEED)
    random.shuffle(files)

    if max_images:
        files = files[:max_images]

    print(f"[INFO] Chargement de {len(files)} images pour la classe {class_labels[label]}")

    for file in tqdm(files, desc=f"Classe {class_labels[label]}", leave=False):
        try:
            img = Image.open(os.path.join(folder_path, file)).convert('L')
            img = img.resize(img_size)
            data.append(np.array(img).flatten())
            labels.append(label)
        except Exception as e:
            print(f"[WARNING] Erreur avec {file} : {e}")

    return data, labels


def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Entraîne le modèle donné et affiche les résultats."""
    print(f"\n[INFO] Entraînement du modèle {model_name} en cours...")
    model.fit(X_train, y_train)
    print(f"[INFO] Prédiction avec le modèle {model_name}...")
    y_pred = model.predict(X_test)

    print(f"\n[RESULTS] Rapport de classification pour {model_name} :\n")
    print(classification_report(y_test, y_pred, target_names=classes))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Classe prédite')
    plt.ylabel('Classe réelle')
    plt.title(f'Matrice de confusion - {model_name}')
    plt.tight_layout()
    plt.show()

    # Affichage de quelques prédictions
    print(f"[INFO] Exemples de prédictions avec {model_name} :")
    for i, pred in enumerate(y_pred[:5]):
        print(f"Image {i + 1} : {class_labels[pred]} Detected")


if __name__ == "__main__":
    print("[INFO] Démarrage du chargement des données...")

    data, labels = [], []
    for label, path in class_paths.items():
        d, l = load_images_flat(path, label, img_size=IMG_SIZE, max_images=SAMPLES_PER_CLASS)
        data += d
        labels += l

    print(f"\n[INFO] Nombre total d'images chargées : {len(data)}")

    X = np.array(data)
    y = np.array(labels)

    print(f"[INFO] Dimensions des features : {X.shape}")
    print("[INFO] Découpage des données en train/test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Entraînement et évaluation des modèles
    train_and_evaluate_model(RandomForestClassifier(n_estimators=100, random_state=SEED),
                             "Random Forest", X_train, X_test, y_train, y_test)

    # train_and_evaluate_model(LinearSVC(max_iter=2500, random_state=SEED),
                             # "Linear SVM", X_train, X_test, y_train, y_test)

    train_and_evaluate_model(KNeighborsClassifier(n_neighbors=10),
                             "K-Nearest Neighbors", X_train, X_test, y_train, y_test)

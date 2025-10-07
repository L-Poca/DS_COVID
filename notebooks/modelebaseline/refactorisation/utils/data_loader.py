# refactorisation/utils/data_loader.py
import os
import random
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

# Charger .env
load_dotenv()

# Racine du projet
ROOT = Path(__file__).resolve().parents[1]

# Charger settings.json
DEFAULT_SETTINGS_PATH = ROOT / "config" / "settings.json"
if not DEFAULT_SETTINGS_PATH.exists():
    raise FileNotFoundError(f"settings.json introuvable : {DEFAULT_SETTINGS_PATH}")

with open(DEFAULT_SETTINGS_PATH, "r", encoding="utf-8") as f:
    SETTINGS = json.load(f)

SEED = int(os.getenv("SEED", 42))
random.seed(SEED)
IMG_SIZE = tuple(SETTINGS.get("IMG_SIZE", [128, 128]))
SAMPLES_PER_CLASS = SETTINGS.get("SAMPLES_PER_CLASS", 100)
CLASSES = SETTINGS.get("CLASSES", [])

# Paths
DATASET_ROOT = os.getenv(
    "DATASET_PATH",
    str(ROOT / "data" / "raw" / "COVID-19_Radiography_Dataset" / "COVID-19_Radiography_Dataset")
)
MASKS_ROOT = os.getenv("MASKS_PATH", DATASET_ROOT)

DEFAULT_FOLDER_MAP = {
    "COVID": "COVID",
    "NORMAL": "Normal",
    "VIRAL": "Viral Pneumonia",
    "LUNG": "Lung_Opacity"
}

def _list_png_files(folder_path):
    try:
        return [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]
    except FileNotFoundError:
        return []

def load_images_flat_for_class(folder_path, label_index, img_size, max_images=None):
    """Charge les images (aplaties)."""
    data, labels = [], []
    files = _list_png_files(folder_path)
    random.shuffle(files)

    if max_images:
        files = files[:max_images]

    for file in tqdm(files, desc=f"Images - label_{label_index}", leave=False):
        file_path = os.path.join(folder_path, file)
        try:
            img = Image.open(file_path).convert("L")
            img = img.resize(img_size)
            data.append(np.array(img).flatten())
            labels.append(label_index)
        except Exception as e:
            print(f"[WARNING] Erreur lecture {file_path} : {e}")
    return data, labels, files

def load_masks_flat_for_class(folder_path, label_index, mask_size, file_list):
    """Charge les masques correspondants (même ordre que file_list)."""
    data = []
    for file in tqdm(file_list, desc=f"Masques - label_{label_index}", leave=False):
        file_path = os.path.join(folder_path, file)
        try:
            mask = Image.open(file_path).convert("L")
            mask = mask.resize(mask_size)
            data.append(np.array(mask).flatten())
        except Exception as e:
            print(f"[WARNING] Erreur lecture masque {file_path} : {e}")
            data.append(np.zeros(mask_size).flatten())  # fallback masque vide
    return data

def masking(img: np.array, mask: np.array) -> np.array:
    """Applique un masque binaire à une image."""
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)  # Normalisation
    mask_binary = (mask > 0).astype(np.float32)  # Binarisation
    return img * mask_binary

def load_dataset(img_size=None, samples_per_class=None, classes=None, folder_map=None, dataset_root=None, masks_root=None):
    """
    Charge dataset + masques et applique le masquage.
    """
    img_size = tuple(img_size) if img_size else IMG_SIZE
    samples_per_class = samples_per_class if samples_per_class is not None else SAMPLES_PER_CLASS
    classes = classes if classes else CLASSES
    folder_map = folder_map if folder_map else DEFAULT_FOLDER_MAP
    dataset_root = dataset_root if dataset_root else DATASET_ROOT
    masks_root = masks_root if masks_root else MASKS_ROOT

    X_all, y_all = [], []

    for idx, class_name in enumerate(classes):
        folder_name = folder_map.get(class_name, class_name)
        images_folder = os.path.join(dataset_root, folder_name, "images")
        masks_folder = os.path.join(masks_root, folder_name, "masks")

        if not os.path.exists(images_folder):
            images_folder_alt = os.path.join(dataset_root, folder_name)
            if os.path.exists(images_folder_alt):
                images_folder = images_folder_alt
            else:
                print(f"[WARNING] Dossier introuvable pour {class_name} : {images_folder}")
                continue

        data_img, labels, files = load_images_flat_for_class(images_folder, idx, img_size, max_images=samples_per_class)

        if os.path.exists(masks_folder):
            data_masks = load_masks_flat_for_class(masks_folder, idx, img_size, files)
        else:
            print(f"[WARNING] Pas de dossier masque pour {class_name}, masques ignorés")
            data_masks = [np.ones(img_size).flatten()] * len(data_img)

        # Appliquer masquage
        data_masked = [masking(img.reshape(img_size), mask.reshape(img_size)).flatten()
                       for img, mask in zip(data_img, data_masks)]

        X_all.extend(data_masked)
        y_all.extend(labels)

    if len(X_all) == 0:
        raise RuntimeError("Aucune image chargée — vérifie DATASET_PATH/MASKS_PATH et la structure des dossiers.")

    X = np.array(X_all)
    y = np.array(y_all, dtype=int)
    return X, y

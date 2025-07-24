import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def list_image_files(image_folder):
    """Retourne la liste des fichiers images dans un dossier."""
    images_list = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Nombre d'images trouvées dans {image_folder} : {len(images_list)}")
    if not images_list:
        raise ValueError(f"Aucune image trouvée dans le dossier : {image_folder}")
    return images_list

def load_image(image_path, dsize=None, grayscale=True):
    """Charge et prétraite une image (redimensionnement, conversion)."""
    if grayscale:
        flag = cv2.IMREAD_GRAYSCALE
    else:
        flag = cv2.IMREAD_COLOR
    img = cv2.imread(image_path, flag)
    if img is None:
        raise ValueError(f"Image non trouvée : {image_path}")
    if dsize:
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
    return img

def flatten_image(img):
    """Aplati une image numpy en 1D."""
    return img.flatten()

def apply_mask(image, mask):
    """Applique un masque binaire à une image."""
    if mask.shape != image.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    if len(image.shape) == 3 and image.shape[2] == 3:
        mask_bin = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        mask_bin = mask
    return cv2.bitwise_and(image, mask_bin)

def extract_stats(image, mask):
    """Extrait des statistiques de l'image et de la zone masquée."""
    total_pixels = mask.size
    nbr_pixels_affiches = np.count_nonzero(mask)
    ratio_pixels = nbr_pixels_affiches / total_pixels
    pourcentage_pixels = 100 * ratio_pixels
    image_gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lum_image = np.mean(image_gray)
    contrast_image = np.std(image_gray)
    pixels_masques = image_gray[mask > 0]
    lum_masque = np.mean(pixels_masques) if pixels_masques.size > 0 else 0
    contrast_masque = np.std(pixels_masques) if pixels_masques.size > 0 else 0
    return {
        "nbr_pixels_affiches": nbr_pixels_affiches,
        "total_pixels": total_pixels,
        "ratio_pixels_affiches": ratio_pixels,
        "pourcentage_pixels_affiches": pourcentage_pixels,
        "luminosite_image_complete": lum_image,
        "contraste_image_complete": contrast_image,
        "luminosite_zone_masquee": lum_masque,
        "contraste_zone_masquee": contrast_masque
    }

def process_image(image_path, dsize=None, grayscale=True, mask_path=None, flat_array=False):
    """Charge une image, applique un masque si fourni, retourne pixels et stats."""
    img = load_image(image_path, dsize, grayscale)
    pixels = flatten_image(img) if flat_array else img
    stats = None
    if mask_path and os.path.exists(mask_path):
        mask = load_image(mask_path, dsize, True)
        img_masked = apply_mask(img, mask)
        stats = extract_stats(img, mask)
    else:
        img_masked = None
    return pixels, img_masked, stats

def build_dataframe(images, stats_keys=None):
    """Construit un DataFrame à partir des images et stats."""
    if stats_keys:
        nb_stats = len(stats_keys)
        columns = ['filename'] + stats_keys + [f'pixel_{i}' for i in range(len(images[0]) - 1 - nb_stats)]
    else:
        columns = ['filename'] + [f'pixel_{i}' for i in range(len(images[0]) - 1)]
    return pd.DataFrame(images, columns=columns)

def export_dataframe_to_csv(df, FileName, dsize):
    """Exporte le DataFrame en CSV."""
    dsize_str = f"{dsize[0]}x{dsize[1]}" if dsize else "original"
    Destination_CSV_Path = rf'../data/generated/{FileName}_{dsize_str}.csv'
    os.makedirs('../data/generated', exist_ok=True)
    df.to_csv(Destination_CSV_Path, index=False, header=True, sep=',')
    print(f"\nDataFrame enregistré dans le fichier : \n{Destination_CSV_Path}")

def show_image_and_stats(image, image_masquee, stats, image_file):
    """Affiche l'image originale, masquée et les stats."""
    plt.close()
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray' if len(image.shape)==2 else None)
    plt.title('Image originale')
    plt.axis('off')
    if image_masquee is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(image_masquee, cmap='gray' if len(image_masquee.shape)==2 else None)
        plt.title('Image masquée')
        plt.axis('off')
    if stats:
        plt.suptitle(f"{image_file}\n" + "\n".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in stats.items()]), fontsize=10)
    plt.tight_layout()
    plt.show(block=True)


def main_workflow(image_folder, dsize=None, grayscale=True, flat_array=False, mask_folder=None, show_each=False, FileName=None, export_csv=True):
    image_files = list_image_files(image_folder)
    images = []
    images_masked = []
    stats_list = []
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, image_file) if mask_folder else None
        pixels, img_masked, stats = process_image(image_path, dsize, grayscale, mask_path, flat_array)
        row = [image_file]
        if stats:
            row += list(stats.values())
            stats_list.append(stats)
        else:
            row += [None]*8
        row += pixels.tolist() if hasattr(pixels, 'tolist') else list(pixels)
        images.append(row)
        if img_masked is not None:
            images_masked.append(img_masked)
        if show_each:
            show_image_and_stats(pixels if not flat_array else pixels.reshape(dsize), img_masked, stats, image_file)
        print(f"Image {i+1}/{len(image_files)} traitée : {image_file}", end='\r')
    print(f"\n{len(images)} images traitées.")
    stats_keys = list(stats_list[0].keys()) if stats_list else None
    df = build_dataframe(images, stats_keys)
    print(f"DataFrame shape : {df.shape}")
    if export_csv and FileName:
        export_dataframe_to_csv(df, FileName, dsize)
    return df, images_masked, stats_list


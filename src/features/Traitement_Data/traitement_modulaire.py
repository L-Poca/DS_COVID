import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def list_image_files(image_folder):
    """Retourne la liste des fichiers images dans un dossier."""
    images_list = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    #print(f"Nombre d'images trouvées dans {image_folder} : {len(images_list)}")
    print(f"Nombre d'images trouvées : {len(images_list)}")
    if not images_list:
        raise ValueError(f"Aucune image trouvée dans le dossier : {image_folder}")
    return images_list

def load_image(image_path, dsize=None, grayscale=True, categorie=None, sub_categorie=None, save_images=False):
    """Charge et prétraite une image (redimensionnement, conversion)."""
    # Appliquer GrayScale si demandé
    if grayscale:
        flag = cv2.IMREAD_GRAYSCALE
    else:
        flag = cv2.IMREAD_COLOR

    # Chargement de l'image avec OpenCV
    img = cv2.imread(image_path, flag)

    # Si l'image n'est pas trouvée, on essaie avec PIL
    if img is None:
        # Fallback PIL
        try:
            from PIL import Image
            img_pil = Image.open(image_path)
            if grayscale:
                img = np.array(img_pil.convert('L'))
            else:
                img = np.array(img_pil.convert('RGB'))
            #print(f"Image chargée avec PIL : {image_path}")
        

        except Exception as e:
            raise ValueError(f"Image non trouvée ou illisible : {image_path} ({e})")
        
    # Redimensionnement si demandé
    if dsize:
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)

        # Enregistrement de l'image si demandé
        if save_images and categorie:
            save_path = os.path.join(f'../data/processed/{dsize[0]}x{dsize[1]}/{categorie}/{sub_categorie}', os.path.basename(image_path))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img)

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

    # Statistiques sur les pixels
    total_pixels = mask.size # Nombre total de pixels dans le masque
    nbr_pixels_affiches = np.count_nonzero(mask) # Nombre de pixels affichés (non masqués)
    ratio_pixels = nbr_pixels_affiches / total_pixels # Ratio de pixels affichés
    pourcentage_pixels = 100 * ratio_pixels # Pourcentage de pixels affichés

    # Conversion de l'image en niveaux de gris si nécessaire
    image_gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calcul de la luminosité et du contraste
    lum_image = np.mean(image_gray) # Luminosité de l'image complète
    contrast_image = np.std(image_gray) # Contraste de l'image complète
    pixels_masques = image_gray[mask > 0]  # Pixels de la zone masquée
    lum_masque = np.mean(pixels_masques) if pixels_masques.size > 0 else 0 # Luminosité de la zone masquée
    contrast_masque = np.std(pixels_masques) if pixels_masques.size > 0 else 0 # Contraste de la zone masquée

    # Création du dictionnaire de résultats
    results = {
        "nbr_pixels_affiches": nbr_pixels_affiches,
        "total_pixels": total_pixels,
        "ratio_pixels_affiches": ratio_pixels,
        "pourcentage_pixels_affiches": pourcentage_pixels,
        "luminosite_image_complete": lum_image,
        "contraste_image_complete": contrast_image,
        "luminosite_zone_masquee": lum_masque,
        "contraste_zone_masquee": contrast_masque
    }

    return results

def process_image(image_path,dsize=None, grayscale=True, mask_path=None, flat_array=False, categorie=None, save_images=False):

    """Charge une image, applique un masque si fourni, retourne pixels et stats."""

    # Chargement de l'image
    img = load_image(image_path,
                     dsize,
                     grayscale=grayscale,
                     categorie=categorie,
                     sub_categorie="images",
                     save_images=save_images)

    # Aplatissement de l'image si nécessaire
    pixels = flatten_image(img) if flat_array else img

    # Initialisation des statistiques
    stats = None

    # Si un masque est fourni, on l'applique
    if mask_path and os.path.exists(mask_path):
        # Chargement du masque
        mask = load_image(mask_path,
                          dsize,
                          grayscale=grayscale,
                          categorie=categorie,
                          sub_categorie="masks",
                          save_images=save_images)
        
        # Appliquer le masque à l'image
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
    
    plt.close()# sensé éviter les doublons d'affichage

    # Affichage de l'image originale
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray' if len(image.shape)==2 else None)
    plt.title('Image originale')
    plt.axis('off')

    # Affichage de l'image masquée si elle existe
    if image_masquee is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(image_masquee, cmap='gray' if len(image_masquee.shape)==2 else None)
        plt.title('Image masquée')
        plt.axis('off')

    # Affichage des statistiques
    if stats:
        plt.suptitle(f"{image_file}\n" + "\n".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in stats.items()]), fontsize=10)
    
    # Ajustement de la mise en page
    plt.tight_layout()
    plt.show(block=True)


def main_workflow(image_folder, dsize=None, grayscale=True, flat_array=False, mask_folder=None, show_each=False, FileName=None, export_csv=False, Categorie=None, save_images=True):
    # Initialisation
    image_files = list_image_files(image_folder)
    images = []
    images_masked = []
    stats_list = []

    # Parcours des images
    print(f"Traitement de {len(image_files)} images...", end='\r')
    for i, image_file in enumerate(image_files):

        # Chemins des images et masques
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, image_file) if mask_folder else None

        # Chargement et traitement de l'image
        pixels, img_masked, stats = process_image(image_path, dsize, grayscale, mask_path, flat_array=flat_array, categorie=Categorie, save_images=save_images)
        row = [image_file] # Nom du fichier image

        # Ajout des statistiques si disponibles
        if stats:
            row += list(stats.values())
            stats_list.append(stats)
        else:
            row += [None]*8

        # Si l'image est aplatie, on ajoute les pixels
        row += pixels.tolist() if hasattr(pixels, 'tolist') else list(pixels)
        images.append(row)

        # Si une image masquée est obtenue, on l'ajoute à la liste
        if img_masked is not None:
            images_masked.append(img_masked)

            # Enregistrement de l'image masquée si demandé
            if save_images:
                save_path = os.path.join(f'../data/processed/{dsize[0]}x{dsize[1]}/{Categorie}/Masked_images', f'masked_{image_file}')
                os.makedirs(os.path.dirname(save_path), exist_ok=True) # Overwrite if exists
                cv2.imwrite(save_path, img_masked)

        # Affichage de l'image et des stats pour debug
        if show_each:
            show_image_and_stats(pixels if not flat_array else pixels.reshape(dsize), img_masked, stats, image_file)
        
        # Mise à jour de la progression
        print(f"Image {i+1}/{len(image_files)} traitée : {image_file}", end='\r')

    # Fin du traitement
    print(f"\n{len(images)} images traitées.", end='\r')

    # Construction du DataFrame
    stats_keys = list(stats_list[0].keys()) if stats_list else None
    df = build_dataframe(images, stats_keys)
    print(f"DataFrame shape : {df.shape}")

    # Export du DataFrame en CSV
    if export_csv and FileName:
        export_dataframe_to_csv(df, FileName, dsize)

    return df, images_masked, stats_list


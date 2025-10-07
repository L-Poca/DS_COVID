import cv2

import os

from sklearn.model_selection import GridSearchCV
from sklearn.manifold import Isomap
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib import offsetbox

def appliquer_masque_et_stats(image_path, masque_path):
    """
    Applique un masque binaire sur une image et extrait des statistiques.
    Retourne l'image masquée et un dictionnaire de stats.
    """
    # Charger l'image couleur et le masque en niveaux de gris
    image = cv2.imread(image_path)
    masque = cv2.imread(masque_path, cv2.IMREAD_GRAYSCALE)

    # Redimensionner le masque si besoin pour qu'il corresponde à l'image
    if masque.shape[:2] != image.shape[:2]:
        masque = cv2.resize(masque, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Binariser le masque (0 ou 255)
    _, masque_bin = cv2.threshold(masque, 127, 255, cv2.THRESH_BINARY)

    # Adapter le masque au nombre de canaux de l'image si besoin
    if len(image.shape) == 3 and image.shape[2] == 3:
        masque_bin_3c = cv2.cvtColor(masque_bin, cv2.COLOR_GRAY2BGR)
    else:
        masque_bin_3c = masque_bin

    # Appliquer le masque à l'image
    image_masquee = cv2.bitwise_and(image, masque_bin_3c)

    # Statistiques sur le masque et l'image
    total_pixels = masque_bin.size
    nbr_pixels_affiches = np.count_nonzero(masque_bin)
    ratio_pixels = nbr_pixels_affiches / total_pixels
    pourcentage_pixels = 100 * ratio_pixels

    # Image en niveaux de gris pour les stats de luminosité/contraste
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Stats sur toute l'image
    lum_image = np.mean(image_gray)
    contrast_image = np.std(image_gray)

    # Stats sur la zone masquée (pixels où masque > 0)
    pixels_masques = image_gray[masque_bin > 0]
    lum_masque = np.mean(pixels_masques) if pixels_masques.size > 0 else 0
    contrast_masque = np.std(pixels_masques) if pixels_masques.size > 0 else 0

    stats = {
        "nbr_pixels_affiches": nbr_pixels_affiches,
        "total_pixels": total_pixels,
        "ratio_pixels_affiches": ratio_pixels,
        "pourcentage_pixels_affiches": pourcentage_pixels,
        "luminosite_image_complete": lum_image,
        "contraste_image_complete": contrast_image,
        "luminosite_zone_masquee": lum_masque,
        "contraste_zone_masquee": contrast_masque
    }
    return image_masquee, stats

def image_to_array(image_path, dsize=None, grayscale=True):
    """
    Convertit une image en tableau numpy, en gris ou couleur, et la redimensionne si besoin.
    """
    # Choix du mode de lecture
    if grayscale:
        C_selected = cv2.COLOR_BGR2GRAY
    else:
        C_selected = cv2.COLOR_BGR2RGB
    img = cv2.imread(image_path, C_selected)
    # Redimensionnement si demandé
    if dsize:
        img_resized = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
        img_array = np.array(img_resized)
    else:
        img_array = np.array(img)
    return img_array

def transform_images_to_arrays(image_folder, dsize=None, grayscale=True, flat_array=False, mask_folder=None, show_each=False):
    """
    Transforme une liste de fichiers image en tableaux numpy, applique éventuellement un masque,
    extrait les statistiques, et affiche la progression et les images/statistiques si demandé.
    """
    # Liste des fichiers images
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total = len(image_files)

    # Verification de la présence d'images
    if total == 0:
        raise ValueError(f"Aucune image trouvée dans le dossier : \n{image_folder}")
    print(f"Traitement des {total} images dans le dossier : \n{image_folder}")


    images = []
    images_masked = []
    stats_list = []

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        try:
            # Conversion image en tableau numpy
            img_array = image_to_array(image_path, dsize=dsize, grayscale=grayscale)
            if flat_array:
                pixels = img_array.flatten()
            else:
                pixels = img_array
            pixels_with_filename = [image_file] + pixels.tolist()

            # Initialisation de l'image masquée et des stats
            stats = None
            image_masquee = None

            # Si un dossier de masques est fourni, appliquer le masque et extraire les stats
            if mask_folder:

                mask_path = os.path.join(mask_folder, image_file)

                # Vérifier si le masque existe
                if os.path.exists(mask_path):
                    image_masquee, stats = appliquer_masque_et_stats(image_path, mask_path)
                    pixels_with_filename += list(stats.values())
                
                # Si le masque n'existe pas, ajouter des NaN pour les stats
                else:
                    pixels_with_filename += [np.nan]*8  # 8 stats

            # Ajouter l'image et son masque

            images.append(pixels_with_filename)
            if image_masquee is not None:
                images_masked.append(image_masquee) 
            if stats is not None:
                stats_list.append(stats)
            
            
            

            # Affichage de la progression dans le terminal
            print(f"Traitement de l'image {i+1}/{total} : {image_file} | Shape : {img_array.shape}", end='       \r')
            
            # Affichage image + image masquée + stats si demandé
            if show_each and mask_folder and image_masquee is not None and stats is not None:

                plt.close()  # Effacer la figure précédente

                # Affichage Image Originale
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
                plt.title('Image originale')
                plt.axis('off')

                # Affichage Image Masquée
                plt.subplot(1, 2, 2)
                plt.imshow(cv2.cvtColor(image_masquee, cv2.COLOR_BGR2RGB))
                plt.title('Image masquée')
                plt.axis('off')

                # Affichage des stats dans le titre
                plt.suptitle(
                    f"{image_file}\n" +
                    "\n".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in stats.items()]),
                    fontsize=10
                )

                # Affichage de la figure
                plt.tight_layout()
                plt.show(block=True)

        except ValueError as e:
            print(f"Erreur pour l'image {image_file}: {e}", end='       \r')

    print(f"\nTraitement terminé. {len(images)} images transformées en tableaux numpy.")
    return images, images_masked, stats_list

def images_numpy_to_dataframe(images, dsize=None, FileName=None, Export_CSV=True, Stats=None):
    """
    Convertit une liste de tableaux numpy d'images en DataFrame, et exporte éventuellement en CSV.
    """

    imagesmax = len(images[0]) - 1

    # Nommer les colonnes du DataFrame
    if Stats: 
        stats_keys = list(Stats.keys())  # stats est le dict retourné par appliquer_masque_et_stats
        nb_stats = len(stats_keys)
        columns = ['filename'] + stats_keys + [f'pixel_{i}' for i in range(len(images[0]) - 1 - nb_stats)] 
    else:
        columns = ['filename'] + [f'pixel_{i}' for i in range(len(images[0]) - 1)]



    print(f"\nCréation du DataFrame avec {len(columns)} colonnes et {len(images)} lignes pour un total de {len(images) * len(columns)} valeurs.")
    df = pd.DataFrame(images, columns=columns)

    print(f"\nShape du DataFrame : {df.shape}")
    print("\nCréation du DataFrame terminée.")

    # Exportation CSV
    if Export_CSV:
        print("\nEnregistrement du DataFrame dans un fichier CSV...")
        if (FileName is None) & Export_CSV:
            raise ValueError("\nFileName doit être spécifié si Export_CSV est True.")
        dsize_str = f"{dsize[0]}x{dsize[1]}" if dsize else "original"
        Destination_CSV_Path = rf'../data/generated/{FileName}_{dsize_str}.csv'
        os.makedirs('../data/generated', exist_ok=True)
        df.to_csv(Destination_CSV_Path, index=False, header=True, sep=',')
        print(f"\nDataFrame enregistré dans le fichier : \n{Destination_CSV_Path}")


    return df

def plot_components(data, model, images=None, ax=None,thumb_frac=0.05, cmap='gray_r', prefit = False): # c'est la fct du cours
    ax = ax or plt.gca()
    
    if not prefit :
        proj = model.fit_transform(data)
    else:
        proj = data
    ax.plot(proj[:, 0], proj[:, 1], '.b')
    
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # On ne montre pas le points trop proches
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=cmap),
                                      proj[i])
            ax.add_artist(imagebox)

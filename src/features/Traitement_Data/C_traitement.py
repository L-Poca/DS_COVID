import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def Get_All_Paths():
    # Chemin vers le dossier contenant les images
    current_file = Path(__file__)
    Global_Path = current_file.parent.parent.parent.parent
    Data_Path = f"{Global_Path}/data/raw/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset"

    #C:\\Users\\serin\\OneDrive\\Bureau\\projet_covid\\COVID-19_Radiography_Dataset
    paths = {
        "img_covid": rf"{Data_Path}/COVID/images",
        "mask_covid": rf"{Data_Path}\COVID\masks",
        "img_normal": rf"{Data_Path}//Normal//images",
        "mask_normal": rf"{Data_Path}\\Normal\\masks",
        "img_lung": rf"{Data_Path}\\Lung_Opacity\\images",
        "mask_lung": rf"{Data_Path}\\Lung_Opacity\\masks",
        "img_pneumonia": rf"{Data_Path}\\Viral Pneumonia\\images",
        "mask_pneumonia": rf"{Data_Path}\\Viral Pneumonia\\masks"
    }
    return paths

def Get_All_Paths_2():
    # Chemin vers le dossier contenant les images
    current_file = Path(__file__)
    Global_Path = current_file.parent.parent.parent.parent
    Data_Path = f"{Global_Path}/data/raw/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset"
    paths = {
        "covid": {
            "img": rf"{Data_Path}/COVID/images",
            "mask": rf"{Data_Path}/COVID/masks"
        },
        "normal": {
            "img": rf"{Data_Path}/Normal/images",
            "mask": rf"{Data_Path}/Normal/masks"
        },
        "lung": {
            "img": rf"{Data_Path}/Lung_Opacity/images",
            "mask": rf"{Data_Path}/Lung_Opacity/masks"
        },
        "pneumonia": {
            "img": rf"{Data_Path}/Viral Pneumonia/images",
            "mask": rf"{Data_Path}/Viral Pneumonia/masks"
        }
    }
    return paths

def calculer_luminosite_contraste(image):

    # Convertit l'image en niveaux de gris
    image_gray = image.convert('L')
    pixels = np.array(image_gray)
    luminosite = np.mean(pixels)
    contraste = np.std(pixels)
    return luminosite, contraste

def Parcours_Dossier_Images(paths):
    """
    Parcourt un ou plusieurs dossiers d'images et calcule la luminosité et le contraste pour chaque image.
    """
    results = {}
    for key, path_img in paths.items():
        print(f"Traitement du dossier : {key}")
        if not os.path.exists(path_img):
            print(f"Le chemin {path_img} n'existe pas.")
            continue
        lum, cont = Extract_Luminosite_Contraste(path_img)
        results[f"luminosite_{key}"] = lum
        results[f"contraste_{key}"] = cont
        if lum and cont:
            Show_plot(lum, cont, key)
    return results

def Parcours_Dossier_Images_2(paths):
    """
    Parcourt un ou plusieurs dossiers d'images et calcule la luminosité et le contraste pour chaque image.
    """
    results = {}
    for key, paths_dict in paths.items():
        print(f"Traitement du dossier : {key}")
        img_path = paths_dict['img']
        mask_path = paths_dict['mask']
        if not os.path.exists(img_path):
            print(f"Le chemin {img_path} n'existe pas.")
            continue
        lum, cont = Extract_Luminosite_Contraste(img_path)
        results[f"luminosite_{key}"] = lum
        results[f"contraste_{key}"] = cont
        if lum and cont:
            Show_plot(lum, cont, key)
    return results  
    

def Extract_Luminosite_Contraste(path_img):
    # Listes pour stocker les résultats
    luminosites = []
    contrastes = []
    # Parcours du dossier d'images
    for fichier in os.listdir(path_img):
        if fichier.lower().endswith(('.png', '.jpg')):
            chemin_image = os.path.join(path_img, fichier)
            try:
                image = Image.open(chemin_image)
                lum, cont = calculer_luminosite_contraste(image)
                luminosites.append(lum)
                contrastes.append(cont)
                print(f"Image: {fichier}, Luminosité: {lum}, Contraste: {cont}")
            except Exception as e:
                print(f"Erreur avec l'image {fichier} : {e}")
    return luminosites, contrastes


def Run_All_Dossiers():
    dict_path = Get_All_Paths()
    Parcours_Dossier_Images(dict_path)

def Run_All_Dossiers_2():
    dict_path = Get_All_Paths_2()
    Parcours_Dossier_Images_2(dict_path)

def Show_plot(luminosites, contrastes, Categories=None):
    # Affichage des histogrammes
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(luminosites, bins=30, color='yellow', edgecolor='black')
    plt.title(f'Histogramme de la luminosité image {Categories}')
    plt.xlabel('Luminosité')
    plt.ylabel("Nombre d'images")

    plt.subplot(1, 2, 2)
    plt.hist(contrastes, bins=30, color='gray', edgecolor='black')
    plt.title(f'Histogramme du contraste image {Categories}')
    plt.xlabel('Contraste')
    plt.ylabel("Nombre d'images")

    plt.tight_layout()
    plt.show()
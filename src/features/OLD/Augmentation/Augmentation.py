"""Module d'augmentation de données pour les images médicales COVID-19.

Ce module fournit des utilitaires pour créer et visualiser des versions
augmentées d'images radiographiques dans le cadre de la classification COVID-19.

Auteur: [Votre nom]
Date: 2025
"""

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_tmp_dir(parent_dir):
    """Crée un répertoire temporaire pour stocker les images à augmenter.

    Cette fonction crée un dossier "tmp" dans le répertoire parent spécifié.
    Le dossier est utilisé pour organiser temporairement les images avant
    l'application des transformations d'augmentation.

    Args:
        parent_dir (str): Chemin vers le répertoire parent où créer le dossier
            tmp

    Returns:
        str: Chemin complet vers le répertoire temporaire créé

    Example:
        >>> parent = "/path/to/images"
        >>> tmp_dir = create_tmp_dir(parent)
        >>> print(tmp_dir)
        "/path/to/images/tmp"
    """
    # Construction du chemin vers le répertoire temporaire
    tmp_class_dir = os.path.join(parent_dir, "tmp")

    # Création du répertoire (exist_ok=True évite les erreurs si déjà existant)
    os.makedirs(tmp_class_dir, exist_ok=True)

    return tmp_class_dir


def copy_images_no_duplicates(images_dir, tmp_class_dir, n=5):
    """Copie un échantillon d'images vers un répertoire temporaire sans doublons.

    Cette fonction extrait le nom de la classe depuis le chemin source,
    crée un sous-dossier approprié dans le répertoire temporaire, et copie
    les n premières images en évitant les doublons.

    Args:
        images_dir (str): Chemin vers le répertoire source contenant les images
        tmp_class_dir (str): Chemin vers le répertoire temporaire de destination
        n (int, optional): Nombre maximum d'images à copier. Défaut: 5

    Example:
        >>> copy_images_no_duplicates("/data/COVID/images", "/tmp", n=10)
        # Copie max 10 images de COVID vers /tmp/COVID/

    Note:
        - La classe est déterminée par le nom du dossier parent d'images_dir
        - Seules les images non déjà présentes sont copiées
    """
    # Extraction du nom de classe depuis le chemin
    # (ex: "COVID" depuis "/path/COVID/images")
    class_name = os.path.basename(os.path.dirname(images_dir))

    # Création du sous-dossier de classe dans le répertoire temporaire
    class_dir = os.path.join(tmp_class_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    # Récupération de la liste des fichiers déjà présents pour éviter
    # les doublons
    existing = set(os.listdir(class_dir))

    # Copie des n premières images en évitant les doublons
    for img_name in os.listdir(images_dir)[:n]:
        if img_name not in existing:
            source_path = os.path.join(images_dir, img_name)
            shutil.copy(source_path, class_dir)


def get_image_generator(config=None):
    """Crée un générateur d'augmentation d'images avec paramètres configurables.

    Cette fonction initialise un ImageDataGenerator de Keras avec des
    transformations spécifiquement adaptées aux images médicales
    radiographiques. Les paramètres par défaut sont optimisés pour préserver
    la qualité diagnostique tout en augmentant la diversité du dataset.

    Args:
        config (dict, optional): Dictionnaire de configuration des
            transformations. Si None, utilise des valeurs par défaut adaptées
            aux images médicales.

    Returns:
        ImageDataGenerator: Générateur configuré pour l'augmentation d'images

    Example:
        >>> # Utilisation avec config par défaut
        >>> generator = get_image_generator()

        >>> # Utilisation avec config personnalisée
        >>> custom_config = {
        ...     "rotation_range": 10,
        ...     "width_shift_range": 0.05,
        ...     "height_shift_range": 0.05,
        ...     "zoom_range": 0.05,
        ...     "horizontal_flip": False,
        ...     "brightness_range": [0.9, 1.1]
        ... }
        >>> generator = get_image_generator(custom_config)

    Note:
        Les paramètres par défaut sont conservateurs pour préserver
        l'intégrité diagnostique des images médicales.
    """
    # Configuration par défaut optimisée pour les images médicales
    if config is None:
        config = {
            # Rotation max ±15° (réaliste pour radiographies)
            "rotation_range": 15,
            # Décalage horizontal max 10%
            "width_shift_range": 0.1,
            # Décalage vertical max 10%
            "height_shift_range": 0.1,
            # Zoom in/out max 10%
            "zoom_range": 0.1,
            # Retournement horizontal (acceptable pour poumons)
            "horizontal_flip": True,
            # Variation de luminosité ±20%
            "brightness_range": [0.8, 1.2]
        }

    # Création et configuration du générateur d'augmentation
    return ImageDataGenerator(
        rotation_range=config["rotation_range"],
        width_shift_range=config["width_shift_range"],
        height_shift_range=config["height_shift_range"],
        zoom_range=config["zoom_range"],
        horizontal_flip=config["horizontal_flip"],
        brightness_range=config["brightness_range"]
    )

def show_augmented_images(generator, tmp_class_dir, target_size=(256, 256),
                          batch_size=5):
    """Affiche un échantillon d'images augmentées pour visualisation.

    Cette fonction génère et affiche un batch d'images transformées en
    appliquant les augmentations configurées dans le générateur. Utile pour
    vérifier visuellement que les transformations appliquées préservent la
    qualité et le contenu diagnostique des images médicales.

    Args:
        generator (ImageDataGenerator): Générateur d'augmentation configuré
        tmp_class_dir (str): Chemin vers le répertoire temporaire contenant
            les images
        target_size (tuple, optional): Taille cible (largeur, hauteur) pour
            le redimensionnement. Défaut: (256, 256)
        batch_size (int, optional): Nombre d'images à afficher simultanément.
            Défaut: 5

    Example:
        >>> generator = get_image_generator()
        >>> show_augmented_images(generator, "/tmp/COVID",
        ...                      target_size=(128, 128), batch_size=3)

    Note:
        - Les images sont affichées en niveaux de gris (approprié pour
          radiographies)
        - La fonction utilise matplotlib pour l'affichage
        - Les images sont mélangées aléatoirement (shuffle=True)
    """
    # Configuration du générateur de flux depuis le répertoire
    gen = generator.flow_from_directory(
        tmp_class_dir,
        target_size=target_size,    # Redimensionnement des images
        color_mode='grayscale',     # Mode niveaux de gris pour radiographies
        batch_size=batch_size,      # Taille du batch à générer
        class_mode=None,            # Pas de labels nécessaires pour visualisation
        shuffle=True                # Mélange aléatoire des images
    )

    # Génération d'un batch d'images augmentées
    batch = next(gen)

    # Configuration de la figure matplotlib
    plt.figure(figsize=(15, 3))

    # Affichage de chaque image du batch
    for i in range(batch_size):
        plt.subplot(1, batch_size, i + 1)
        # squeeze() retire les dimensions unitaires
        # (ex: (256,256,1) -> (256,256))
        plt.imshow(batch[i].squeeze(), cmap='gray')
        plt.axis('off')  # Masque les axes pour un affichage plus propre

    # Titre global pour la figure
    plt.suptitle("Batch d'images augmentées (Keras)")
    plt.show()


def process_class(base_dir, cls, n=5, image_dir="images"):
    """Pipeline complet de traitement et visualisation d'augmentation.

    Cette fonction orchestre l'ensemble du processus d'augmentation pour une
    classe spécifique d'images médicales : préparation des répertoires, copie
    d'échantillons, configuration de l'augmentation, et visualisation des
    résultats.

    Args:
        base_dir (str): Chemin de base vers le dataset
            (ex: "/data/COVID-19_Dataset")
        cls (str): Nom de la classe à traiter
            (ex: "COVID", "Normal", "Viral Pneumonia")
        n (int, optional): Nombre d'images à traiter pour la visualisation.
            Défaut: 5
        image_dir (str, optional): Nom du sous-dossier contenant les images.
            Défaut: "images"

    Example:
        >>> # Traitement de la classe COVID avec 10 images
        >>> process_class("/data/COVID-19_Dataset", "COVID", n=10)

        >>> # Traitement de la classe Normal avec structure personnalisée
        >>> process_class("/data/dataset", "Normal", n=3, image_dir="scans")

    Note:
        Cette fonction est un point d'entrée pratique pour tester rapidement
        l'augmentation sur différentes classes du dataset COVID-19.

    Workflow:
        1. Construction du chemin vers les images de la classe
        2. Création d'un répertoire temporaire
        3. Copie d'un échantillon d'images
        4. Configuration du générateur d'augmentation
        5. Visualisation des images augmentées
    """
    # Construction du chemin complet vers le dossier d'images de la classe
    images_dir = os.path.join(base_dir, cls, image_dir)

    # Récupération du répertoire parent pour la création du dossier temporaire
    parent_dir = os.path.dirname(images_dir)

    # Étape 1: Création du répertoire temporaire
    tmp_class_dir = create_tmp_dir(parent_dir)

    # Étape 2: Copie d'un échantillon d'images sans doublons
    copy_images_no_duplicates(images_dir, tmp_class_dir, n)

    # Étape 3: Configuration du générateur d'augmentation avec
    # paramètres par défaut
    generator = get_image_generator()

    # Étape 4: Visualisation des images augmentées
    show_augmented_images(generator, tmp_class_dir)
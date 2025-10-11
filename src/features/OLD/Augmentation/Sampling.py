"""Module de sous-échantillonnage pour équilibrer les datasets d'images médicales.

Ce module fournit des utilitaires pour réduire le nombre d'images dans des
classes sur-représentées afin d'équilibrer un dataset médical, particulièrement
utile pour les datasets COVID-19 où certaines classes peuvent être
déséquilibrées.

Auteur: [Votre nom]
Date: 2025
"""

import os
import random
import shutil
from glob import glob


def apply_undersampling(data_dir, target_count, output_dir):
    """Applique un sous-échantillonnage aléatoire sur un répertoire d'images.

    Cette fonction réduit le nombre d'images dans un répertoire en sélectionnant
    aléatoirement un sous-ensemble d'images jusqu'à atteindre le nombre cible.
    Utile pour équilibrer les classes dans un dataset médical où certaines
    classes sont sur-représentées.

    Args:
        data_dir (str): Chemin vers le répertoire source contenant les images
            au format PNG
        target_count (int): Nombre cible d'images à conserver après
            sous-échantillonnage
        output_dir (str): Chemin vers le répertoire de destination où copier
            les images sélectionnées

    Returns:
        list: Liste des messages de log décrivant le processus d'exécution

    Example:
        >>> logs = apply_undersampling(
        ...     "/data/COVID/images",
        ...     1000,
        ...     "/data/balanced/COVID"
        ... )
        >>> for log in logs:
        ...     print(log)
        Recherche dans : /data/COVID/images
        Images trouvées : 1500
        ✅ Undersampling terminé pour /data/COVID/images → /data/balanced/COVID

    Note:
        - Seules les images au format PNG sont prises en compte
        - Si le nombre d'images disponibles est inférieur au target_count,
          toutes les images sont copiées
        - Le répertoire de sortie est créé automatiquement s'il n'existe pas
    """
    import os, random, shutil
    from glob import glob

    logs = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = glob(os.path.join(data_dir, "*.png"))
    logs.append(f"Recherche dans : {data_dir}")
    logs.append(f"Images trouvées : {len(images)}")

    if len(images) > target_count:
        selected_images = random.sample(images, target_count)  # Rajouter une seed pour la reproductibilité
    else:
        selected_images = images

    for img_path in selected_images:
        shutil.copy(img_path, output_dir)

    logs.append(f"✅ Undersampling terminé pour {data_dir} → {output_dir}")
    return logs 


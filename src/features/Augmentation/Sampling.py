import os
import random
import shutil
from glob import glob


def apply_undersampling(data_dir, target_count, output_dir):
    import os, random, shutil
    from glob import glob

    logs = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = glob(os.path.join(data_dir, "*.png"))
    logs.append(f"Recherche dans : {data_dir}")
    logs.append(f"Images trouvées : {len(images)}")

    if len(images) > target_count:
        selected_images = random.sample(images, target_count) # Rajouter une seed pour la reproductibilité
    else:
        selected_images = images

    for img_path in selected_images:
        shutil.copy(img_path, output_dir)

    logs.append(f"✅ Undersampling terminé pour {data_dir} → {output_dir}")
    return logs 


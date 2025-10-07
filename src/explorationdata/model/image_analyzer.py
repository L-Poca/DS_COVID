import os
from PIL import Image


def collect_image_stats(image_folder):
    total_size = 0
    formats = set()
    color_modes = set()
    shapes = set()
    image_dimensions = {}
    image_names_set = set()
    logs = []

    for img_name in os.listdir(image_folder):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        try:
            name_only = os.path.splitext(img_name)[0]
            img_path = os.path.join(image_folder, img_name)
            im = Image.open(img_path)

            image_names_set.add(name_only)
            image_dimensions[name_only] = im.size
            total_size += os.path.getsize(img_path)
            formats.add(os.path.splitext(img_name)[1].lower())
            color_modes.add(im.mode)
            shapes.add(im.size)
        except Exception as e:
            logs.append(f"Erreur sur {img_name}: {e}")

    return {
        "names": image_names_set,
        "dimensions": image_dimensions,
        "formats": formats,
        "color_modes": color_modes,
        "shapes": shapes,
        "total_size": total_size,
        "logs": logs
    }

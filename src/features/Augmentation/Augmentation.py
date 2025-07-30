import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_tmp_dir(parent_dir):
    tmp_class_dir = os.path.join(parent_dir, "tmp")
    os.makedirs(tmp_class_dir, exist_ok=True)
    return tmp_class_dir

def copy_images_no_duplicates(images_dir, tmp_class_dir, n=5):
    class_name = os.path.basename(os.path.dirname(images_dir))
    class_dir = os.path.join(tmp_class_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    existing = set(os.listdir(class_dir))
    for img_name in os.listdir(images_dir)[:n]:
        if img_name not in existing:
            shutil.copy(os.path.join(images_dir, img_name), class_dir)

def get_image_generator(config=None):
    if config is None:
        config = {
            "rotation_range": 15,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
            "zoom_range": 0.1,
            "horizontal_flip": True,
            "brightness_range": [0.8, 1.2]
        }
    return ImageDataGenerator(
        rotation_range=config["rotation_range"],
        width_shift_range=config["width_shift_range"],
        height_shift_range=config["height_shift_range"],
        zoom_range=config["zoom_range"],
        horizontal_flip=config["horizontal_flip"],
        brightness_range=config["brightness_range"]
    )

def show_augmented_images(generator, tmp_class_dir, target_size=(256, 256), batch_size=5):
    gen = generator.flow_from_directory(
        tmp_class_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode=None,
        shuffle=True
    )
    batch = next(gen)
    plt.figure(figsize=(15, 3))
    for i in range(batch_size):
        plt.subplot(1, batch_size, i+1)
        plt.imshow(batch[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.suptitle("Batch d'images augmentées (Keras)")
    plt.show()


def process_class(base_dir, cls, n=5, Image_Dir="images"):
    images_dir = os.path.join(base_dir, cls, Image_Dir)
    parent_dir = os.path.dirname(images_dir)
    tmp_class_dir = create_tmp_dir(parent_dir)
    copy_images_no_duplicates(images_dir, tmp_class_dir, n)
    generator = get_image_generator()
    show_augmented_images(generator, tmp_class_dir)


print("OK")
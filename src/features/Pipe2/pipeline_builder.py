import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50

def build_pipeline(cfg):
    """Construit un modèle keras + pipeline preprocessing à partir du JSON."""
    # --- Augmentation ---
    aug = tf.keras.Sequential([
        layers.Resizing(*cfg["data"]["image_size"]),
        layers.Rescaling(1./255)
    ])
    if cfg["augmentation"].get("random_flip", False):
        aug.add(layers.RandomFlip("horizontal"))
    if cfg["augmentation"].get("random_rotation", 0):
        aug.add(layers.RandomRotation(cfg["augmentation"]["random_rotation"]))

    # --- Base model ---
    base_model_cls = {
        "EfficientNetB0": EfficientNetB0,
        "ResNet50": ResNet50
    }[cfg["model"]["base"]]

    base_model = base_model_cls(
        weights="imagenet",
        include_top=False,
        input_shape=(*cfg["data"]["image_size"], 3)
    )
    base_model.trainable = cfg["model"]["trainable"]

    # --- Full model ---
    model = models.Sequential([
        aug,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(cfg["model"]["dropout"]),
        layers.Dense(cfg["model"]["dense_units"], activation="relu"),
        layers.Dense(cfg["model"]["num_classes"], activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.get(cfg["training"]["optimizer"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
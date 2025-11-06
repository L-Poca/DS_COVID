import json
from pathlib import Path

def train_and_save(model, X_train, y_train, cfg):
    """Entraîne et sauvegarde le modèle + config utilisée."""
    history = model.fit(
        X_train, y_train,
        validation_split=cfg["training"]["validation_split"],
        epochs=cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        verbose=1
    )

    model_path = Path(cfg["output"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    # Sauvegarder la config utilisée
    meta_path = Path("models/metadata") / (model_path.stem + "_config_used.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"✅ Modèle sauvegardé : {model_path}")
    print(f"🧩 Config sauvegardée : {meta_path}")
    return history

def load_trained_model(model_path):
    """Recharge un modèle keras déjà entraîné."""
    return tf.keras.models.load_model(model_path)
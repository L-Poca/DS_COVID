import json
from pathlib import Path

def load_pipeline_config(pipeline_name: str, config_dir: str = "Configs_Pipelines"):
    """Charge un pipeline JSON depuis le dossier configs."""
    config_path = Path(config_dir) / pipeline_name
    if not config_path.exists():
        raise FileNotFoundError(f"❌ Fichier config introuvable: {config_path}")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return cfg
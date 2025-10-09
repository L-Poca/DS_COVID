# refactorisation/utils/load_grid_params.py
import json
from pathlib import Path


def load_grid_params(file_name="grid_search_params.json"):
    ROOT = Path(__file__).resolve().parents[1]
    path = ROOT / "config" / file_name
    if not path.exists():
        raise FileNotFoundError(f"Fichier grid search introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    return params

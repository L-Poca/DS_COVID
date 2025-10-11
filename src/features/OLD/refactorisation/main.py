# refactorisation/main.py
import os
from utils.data_loader import load_dataset
from controllers.trainer import (
    split_data,
    train_with_grid_search,
    evaluate_model
)
from models.baseline import RandomForest, LinearSVM
from views.word_generator import generate_word_report
from utils.load_grid_params import load_grid_params

if __name__ == "__main__":
    print("\n[INFO] Lancement pipeline refactorisé...\n")

    # Charger dataset (utilise config/settings.json et .env par défaut)
    X, y = load_dataset()  # options: load_dataset(img_size=(128,128), samples_per_class=200, ...)

    # Split train/test
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Charger les paramètres du Grid Search
    grid_params = load_grid_params()

    results = []

    # Boucle sur les modèles
    for model_cls in [RandomForest, LinearSVM]:
        model = model_cls()
        param_grid = grid_params.get(model_cls.__name__, None)

        # Entraînement (avec ou sans Grid Search selon param_grid)
        trained_model = train_with_grid_search(model, param_grid, X_train, y_train)

        # Évaluation + matrice de confusion
        result = evaluate_model(
            trained_model,
            X_test,
            y_test,
            model_name=model_cls.__name__
        )
        results.append(result)

    # Générer le rapport Word
    generate_word_report(results)

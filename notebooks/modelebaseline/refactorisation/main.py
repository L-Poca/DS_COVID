import os
from notebooks.modelebaseline.refactorisation.utils.data_loader import load_dataset
from notebooks.modelebaseline.refactorisation.controllers.trainer import (
    split_data,
    train_with_grid_search,
    evaluate_model
)
from notebooks.modelebaseline.refactorisation.models.baseline import (
    RandomForest,
    LinearSVM,
    AdaBoost,
    GradientBoosting
)
from notebooks.modelebaseline.refactorisation.views.word_generator import generate_word_report
from notebooks.modelebaseline.refactorisation.utils.load_grid_params import load_grid_params


if __name__ == "__main__":
    print("[INFO] Lancement pipeline refactorisé...")

    # Charger dataset
    X, y = load_dataset()

    # Split train/test
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Charger les paramètres du Grid Search
    grid_params = load_grid_params()

    results = []

    # Boucle sur les modèles
    for model_func in [RandomForest, LinearSVM, AdaBoost, GradientBoosting]:
        model = model_func()  # Chaque fonction retourne un modèle sklearn
        model_name = model.__class__.__name__
        param_grid = grid_params.get(model_name, None)

        print(f"\n[INFO] ======= Entraînement du modèle : {model_name} =======")

        trained_model = train_with_grid_search(model, param_grid, X_train, y_train)

        result = evaluate_model(
            trained_model,
            X_test,
            y_test,
            model_name=model_name
        )
        results.append(result)

    # Générer le rapport Word
    print("\n[INFO] Génération du rapport Word final...")
    generate_word_report(results)
    print("[INFO] Pipeline terminé avec succès ✅")

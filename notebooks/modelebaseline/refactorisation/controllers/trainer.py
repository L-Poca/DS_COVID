# refactorisation/controllers/trainer.py
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def split_data(X, y, test_size=0.2, seed=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed
    )


def train_model(model, X_train, y_train):
    """Entraîne le modèle simple (sans Grid Search)."""
    model.fit(X_train, y_train)
    return model


def train_with_grid_search(model, param_grid, X_train, y_train, cv=3, scoring="accuracy"):
    """Entraîne le modèle avec Grid Search si param_grid est fourni."""
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        print(f"[INFO] Meilleurs paramètres pour {model.__class__.__name__} : {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        # Si pas de param_grid, entraîne normalement
        return train_model(model, X_train, y_train)


def evaluate_model(
        model,
        X_test,
        y_test,
        model_name,
        output_folder="rapports"
    ):
    """Évalue le modèle et génère la matrice de confusion."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    # matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    cm_path = os.path.join(output_folder, f"cm_{model_name}.png")
    plt.savefig(cm_path)
    plt.close()

    return {
        "name": model_name,
        "report": report,
        "confusion_matrix_path": cm_path
    }

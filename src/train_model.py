from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Définit les chemins vers les données d'entrée et les sorties
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
METRICS_DIR = Path("evaluation/metrics")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# Charge les datasets prétraités nécessaires
def load_data():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    X_train_scaled = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
    X_test_scaled = pd.read_csv(PROCESSED_DIR / "X_test_scaled.csv")

    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


# Calcule les métriques principales d'un modèle
def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }


# Définit les modèles à comparer dans le projet
def build_models():
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=7,
        ),
    }
    return models


# Entraîne chaque modèle sur le bon type de données puis l'évalue
def train_and_compare_models(
    models,
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
):
    results = {}
    trained_models = {}

    for model_name, model in models.items():
        # Choisit les données scaled pour les modèles sensibles à l'échelle
        if model_name in ["logistic_regression", "knn"]:
            X_train_used = X_train_scaled
            X_test_used = X_test_scaled
        else:
            X_train_used = X_train
            X_test_used = X_test

        # Entraîne le modèle sur les données d'entraînement
        model.fit(X_train_used, y_train)

        # Génère les prédictions sur le jeu de test
        y_pred = model.predict(X_test_used)

        # Calcule les métriques d'évaluation
        metrics = evaluate_predictions(y_test, y_pred)

        # Stocke les résultats et le modèle entraîné
        results[model_name] = metrics
        trained_models[model_name] = model

        print(f"\n=== {model_name} ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    return results, trained_models


# Sélectionne le meilleur modèle selon le score F1
def select_best_model(results):
    best_model_name = max(results, key=lambda name: results[name]["f1_score"])
    return best_model_name


# Sauvegarde tous les scores dans un fichier JSON
def save_metrics(results):
    with open(METRICS_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


# Sauvegarde le meilleur modèle entraîné dans le dossier models
def save_best_model(best_model_name, trained_models):
    best_model = trained_models[best_model_name]
    model_path = MODELS_DIR / f"{best_model_name}.pkl"
    joblib.dump(best_model, model_path)
    return model_path


# Exécute tout le pipeline d'entraînement et de comparaison
def main():
    # Charge les données déjà prétraitées
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = load_data()

    # Construit les modèles à comparer
    models = build_models()

    # Entraîne les modèles et calcule leurs performances
    results, trained_models = train_and_compare_models(
        models,
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )

    # Sauvegarde les métriques de comparaison
    save_metrics(results)

    # Choisit le meilleur modèle selon F1-score
    best_model_name = select_best_model(results)

    # Sauvegarde le meilleur modèle
    model_path = save_best_model(best_model_name, trained_models)

    print("\n=== Best model ===")
    print(f"Selected model: {best_model_name}")
    print(f"Saved at: {model_path}")


# Lance le script seulement s'il est exécuté directement
if __name__ == "__main__":
    main()
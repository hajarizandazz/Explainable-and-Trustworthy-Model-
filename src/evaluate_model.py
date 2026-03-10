from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# Définit les chemins d'entrée et de sortie
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
METRICS_DIR = Path("evaluation/metrics")
FIGURES_DIR = Path("evaluation/figures")

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Vérifie que les fichiers utiles existent
def check_required_files() -> None:
    required_files = [
        PROCESSED_DIR / "X_test.csv",
        PROCESSED_DIR / "X_test_scaled.csv",
        PROCESSED_DIR / "y_test.csv",
        METRICS_DIR / "model_comparison.json",
    ]

    missing_files = [file for file in required_files if not file.exists()]

    if missing_files:
        missing_str = "\n".join(str(file) for file in missing_files)
        raise FileNotFoundError(
            "Some required files are missing. Run preprocess.py and train_model.py first.\n"
            f"Missing files:\n{missing_str}"
        )


# Charge les résultats de comparaison des modèles
def load_model_comparison() -> dict:
    with open(METRICS_DIR / "model_comparison.json", "r", encoding="utf-8") as f:
        return json.load(f)


# Récupère le nom du meilleur modèle
def get_best_model_name(results: dict) -> str:
    if "best_model" in results:
        return results["best_model"]

    valid_models = {
        key: value for key, value in results.items() if isinstance(value, dict) and "f1_score" in value
    }
    return max(valid_models, key=lambda name: valid_models[name]["f1_score"])


# Charge le meilleur modèle sauvegardé
def load_best_model(best_model_name: str):
    model_path = MODELS_DIR / f"{best_model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


# Charge les données de test utiles
def load_test_data():
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    X_test_scaled = pd.read_csv(PROCESSED_DIR / "X_test_scaled.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    return X_test, X_test_scaled, y_test


# Choisit les bonnes données selon le type de modèle
def select_test_data_for_model(best_model_name: str, X_test: pd.DataFrame, X_test_scaled: pd.DataFrame):
    if best_model_name in ["logistic_regression", "knn"]:
        return X_test_scaled
    return X_test


# Calcule les métriques d'évaluation
def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }


# Sauvegarde les métriques dans un fichier JSON
def save_evaluation_metrics(best_model_name: str, metrics: dict, report: dict) -> None:
    output = {
        "best_model": best_model_name,
        "metrics": metrics,
        "classification_report": report,
    }

    with open(METRICS_DIR / "best_model_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)


# Génère et sauvegarde la matrice de confusion
def save_confusion_matrix(y_true, y_pred, best_model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax)
    ax.set_title(f"Confusion Matrix - {best_model_name}")

    plt.savefig(FIGURES_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close()


# Exécute l'évaluation complète du meilleur modèle
def main() -> None:
    # Vérifie les fichiers nécessaires
    check_required_files()

    # Charge les résultats des modèles comparés
    results = load_model_comparison()

    # Récupère le nom du meilleur modèle
    best_model_name = get_best_model_name(results)

    # Charge le meilleur modèle
    model = load_best_model(best_model_name)

    # Charge les données de test
    X_test, X_test_scaled, y_test = load_test_data()

    # Choisit les bonnes données de test selon le modèle
    X_test_used = select_test_data_for_model(best_model_name, X_test, X_test_scaled)

    # Prédit sur le jeu de test
    y_pred = model.predict(X_test_used)

    # Calcule les métriques principales
    metrics = compute_metrics(y_test, y_pred)

    # Génère le rapport de classification
    report = classification_report(y_test, y_pred, output_dict=True)

    # Sauvegarde les résultats
    save_evaluation_metrics(best_model_name, metrics, report)
    save_confusion_matrix(y_test, y_pred, best_model_name)

    # Affiche les résultats principaux
    print("\n=== Best model evaluation ===")
    print(f"Model: {best_model_name}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print("\nEvaluation files saved in:")
    print(METRICS_DIR)
    print(FIGURES_DIR)


# Lance le script seulement s'il est exécuté directement
if __name__ == "__main__":
    main()
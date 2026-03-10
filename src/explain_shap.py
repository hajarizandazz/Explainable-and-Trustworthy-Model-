from pathlib import Path
import json
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


# Définit les chemins principaux
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
METRICS_DIR = Path("evaluation/metrics")
FIGURES_DIR = Path("evaluation/figures")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Vérifie que les fichiers utiles existent
def check_required_files() -> None:
    required_files = [
        PROCESSED_DIR / "X_train.csv",
        PROCESSED_DIR / "X_test.csv",
        PROCESSED_DIR / "X_train_scaled.csv",
        PROCESSED_DIR / "X_test_scaled.csv",
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


# Charge les données adaptées au modèle
def load_shap_data(best_model_name: str):
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    X_train_scaled = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
    X_test_scaled = pd.read_csv(PROCESSED_DIR / "X_test_scaled.csv")

    if best_model_name in ["logistic_regression", "knn"]:
        return X_train_scaled, X_test_scaled
    return X_train, X_test


# Crée l'explainer SHAP
def create_shap_explainer(model, X_train: pd.DataFrame):
    return shap.Explainer(model, X_train)


# Sauvegarde le summary plot global
def save_shap_summary_plot(shap_values, X_test: pd.DataFrame) -> None:
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(FIGURES_DIR / "shap_summary.png", bbox_inches="tight")
    plt.close()


# Sauvegarde une explication locale pour un patient
def save_shap_waterfall_plot(shap_values, instance_index: int) -> None:
    plt.figure()

    # Sélectionne uniquement la classe positive (Diabetes)
    single_explanation = shap_values[instance_index, :, 1]

    shap.plots.waterfall(single_explanation, show=False)
    plt.savefig(FIGURES_DIR / f"shap_waterfall_{instance_index}.png", bbox_inches="tight")
    plt.close()


# Exécute la génération des explications SHAP
def main() -> None:
    # Vérifie les fichiers nécessaires
    check_required_files()

    # Charge les résultats des modèles
    results = load_model_comparison()

    # Récupère le nom du meilleur modèle
    best_model_name = get_best_model_name(results)

    # Charge le meilleur modèle
    model = load_best_model(best_model_name)

    # Charge les données adaptées
    X_train_used, X_test_used = load_shap_data(best_model_name)

    # Réduit la taille pour aller plus vite
    X_test_sample = X_test_used.iloc[:50].copy()

    # Crée l'explainer et calcule les valeurs SHAP
    explainer = create_shap_explainer(model, X_train_used)
    shap_values = explainer(X_test_sample)

    # Sauvegarde l'explication globale
    save_shap_summary_plot(shap_values, X_test_sample)

    # Sauvegarde deux explications locales
    save_shap_waterfall_plot(shap_values, instance_index=0)
    save_shap_waterfall_plot(shap_values, instance_index=1)

    print("\n=== SHAP explanations completed ===")
    print(f"Best model used: {best_model_name}")
    print(f"Files saved in: {FIGURES_DIR}")


# Lance le script seulement s'il est exécuté directement
if __name__ == "__main__":
    main()
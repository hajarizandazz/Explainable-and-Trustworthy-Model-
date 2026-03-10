from pathlib import Path
import json
import joblib
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer


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


# Charge les données utiles pour LIME
def load_lime_data(best_model_name: str):
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    X_train_scaled = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
    X_test_scaled = pd.read_csv(PROCESSED_DIR / "X_test_scaled.csv")

    if best_model_name in ["logistic_regression", "knn"]:
        return X_train_scaled, X_test_scaled
    return X_train, X_test


# Crée l'explainer LIME tabulaire
def create_lime_explainer(X_train: pd.DataFrame) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["No Diabetes", "Diabetes"],
        mode="classification",
    )


# Génère et sauvegarde une explication LIME
def generate_lime_explanation(explainer, model, X_test: pd.DataFrame, instance_index: int) -> None:
    instance = X_test.iloc[instance_index].values

    def predict_with_feature_names(data):
        data_df = pd.DataFrame(data, columns=X_test.columns)
        return model.predict_proba(data_df)

    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_with_feature_names,
        num_features=8,
    )

    output_file = FIGURES_DIR / f"lime_explanation_{instance_index}.html"
    explanation.save_to_file(str(output_file))

    print(f"LIME explanation saved: {output_file}")


# Exécute la génération des explications LIME
def main() -> None:
    # Vérifie les fichiers nécessaires
    check_required_files()

    # Charge les résultats des modèles
    results = load_model_comparison()

    # Récupère le nom du meilleur modèle
    best_model_name = get_best_model_name(results)

    # Charge le meilleur modèle
    model = load_best_model(best_model_name)

    # Charge les données adaptées au modèle
    X_train_used, X_test_used = load_lime_data(best_model_name)

    # Crée l'explainer LIME
    explainer = create_lime_explainer(X_train_used)

    # Génère deux explications locales
    generate_lime_explanation(explainer, model, X_test_used, instance_index=0)
    generate_lime_explanation(explainer, model, X_test_used, instance_index=1)

    print("\n=== LIME explanations completed ===")
    print(f"Best model used: {best_model_name}")


# Lance le script seulement s'il est exécuté directement
if __name__ == "__main__":
    main()
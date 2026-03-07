from pathlib import Path
import pandas as pd
import numpy as np 
## pour remplir les valeurs manquantes 
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# les chemins principaux du projet
RAW_DATA_PATH = Path("data/raw/diabetes.csv")
PROCESSED_DIR = Path("data/processed")
SCALER_PATH = Path("models/scaler.pkl")


# Crée les dossiers de sortie si nécessaire
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)


# Charge le dataset brut depuis le fichier CSV
def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    return pd.read_csv(csv_path)


# Affiche un résumé simple avant preprocessing
def display_basic_info(df: pd.DataFrame) -> None:
    print("\n=== Basic dataset info ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumns:")
    print(list(df.columns))

    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

# Affiche les valeurs manquantes après le nettoyage
def display_missing_after_cleaning(df: pd.DataFrame) -> None:
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

# Remplace les zéros non réalistes par des valeurs manquantes
def replace_invalid_zeros(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    zero_invalid_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    print("\nInvalid zeros replaced with NaN:")
    for col in zero_invalid_columns:
        zero_count = (df[col] == 0).sum()
        print(f"{col}: {zero_count}")
        df[col] = df[col].replace(0, np.nan)

    return df


# Impute les valeurs manquantes avec la médiane de chaque colonne
def impute_missing_values(X_train: pd.DataFrame, X_test: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")

    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_imputed, X_test_imputed, imputer


# Standardise les variables numériques ( features ) pour les modèles sensibles à l’échelle ( moyenne 0 et l'écart type 1 )
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, scaler


# Sauvegarde les datasets traités dans le dossier data/processed
def save_processed_data(
    df_cleaned: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
) -> None:
    df_cleaned.to_csv(PROCESSED_DIR / "diabetes_cleaned.csv", index=False)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    X_train_scaled.to_csv(PROCESSED_DIR / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(PROCESSED_DIR / "X_test_scaled.csv", index=False)


# Affiche un résumé simple après preprocessing
def display_processed_info(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    print("\n=== Processed data info ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("\nClass distribution in y_train (%):")
    print((y_train.value_counts(normalize=True) * 100).round(2))

    print("\nClass distribution in y_test (%):")
    print((y_test.value_counts(normalize=True) * 100).round(2))


# Exécute tout le pipeline de preprocessing
def main() -> None:
    # Charge les données brutes
    df = load_dataset(RAW_DATA_PATH)

    # Affiche les infos de départ
    display_basic_info(df)

    # Nettoie les zéros incohérents
    df = replace_invalid_zeros(df)
    display_missing_after_cleaning(df)

    # Sépare les variables explicatives et la cible
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Fait un split train/test en gardant la proportion des classes
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Impute les valeurs manquantes à partir du train uniquement
    X_train_imputed, X_test_imputed, imputer = impute_missing_values(X_train, X_test)

    # Standardise les données à partir du train uniquement
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train_imputed,
        X_test_imputed,
    )

    # Sauvegarde le scaler pour réutilisation plus tard
    joblib.dump(scaler, SCALER_PATH)

    # Sauvegarde aussi l’imputer si besoin futur
    joblib.dump(imputer, PROCESSED_DIR / "imputer.pkl")

    # Sauvegarde toutes les données traitées
    save_processed_data(
        df,
        X_train_imputed,
        X_test_imputed,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
    )

    # Affiche un résumé final
    display_processed_info(
        X_train_imputed,
        X_test_imputed,
        y_train,
        y_test,
    )

    print("\nPreprocessing completed successfully.")
    print(f"Processed files saved in: {PROCESSED_DIR}")


# Lance le script seulement s’il est exécuté directement
if __name__ == "__main__":
    main()
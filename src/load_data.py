from pathlib import Path
import pandas as pd


# le chemin vers le fichier CSV dans le dossier data/raw
DATA_PATH = Path("data/raw/diabetes.csv")


# Charge le dataset depuis le fichier CSV
def load_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    return pd.read_csv(csv_path)

# Affiche les informations principales du dataset
def display_dataset_info(df: pd.DataFrame) -> None:
    print("\n=== Dataset shape ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n=== Column names ===")
    print(list(df.columns))

    print("\n=== First 5 rows ===")
    print(df.head())


# Point d'entrée principal du script
def main() -> None:
    df = load_dataset(DATA_PATH)
    display_dataset_info(df)


# Lance le script seulement s'il est exécuté directement
if __name__ == "__main__":
    main()
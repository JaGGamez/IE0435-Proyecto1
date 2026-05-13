import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

# Best hyperparameters found during GridSearchCV
BEST_PARAMS = {
    "criterion":    "entropy",
    "max_depth":    10,
    "n_estimators": 200,
    "random_state": 42,
    "n_jobs":       -1,
}

CSV_FOLDER  = "All_csv"
OUTPUT_FILE = "B83417_Jacob_Gonzalez.joblib"


def load_datasets(folder: str) -> pd.DataFrame:
    dfs = []

    named_label = {
        "dataset (2).csv":          "label",
        "dataset_3_3.csv":          "contaminacion",
        "dataset_clasificacion.csv": "label",
        "dataset_con_etiquetas.csv": "etiqueta",
        "matriz_final.csv":         "etiqueta_arroz",
    }
    for fname, label_col in named_label.items():
        df = pd.read_csv(os.path.join(folder, fname))
        df = df.rename(columns={label_col: "label"})
        df = df.iloc[:, :16384].assign(label=df["label"])
        dfs.append(df)

    df = pd.read_csv(os.path.join(folder, "dataset_imagenes.csv"), sep=";")
    df = df.rename(columns={"etiqueta": "label"})
    df = df.iloc[:, :16384].assign(label=df["label"])
    dfs.append(df)

    for fname in ["dataset (1).csv", "dataset.csv", "dataset_C26797.csv",
                  "dataset_daniel_valverde.csv", "matriz_final (1).csv"]:
        df = pd.read_csv(os.path.join(folder, fname), header=None)
        df.columns = [f"pixel_{i}" for i in range(df.shape[1] - 1)] + ["label"]
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True).drop_duplicates()
    return combined


def main():
    print("Loading full dataset...")
    data = load_datasets(CSV_FOLDER)
    X = data.iloc[:, :16384].values.astype(np.uint8)
    y = data["label"].values.astype(np.uint8)
    print(f"  Samples: {len(X)}  |  Class 0: {int(np.sum(y==0))}  Class 1: {int(np.sum(y==1))}")

    print(f"Training Random Forest on full dataset with best params: {BEST_PARAMS}")
    model = Pipeline([
        ("vt",  VarianceThreshold(threshold=0.0)),
        ("clf", RandomForestClassifier(**BEST_PARAMS)),
    ])
    model.fit(X, y)

    joblib.dump(model, OUTPUT_FILE)
    size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"Model saved as '{OUTPUT_FILE}' ({size_kb:.1f} KB)")
    print("Done.")


if __name__ == "__main__":
    main()

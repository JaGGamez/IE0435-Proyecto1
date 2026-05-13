import os
import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time

warnings.filterwarnings("ignore")

CSV_FOLDER = "All_csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# 1. Load and unify all datasets
# ---------------------------------------------------------------------------

def load_datasets(folder: str) -> pd.DataFrame:
    dfs = []

    # --- Files with proper headers and last column as label ---
    named_label = {
        "dataset (2).csv":          "label",
        "dataset_3_3.csv":          "contaminacion",
        "dataset_clasificacion.csv": "label",
        "dataset_con_etiquetas.csv": "etiqueta",
        "matriz_final.csv":         "etiqueta_arroz",
    }
    for fname, label_col in named_label.items():
        path = os.path.join(folder, fname)
        df = pd.read_csv(path)
        df = df.rename(columns={label_col: "label"})
        # keep only 16384 pixel columns + label
        df = df.iloc[:, :16384].assign(label=df["label"])
        dfs.append(df)

    # --- Semicolon-delimited file ---
    df = pd.read_csv(os.path.join(folder, "dataset_imagenes.csv"), sep=";")
    df = df.rename(columns={"etiqueta": "label"})
    df = df.iloc[:, :16384].assign(label=df["label"])
    dfs.append(df)

    # --- Files without header (first row is data) ---
    no_header_files = [
        "dataset (1).csv",
        "dataset.csv",
        "dataset_C26797.csv",
        "dataset_daniel_valverde.csv",
        "matriz_final (1).csv",
    ]
    for fname in no_header_files:
        path = os.path.join(folder, fname)
        df = pd.read_csv(path, header=None)
        df.columns = [f"pixel_{i}" for i in range(df.shape[1] - 1)] + ["label"]
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Drop exact duplicates
    before = len(combined)
    combined = combined.drop_duplicates()
    after = len(combined)
    print(f"  Loaded {before} rows total, {before - after} duplicates removed -> {after} unique samples")

    return combined


# ---------------------------------------------------------------------------
# 2. Evaluate a model with GridSearchCV and return a results dict
# ---------------------------------------------------------------------------

def evaluate(name: str, pipeline: Pipeline, param_grid: dict,
             X_train, X_test, y_train, y_test) -> dict:

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1, refit=True)

    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = grid.predict(X_test)

    result = {
        "Model":       name,
        "Best params": grid.best_params_,
        "CV F1":       round(grid.best_score_, 4),
        "Accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "Precision":   round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":      round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1 (test)":   round(f1_score(y_test, y_pred, zero_division=0), 4),
        "Time (s)":    round(elapsed, 1),
        "Confusion":   confusion_matrix(y_test, y_pred).tolist(),
        "Report":      classification_report(y_test, y_pred, target_names=["Negativo(0)", "Positivo(1)"]),
    }
    return result


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)
    data = load_datasets(CSV_FOLDER)

    X = data.iloc[:, :16384].values.astype(np.uint8)
    y = data["label"].values.astype(np.uint8)
    print(f"  Features: {X.shape[1]}  |  Samples: {X.shape[0]}")
    print(f"  Class distribution -> 0: {int(np.sum(y==0))}  1: {int(np.sum(y==1))}")

    # Train/test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train: {len(X_train)}  Test: {len(X_test)}\n")

    # VarianceThreshold removes pixels that never change across training samples
    vt = VarianceThreshold(threshold=0.0)
    X_train_vt = vt.fit_transform(X_train)
    X_test_vt  = vt.transform(X_test)
    print(f"  Features after VarianceThreshold: {X_train_vt.shape[1]}")

    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    results = []

    # --- Decision Tree ---
    print("\n[1/4] Decision Tree...")
    pipe_dt = Pipeline([("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))])
    grid_dt = {
        "clf__criterion":        ["gini", "entropy"],
        "clf__max_depth":        [3, 5, 10, 20, None],
        "clf__min_samples_split": [2, 5, 10],
    }
    results.append(evaluate("Decision Tree", pipe_dt, grid_dt,
                             X_train_vt, X_test_vt, y_train, y_test))
    print(f"  Best: {results[-1]['Best params']}  |  Test F1: {results[-1]['F1 (test)']}")

    # --- Random Forest ---
    print("\n[2/5] Random Forest...")
    pipe_rf = Pipeline([("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))])
    grid_rf = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth":    [5, 10, 20, None],
        "clf__criterion":    ["gini", "entropy"],
    }
    results.append(evaluate("Random Forest", pipe_rf, grid_rf,
                             X_train_vt, X_test_vt, y_train, y_test))
    print(f"  Best: {results[-1]['Best params']}  |  Test F1: {results[-1]['F1 (test)']}")

    # --- Naive Bayes (Bernoulli, ideal for binary features) ---
    print("\n[3/5] Bernoulli Naive Bayes...")
    pipe_nb = Pipeline([("clf", BernoulliNB())])
    grid_nb = {
        "clf__alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
    }
    results.append(evaluate("Naive Bayes (Bernoulli)", pipe_nb, grid_nb,
                             X_train_vt, X_test_vt, y_train, y_test))
    print(f"  Best: {results[-1]['Best params']}  |  Test F1: {results[-1]['F1 (test)']}")

    # --- KNN ---
    print("\n[4/5] K-Nearest Neighbors...")
    pipe_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    KNeighborsClassifier()),
    ])
    grid_knn = {
        "clf__n_neighbors": [3, 5, 7, 11, 15],
        "clf__weights":     ["uniform", "distance"],
        "clf__metric":      ["euclidean", "manhattan"],
    }
    results.append(evaluate("KNN", pipe_knn, grid_knn,
                             X_train_vt, X_test_vt, y_train, y_test))
    print(f"  Best: {results[-1]['Best params']}  |  Test F1: {results[-1]['F1 (test)']}")

    # --- SVM (Linear, efficient for high-dimensional binary data) ---
    print("\n[5/5] SVM (LinearSVC)...")
    pipe_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LinearSVC(random_state=RANDOM_STATE, max_iter=5000)),
    ])
    grid_svm = {
        "clf__C":   [0.001, 0.01, 0.1, 1.0, 10.0],
        "clf__loss": ["hinge", "squared_hinge"],
    }
    results.append(evaluate("SVM (Linear)", pipe_svm, grid_svm,
                             X_train_vt, X_test_vt, y_train, y_test))
    print(f"  Best: {results[-1]['Best params']}  |  Test F1: {results[-1]['F1 (test)']}")

    # ---------------------------------------------------------------------------
    # 4. Summary table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    summary = pd.DataFrame([{
        "Model":     r["Model"],
        "Accuracy":  r["Accuracy"],
        "Precision": r["Precision"],
        "Recall":    r["Recall"],
        "F1 (test)": r["F1 (test)"],
        "CV F1":     r["CV F1"],
        "Time (s)":  r["Time (s)"],
    } for r in results])

    summary = summary.sort_values("F1 (test)", ascending=False).reset_index(drop=True)
    print(summary.to_string(index=False))

    best = summary.iloc[0]["Model"]
    print(f"\nBest model: {best}")

    # Detailed reports
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORTS")
    print("=" * 60)
    for r in results:
        print(f"\n--- {r['Model']} ---")
        print(f"Best hyperparameters: {r['Best params']}")
        print(f"Confusion matrix:\n{np.array(r['Confusion'])}")
        print(r["Report"])

    # Save summary CSV
    summary.to_csv("classification_results.csv", index=False)
    print("Summary saved to 'classification_results.csv'")


if __name__ == "__main__":
    main()

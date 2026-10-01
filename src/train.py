from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import KMeansPrototypeClassifier, KMeansTimeRegressor
from .prepare_data import FEATURE_COLUMNS, ROOT, build_dataset

warnings.filterwarnings("ignore")

MODELS = ROOT / "models"
FIGURES = ROOT / "results" / "figures"
METRICS = ROOT / "results" / "metrics"


def split_classification(df):
    X = df[FEATURE_COLUMNS]
    y = df["erupting"].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=1 / 3, stratify=y_temp, random_state=42
    )
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def split_regression(df):
    df = df.dropna(subset=["time_to_eruption_hours"]).copy()
    X = df[FEATURE_COLUMNS]
    y = df["time_to_eruption_hours"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=1 / 3, random_state=42
    )
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def classification_models():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000, class_weight="balanced"
            )),
        ]),
        "kmeans": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KMeansPrototypeClassifier(
                n_clusters_per_class=8
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=15,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "neural_network": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(128, 128, 128, 128),
                activation="relu",
                alpha=1e-4,
                batch_size=50,
                learning_rate_init=1e-3,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
            )),
        ]),
    }


def regression_models():
    return {
        "kmeans": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KMeansTimeRegressor(n_clusters=10)),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=15,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "neural_network": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(128, 128, 128, 128),
                activation="relu",
                alpha=1e-4,
                batch_size=50,
                learning_rate_init=1e-3,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
            )),
        ]),
    }


def evaluate_classification(name, model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    return {
        "model": name,
        "accuracy": float(accuracy_score(y, pred)),
        "kappa": float(cohen_kappa_score(y, pred)),
        "auroc": float(roc_auc_score(y, proba)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classification_report": classification_report(
            y, pred, output_dict=True
        ),
    }


def evaluate_regression(name, model, X, y):
    pred = model.predict(X)
    return {
        "model": name,
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
    }


def plot_confusion_matrix(name, cm):
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name.replace('_', ' ').title()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}_confusion_matrix.png", dpi=180)
    plt.close()


def plot_classification_metrics(results):
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["kappa"], width, label="Cohen's Kappa")
    ax.bar(x + width / 2, df["auroc"], width, label="AUROC")
    ax.set_xticks(
        x, [m.replace("_", " ").title() for m in df["model"]]
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Classification Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "classification_comparison.png", dpi=180)
    plt.close()


def plot_regression_metrics(results):
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["rmse"], width=0.36, label="RMSE")
    ax.bar(x + 0.18, df["r2"], width=0.36, label="R²")
    ax.set_xticks(
        x, [m.replace("_", " ").title() for m in df["model"]]
    )
    ax.set_ylabel("Metric value")
    ax.set_title("Regression Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "regression_comparison.png", dpi=180)
    plt.close()


def plot_feature_importance(rf_model):
    importances = rf_model.feature_importances_
    df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": importances,
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(df["feature"], df["importance"])
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_importance.png", dpi=180)
    plt.close()


def plot_regression_scatter(model, X_test, y_test):
    pred = model.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred, alpha=0.45)
    lo = min(float(y_test.min()), float(pred.min()))
    hi = max(float(y_test.max()), float(pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Observed time to eruption (hours)")
    plt.ylabel("Predicted time to eruption (hours)")
    plt.title("Observed vs Predicted Time to Eruption")
    plt.tight_layout()
    plt.savefig(FIGURES / "observed_vs_predicted.png", dpi=180)
    plt.close()


def main():
    for directory in [MODELS, FIGURES, METRICS]:
        directory.mkdir(parents=True, exist_ok=True)

    df = build_dataset()

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_classification(df)
    cls_results = []

    for name, model in classification_models().items():
        model.fit(X_train, y_train)
        dev = evaluate_classification(name, model, X_dev, y_dev)
        test = evaluate_classification(name, model, X_test, y_test)

        cls_results.append({
            "model": name,
            "dev_kappa": dev["kappa"],
            "dev_auroc": dev["auroc"],
            "test_kappa": test["kappa"],
            "test_auroc": test["auroc"],
            "test_accuracy": test["accuracy"],
        })

        plot_confusion_matrix(
            name, np.array(test["confusion_matrix"])
        )
        joblib.dump(
            model, MODELS / f"{name}_classifier.joblib"
        )

    with open(
        METRICS / "classification_metrics.json", "w", encoding="utf-8"
    ) as f:
        json.dump(cls_results, f, indent=2)

    plot_classification_metrics([
        {
            "model": r["model"],
            "kappa": r["test_kappa"],
            "auroc": r["test_auroc"],
        }
        for r in cls_results
    ])

    rf_classifier = joblib.load(
        MODELS / "random_forest_classifier.joblib"
    )
    plot_feature_importance(rf_classifier)

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_regression(df)
    reg_results = []

    for name, model in regression_models().items():
        model.fit(X_train, y_train)
        dev = evaluate_regression(name, model, X_dev, y_dev)
        test = evaluate_regression(name, model, X_test, y_test)

        reg_results.append({
            "model": name,
            "dev_rmse": dev["rmse"],
            "dev_r2": dev["r2"],
            "test_rmse": test["rmse"],
            "test_r2": test["r2"],
        })

        joblib.dump(
            model, MODELS / f"{name}_regressor.joblib"
        )

    with open(
        METRICS / "regression_metrics.json", "w", encoding="utf-8"
    ) as f:
        json.dump(reg_results, f, indent=2)

    plot_regression_metrics([
        {
            "model": r["model"],
            "rmse": r["test_rmse"],
            "r2": r["test_r2"],
        }
        for r in reg_results
    ])

    rf_regressor = joblib.load(
        MODELS / "random_forest_regressor.joblib"
    )
    plot_regression_scatter(rf_regressor, X_test, y_test)

    summary = {
        "rows": int(len(df)),
        "positive_labels": int(df["erupting"].sum()),
        "negative_labels": int((df["erupting"] == 0).sum()),
        "classification_models": cls_results,
        "regression_models": reg_results,
    }

    with open(
        METRICS / "summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print()
    print("Training complete.")
    print(pd.DataFrame(cls_results).to_string(index=False))
    print()
    print(pd.DataFrame(reg_results).to_string(index=False))
    print()
    print(f"Artifacts saved under: {MODELS}")
    print(f"Figures saved under: {FIGURES}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from sklearn.model_selection import (
    train_test_split,
)

from sklearn.neural_network import (
    MLPClassifier,
    MLPRegressor,
)

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.preprocessing import (
    StandardScaler,
)

from .models import (
    KMeansPrototypeClassifier,
    KMeansTimeRegressor,
)

from .prepare_data import (
    FEATURE_COLUMNS,
    ROOT,
    build_dataset,
)


warnings.filterwarnings(
    "ignore"
)


MODELS = (
    ROOT
    / "models"
)

FIGURES = (
    ROOT
    / "results"
    / "figures"
)

METRICS = (
    ROOT
    / "results"
    / "metrics"
)


# ============================================================
# DATA SPLITTING
# ============================================================

def split_classification_data(df):

    X = df[
        FEATURE_COLUMNS
    ]

    y = (
        df["erupting"]
        .astype(int)
    )

    # 70% training
    # 30% temporary
    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            stratify=y,
            random_state=42,
        )
    )

    # Of the remaining 30%:
    # 20% total -> development
    # 10% total -> test
    X_dev, X_test, y_dev, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=1 / 3,
            stratify=y_temp,
            random_state=42,
        )
    )

    return (
        X_train,
        X_dev,
        X_test,
        y_train,
        y_dev,
        y_test,
    )


def split_regression_data(df):

    df = df.dropna(
        subset=[
            "time_to_eruption_hours"
        ]
    ).copy()

    X = df[
        FEATURE_COLUMNS
    ]

    y = df[
        "time_to_eruption_hours"
    ]

    # 70% training
    # 30% temporary
    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
        )
    )

    # 20% development
    # 10% test
    X_dev, X_test, y_dev, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=1 / 3,
            random_state=42,
        )
    )

    return (
        X_train,
        X_dev,
        X_test,
        y_train,
        y_dev,
        y_test,
    )


# ============================================================
# CLASSIFICATION MODELS
# ============================================================

def get_classification_models():

    return {

        "logistic_regression": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),

        "kmeans": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    KMeansPrototypeClassifier(
                        n_clusters_per_class=8,
                        random_state=42,
                        n_init=20,
                    ),
                ),
            ]
        ),

        "random_forest": (
            RandomForestClassifier(
                n_estimators=400,
                max_depth=15,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        ),

        "neural_network": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(
                            128,
                            128,
                            128,
                            128,
                        ),
                        activation="relu",
                        alpha=1e-4,
                        batch_size=50,
                        learning_rate_init=1e-3,
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


# ============================================================
# REGRESSION MODELS
# ============================================================

def get_regression_models():

    return {

        "kmeans": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    KMeansTimeRegressor(
                        n_clusters=10,
                        random_state=42,
                        n_init=20,
                    ),
                ),
            ]
        ),

        "random_forest": (
            RandomForestRegressor(
                n_estimators=400,
                max_depth=15,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )
        ),

        "neural_network": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(
                            128,
                            128,
                            128,
                            128,
                        ),
                        activation="relu",
                        alpha=1e-4,
                        batch_size=50,
                        learning_rate_init=1e-3,
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


# ============================================================
# CLASSIFICATION EVALUATION
# ============================================================

def evaluate_classification(
    name,
    model,
    X,
    y,
):

    predictions = (
        model.predict(X)
    )

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    return {

        "model": name,

        "accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
        ),

        "kappa": float(
            cohen_kappa_score(
                y,
                predictions,
            )
        ),

        "auroc": float(
            roc_auc_score(
                y,
                probabilities,
            )
        ),

        "confusion_matrix": (
            confusion_matrix(
                y,
                predictions,
            ).tolist()
        ),

        "classification_report": (
            classification_report(
                y,
                predictions,
                output_dict=True,
            )
        ),
    }


# ============================================================
# REGRESSION EVALUATION
# ============================================================

def evaluate_regression(
    name,
    model,
    X,
    y,
):

    predictions = (
        model.predict(X)
    )

    rmse = (
        mean_squared_error(
            y,
            predictions,
        )
        ** 0.5
    )

    r2 = r2_score(
        y,
        predictions,
    )

    return {

        "model": name,

        "rmse": float(
            rmse
        ),

        "r2": float(
            r2
        ),
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    name,
    matrix,
):

    plt.figure(
        figsize=(5, 4)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
    )

    plt.title(
        f"{name.replace('_', ' ').title()} "
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES
        / f"{name}_confusion_matrix.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# CLASSIFICATION COMPARISON
# ============================================================

def plot_classification_comparison(
    results,
):

    df = pd.DataFrame(
        results
    )

    x = np.arange(
        len(df)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        x - width / 2,
        df["kappa"],
        width,
        label="Cohen's Kappa",
    )

    ax.bar(
        x + width / 2,
        df["auroc"],
        width,
        label="AUROC",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            name.replace(
                "_",
                " ",
            ).title()
            for name in df["model"]
        ]
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Classification Model Comparison"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES
        / "classification_comparison.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# REGRESSION COMPARISON
# ============================================================

def plot_regression_comparison(
    results,
):

    df = pd.DataFrame(
        results
    )

    x = np.arange(
        len(df)
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        x - 0.18,
        df["rmse"],
        width=0.36,
        label="RMSE",
    )

    ax.bar(
        x + 0.18,
        df["r2"],
        width=0.36,
        label="R²",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            name.replace(
                "_",
                " ",
            ).title()
            for name in df["model"]
        ]
    )

    ax.set_ylabel(
        "Metric Value"
    )

    ax.set_title(
        "Regression Model Comparison"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES
        / "regression_comparison.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(
    random_forest,
):

    importances = (
        random_forest
        .feature_importances_
    )

    data = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": importances,
        }
    )

    data = data.sort_values(
        "importance",
        ascending=True,
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.barh(
        data["feature"],
        data["importance"],
    )

    plt.xlabel(
        "Importance"
    )

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES
        / "feature_importance.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# OBSERVED VS PREDICTED
# ============================================================

def plot_observed_vs_predicted(
    model,
    X_test,
    y_test,
):

    predictions = (
        model.predict(
            X_test
        )
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.scatter(
        y_test,
        predictions,
        alpha=0.45,
    )

    minimum = min(
        float(y_test.min()),
        float(predictions.min()),
    )

    maximum = max(
        float(y_test.max()),
        float(predictions.max()),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )

    plt.xlabel(
        "Observed Time to Eruption (hours)"
    )

    plt.ylabel(
        "Predicted Time to Eruption (hours)"
    )

    plt.title(
        "Observed vs Predicted Time to Eruption"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES
        / "observed_vs_predicted.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    MODELS.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 60)
    print("BUILDING DATASET")
    print("=" * 60)

    df = build_dataset()

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    print()
    print("=" * 60)
    print("CLASSIFICATION")
    print("=" * 60)

    (
        X_train,
        X_dev,
        X_test,
        y_train,
        y_dev,
        y_test,
    ) = split_classification_data(
        df
    )

    classification_results = []

    classification_models = (
        get_classification_models()
    )

    for name, model in (
        classification_models.items()
    ):

        print()
        print(
            f"Training classifier: "
            f"{name}"
        )

        model.fit(
            X_train,
            y_train,
        )

        development_results = (
            evaluate_classification(
                name,
                model,
                X_dev,
                y_dev,
            )
        )

        test_results = (
            evaluate_classification(
                name,
                model,
                X_test,
                y_test,
            )
        )

        classification_results.append(
            {
                "model": name,

                "dev_kappa": (
                    development_results[
                        "kappa"
                    ]
                ),

                "dev_auroc": (
                    development_results[
                        "auroc"
                    ]
                ),

                "test_kappa": (
                    test_results[
                        "kappa"
                    ]
                ),

                "test_auroc": (
                    test_results[
                        "auroc"
                    ]
                ),

                "test_accuracy": (
                    test_results[
                        "accuracy"
                    ]
                ),
            }
        )

        plot_confusion_matrix(
            name,
            np.array(
                test_results[
                    "confusion_matrix"
                ]
            ),
        )

        joblib.dump(
            model,
            MODELS
            / f"{name}_classifier.joblib",
        )

    with open(
        METRICS
        / "classification_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            classification_results,
            file,
            indent=2,
        )

    plot_classification_comparison(
        [
            {
                "model": result["model"],
                "kappa": result[
                    "test_kappa"
                ],
                "auroc": result[
                    "test_auroc"
                ],
            }
            for result
            in classification_results
        ]
    )

    random_forest_classifier = (
        joblib.load(
            MODELS
            / "random_forest_classifier.joblib"
        )
    )

    plot_feature_importance(
        random_forest_classifier
    )

    # REGRESSION
    print()
    print("=" * 60)
    print("REGRESSION")
    print("=" * 60)

    (
        X_train,
        X_dev,
        X_test,
        y_train,
        y_dev,
        y_test,
    ) = split_regression_data(
        df
    )

    regression_results = []

    regression_models = (
        get_regression_models()
    )

    for name, model in (
        regression_models.items()
    ):

        print()
        print(
            f"Training regressor: "
            f"{name}"
        )

        model.fit(
            X_train,
            y_train,
        )

        development_results = (
            evaluate_regression(
                name,
                model,
                X_dev,
                y_dev,
            )
        )

        test_results = (
            evaluate_regression(
                name,
                model,
                X_test,
                y_test,
            )
        )

        regression_results.append(
            {
                "model": name,

                "dev_rmse": (
                    development_results[
                        "rmse"
                    ]
                ),

                "dev_r2": (
                    development_results[
                        "r2"
                    ]
                ),

                "test_rmse": (
                    test_results[
                        "rmse"
                    ]
                ),

                "test_r2": (
                    test_results[
                        "r2"
                    ]
                ),
            }
        )

        joblib.dump(
            model,
            MODELS
            / f"{name}_regressor.joblib",
        )

    with open(
        METRICS
        / "regression_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            regression_results,
            file,
            indent=2,
        )

    plot_regression_comparison(
        [
            {
                "model": result["model"],
                "rmse": result[
                    "test_rmse"
                ],
                "r2": result[
                    "test_r2"
                ],
            }
            for result
            in regression_results
        ]
    )

    random_forest_regressor = (
        joblib.load(
            MODELS
            / "random_forest_regressor.joblib"
        )
    )

    plot_observed_vs_predicted(
        random_forest_regressor,
        X_test,
        y_test,
    )

    # FINAL SUMMARY
    summary = {

        "rows": int(
            len(df)
        ),

        "positive_labels": int(
            df["erupting"].sum()
        ),

        "negative_labels": int(
            (
                df["erupting"]
                == 0
            ).sum()
        ),

        "regression_rows": int(
            df[
                "time_to_eruption_hours"
            ]
            .notna()
            .sum()
        ),

        "classification_models": (
            classification_results
        ),

        "regression_models": (
            regression_results
        ),
    }

    with open(
        METRICS
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print()
    print("Classification results:")

    print(
        pd.DataFrame(
            classification_results
        ).to_string(
            index=False
        )
    )

    print()
    print("Regression results:")

    print(
        pd.DataFrame(
            regression_results
        ).to_string(
            index=False
        )
    )

    print()
    print(
        f"Models saved to:\n"
        f"{MODELS}"
    )

    print()
    print(
        f"Figures saved to:\n"
        f"{FIGURES}"
    )

    print()
    print(
        f"Metrics saved to:\n"
        f"{METRICS}"
    )


if __name__ == "__main__":
    main()

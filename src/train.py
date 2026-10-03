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
)

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.neural_network import (
    MLPClassifier,
)

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.preprocessing import (
    StandardScaler,
)

from sklearn.impute import (
    SimpleImputer,
)

from .models import (
    KMeansPrototypeClassifier,
)

from .prepare_data import (
    FEATURE_COLUMNS,
    FORECAST_HORIZON_DAYS,
    ROOT,
    TARGET_COLUMN,
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


# TIME-BASED SPLIT
def split_time_series_data(df):
    """
    Split the daily observations chronologically.

    No random shuffling is used.

    70% -> training
    15% -> development
    15% -> testing
    """

    data = (
        df.sort_values("date")
        .reset_index(drop=True)
    )

    n = len(data)

    train_end = int(
        n * 0.70
    )

    dev_end = int(
        n * 0.85
    )

    train = data.iloc[
        :train_end
    ]

    dev = data.iloc[
        train_end:dev_end
    ]

    test = data.iloc[
        dev_end:
    ]

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train[
        TARGET_COLUMN
    ].astype(int)

    X_dev = dev[
        FEATURE_COLUMNS
    ]

    y_dev = dev[
        TARGET_COLUMN
    ].astype(int)

    X_test = test[
        FEATURE_COLUMNS
    ]

    y_test = test[
        TARGET_COLUMN
    ].astype(int)

    split_info = {
        "train_start": str(
            train["date"].min().date()
        ),
        "train_end": str(
            train["date"].max().date()
        ),
        "dev_start": str(
            dev["date"].min().date()
        ),
        "dev_end": str(
            dev["date"].max().date()
        ),
        "test_start": str(
            test["date"].min().date()
        ),
        "test_end": str(
            test["date"].max().date()
        ),
        "train_rows": int(
            len(train)
        ),
        "dev_rows": int(
            len(dev)
        ),
        "test_rows": int(
            len(test)
        ),
        "train_positive": int(
            y_train.sum()
        ),
        "dev_positive": int(
            y_dev.sum()
        ),
        "test_positive": int(
            y_test.sum()
        ),
    }

    return (
        X_train,
        X_dev,
        X_test,
        y_train,
        y_dev,
        y_test,
        split_info,
        train,
        dev,
        test,
    )


# CLASSIFICATION MODELS
def get_classification_models():

    return {

        "logistic_regression": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
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
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
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

        "random_forest": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=15,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "neural_network": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
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


# CLASSIFICATION EVALUATION
def evaluate_classification(
    model,
    X,
    y,
    threshold=0.5,
):

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {

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

        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
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
                zero_division=0,
            )
        ),
    }


# CLEAN OLD GENERATED OUTPUTS
def clean_old_outputs():

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

    # Remove old classification and regression models.
    # The new run will recreate only the four classifiers.
    for path in MODELS.glob(
        "*.joblib"
    ):
        path.unlink()

    # Remove old generated figures, including
    # the previous regression figures.
    for path in FIGURES.glob(
        "*.png"
    ):
        path.unlink()

    # Remove old metric files, including
    # regression_metrics.json.
    for path in METRICS.glob(
        "*.json"
    ):
        path.unlink()


# CONFUSION MATRIX
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


# CLASSIFICATION COMPARISON
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
        "Daily 7-Day Eruption Forecast: Model Comparison"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES
        / "classification_comparison.png",
        dpi=180,
    )

    plt.close()


# RANDOM FOREST FEATURE IMPORTANCE
def plot_feature_importance(
    random_forest,
):

    model = (
        random_forest
        .named_steps["model"]
    )

    importances = (
        model.feature_importances_
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
        figsize=(9, 6)
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


# TEST-SET FORECAST TIMELINE
def plot_test_forecast_timeline(
    model,
    X_test,
    test_dates,
    y_test,
):

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        test_dates,
        probabilities,
        label="Predicted eruption probability",
    )

    ax.scatter(
        test_dates[
            y_test.to_numpy() == 1
        ],
        np.ones(
            int(
                (
                    y_test == 1
                ).sum()
            )
        ),
        label="Actual eruption within next 7 days",
        marker="x",
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel(
        "Predicted probability"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_title(
        "Time-Based Test Forecast"
    )

    ax.legend()

    fig.autofmt_xdate()

    plt.tight_layout()

    plt.savefig(
        FIGURES
        / "test_forecast_timeline.png",
        dpi=180,
    )

    plt.close()


def find_best_threshold(y_true, probabilities):
    """
    Select the classification threshold that gives the highest
    Cohen's Kappa on the development set.

    The test set is never used for threshold selection.
    """

    thresholds = np.arange(
        0.05,
        0.96,
        0.01,
    )

    best_threshold = 0.5
    best_kappa = -1.0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        kappa = cohen_kappa_score(
            y_true,
            predictions,
        )

        if kappa > best_kappa:
            best_kappa = kappa
            best_threshold = threshold

    return (
        float(best_threshold),
        float(best_kappa),
    )


# MAIN
def main():

    clean_old_outputs()

    print()

    print("=" * 60)

    print(
        "BUILDING DAILY FORECAST DATASET"
    )

    print("=" * 60)

    df = build_dataset()

    print()

    print("=" * 60)

    print(
        "TIME-BASED CLASSIFICATION"
    )

    print("=" * 60)

    (
        X_train,
        X_dev,
        X_test,
        y_train,
        y_dev,
        y_test,
        split_info,
        train_df,
        dev_df,
        test_df,
    ) = split_time_series_data(
        df
    )

    print()

    print(
        "Chronological split:"
    )

    print(
        f"Training:    "
        f"{split_info['train_start']} "
        f"to "
        f"{split_info['train_end']} "
        f"({split_info['train_rows']:,} rows)"
    )

    print(
        f"Development: "
        f"{split_info['dev_start']} "
        f"to "
        f"{split_info['dev_end']} "
        f"({split_info['dev_rows']:,} rows)"
    )

    print(
        f"Testing:     "
        f"{split_info['test_start']} "
        f"to "
        f"{split_info['test_end']} "
        f"({split_info['test_rows']:,} rows)"
    )

    print()

    print(
        "Positive forecast windows:"
    )

    print(
        f"  Training:    "
        f"{split_info['train_positive']:,}"
    )

    print(
        f"  Development: "
        f"{split_info['dev_positive']:,}"
    )

    print(
        f"  Testing:     "
        f"{split_info['test_positive']:,}"
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

        # Select the classification threshold ONLY
        # using the development set.
        development_probabilities = (
            model.predict_proba(X_dev)[:, 1]
        )

        (
            best_threshold,
            threshold_dev_kappa,
        ) = find_best_threshold(
            y_dev,
            development_probabilities,
        )

        print(
            f"Selected threshold: "
            f"{best_threshold:.2f}"
        )

        print(
            f"Development Kappa at selected threshold: "
            f"{threshold_dev_kappa:.6f}"
        )

        # Evaluate the development set using the
        # threshold selected from the development set.
        development_results = (
            evaluate_classification(
                model,
                X_dev,
                y_dev,
                threshold=best_threshold,
            )
        )

        # Freeze the threshold and apply the same
        # threshold to the unseen test set.
        test_results = (
            evaluate_classification(
                model,
                X_test,
                y_test,
                threshold=best_threshold,
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

                "test_precision": (
                    test_results[
                        "precision"
                    ]
                ),

                "test_recall": (
                    test_results[
                        "recall"
                    ]
                ),

                "test_f1": (
                    test_results[
                        "f1"
                    ]
                ),

                "threshold": (
                    best_threshold
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

    plot_test_forecast_timeline(
        random_forest_classifier,
        X_test,
        test_df["date"],
        y_test,
    )

    summary = {

        "task": (
            "Daily-window classification: "
            "predict whether an eruption starts "
            f"within the next "
            f"{FORECAST_HORIZON_DAYS} days"
        ),

        "forecast_horizon_days": (
            FORECAST_HORIZON_DAYS
        ),

        "rows": int(
            len(df)
        ),

        "positive_labels": int(
            df[TARGET_COLUMN].sum()
        ),

        "negative_labels": int(
            (
                df[TARGET_COLUMN] == 0
            ).sum()
        ),

        "date_start": str(
            df["date"].min().date()
        ),

        "date_end": str(
            df["date"].max().date()
        ),

        "features": FEATURE_COLUMNS,

        "split": split_info,

        "classification_models": (
            classification_results
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

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 60)

    print()

    print(
        "Classification results:"
    )

    print(
        pd.DataFrame(
            classification_results
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

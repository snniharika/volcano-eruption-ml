from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# PATHS
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"



# STREAMLIT PAGE CONFIGURATION
st.set_page_config(
    page_title="Kīlauea Eruption Predictor",
    layout="wide",
)


# TITLE
st.title("Predicting Eruptive Events at Volcanoes from Earthquake Data")

st.caption(
    "Academic ML mini-project based on the supplied paper "
    "and its Kīlauea earthquake/eruption data."
)


# LOAD SUMMARY
summary_path = METRICS / "summary.json"

if not summary_path.exists():
    st.error(
        "Training results were not found. "
        "Run `python -m src.train` first."
    )
    st.stop()

summary = json.loads(
    summary_path.read_text(encoding="utf-8")
)


# DATASET SUMMARY
col1, col2, col3 = st.columns(3)

col1.metric(
    "Earthquake records",
    f"{summary['rows']:,}"
)

col2.metric(
    "Erupting labels",
    f"{summary['positive_labels']:,}"
)

col3.metric(
    "Repose labels",
    f"{summary['negative_labels']:,}"
)


# CLASSIFICATION RESULTS
st.subheader("Classification results")

classification_metrics = pd.DataFrame(
    summary["classification_models"]
)

st.dataframe(
    classification_metrics,
    use_container_width=True,
    hide_index=True,
)


# REGRESSION RESULTS
st.subheader("Regression results")

regression_metrics = pd.DataFrame(
    summary["regression_models"]
)

st.dataframe(
    regression_metrics,
    use_container_width=True,
    hide_index=True,
)


# REGRESSION VISUALIZATIONS
st.subheader("Regression model comparison")

chart_col1, chart_col2 = st.columns(2)

# RMSE CHART
with chart_col1:

    fig_rmse, ax_rmse = plt.subplots(figsize=(7, 5))

    model_labels = [
        model.replace("_", " ").title()
        for model in regression_metrics["model"]
    ]

    ax_rmse.bar(
        model_labels,
        regression_metrics["test_rmse"],
    )

    ax_rmse.set_title("Test RMSE Comparison")
    ax_rmse.set_ylabel("RMSE (hours)")
    ax_rmse.set_xlabel("Model")

    ax_rmse.tick_params(axis="x", rotation=20)

    for index, value in enumerate(
        regression_metrics["test_rmse"]
    ):
        ax_rmse.text(
            index,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    fig_rmse.tight_layout()

    st.pyplot(
        fig_rmse,
        use_container_width=True,
    )

    plt.close(fig_rmse)


# R² CHART
with chart_col2:

    fig_r2, ax_r2 = plt.subplots(figsize=(7, 5))

    ax_r2.bar(
        model_labels,
        regression_metrics["test_r2"],
    )

    ax_r2.set_title("Test R² Comparison")
    ax_r2.set_ylabel("R²")
    ax_r2.set_xlabel("Model")

    ax_r2.tick_params(axis="x", rotation=20)

    for index, value in enumerate(
        regression_metrics["test_r2"]
    ):
        ax_r2.text(
            index,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )

    ax_r2.axhline(
        y=0,
        linewidth=1,
    )

    fig_r2.tight_layout()

    st.pyplot(
        fig_r2,
        use_container_width=True,
    )

    plt.close(fig_r2)


# SINGLE EARTHQUAKE INPUT
st.subheader("Single-earthquake prediction")

st.write(
    "Enter the characteristics of an earthquake and the "
    "earthquake counts from the previous 1, 7, and 30 days."
)


# INPUT FEATURES
c1, c2, c3 = st.columns(3)

latitude = c1.number_input(
    "Latitude",
    value=19.35,
    min_value=-90.0,
    max_value=90.0,
    format="%.5f",
)

longitude = c2.number_input(
    "Longitude",
    value=-155.18,
    min_value=-180.0,
    max_value=180.0,
    format="%.5f",
)

depth = c3.number_input(
    "Depth (km)",
    value=5.0,
    min_value=0.0,
    format="%.2f",
)


c4, c5, c6 = st.columns(3)

magnitude = c4.number_input(
    "Magnitude",
    value=2.0,
    min_value=0.0,
    format="%.2f",
)

rate_1d = c5.number_input(
    "Earthquakes in previous 1 day",
    value=10,
    min_value=0,
)

rate_7d = c6.number_input(
    "Earthquakes in previous 7 days",
    value=40,
    min_value=0,
)

rate_30d = st.number_input(
    "Earthquakes in previous 30 days",
    value=120,
    min_value=0,
)


# CREATE FEATURE DATAFRAME
features = pd.DataFrame([
    {
        "latitude": latitude,
        "longitude": longitude,
        "depth": depth,
        "magnitude": magnitude,
        "eq_rate_1d": rate_1d,
        "eq_rate_7d": rate_7d,
        "eq_rate_30d": rate_30d,
    }
])

# CLASSIFICATION PREDICTION
st.markdown("### Eruption-state classification")

classification_models = [
    "random_forest",
    "logistic_regression",
    "kmeans",
    "neural_network",
]

classification_model_name = st.selectbox(
    "Classification model",
    classification_models,
    key="classification_model",
)


if st.button(
    "Predict eruption state",
    type="primary",
):

    classifier_path = (
        MODELS
        / f"{classification_model_name}_classifier.joblib"
    )

    if not classifier_path.exists():

        st.error(
            f"Model file not found: {classifier_path.name}"
        )

    else:

        classifier = joblib.load(
            classifier_path
        )

        probability = float(
            classifier.predict_proba(
                features
            )[0, 1]
        )

        prediction = int(
            probability >= 0.5
        )

        st.markdown("#### Classification result")

        result_col1, result_col2 = st.columns(2)

        if prediction == 1:

            result_col1.error(
                "Predicted state: ERUPTING"
            )

        else:

            result_col1.success(
                "Predicted state: REPOSE"
            )

        result_col2.metric(
            "Estimated eruption probability",
            f"{probability:.3f}",
        )


# TIME-TO-ERUPTION REGRESSION
st.markdown("### Time-to-eruption regression")

regression_models = [
    "random_forest",
    "kmeans",
    "neural_network",
]

regression_model_name = st.selectbox(
    "Regression model",
    regression_models,
    key="regression_model",
)


if st.button(
    "Predict time to eruption",
):

    regressor_path = (
        MODELS
        / f"{regression_model_name}_regressor.joblib"
    )

    if not regressor_path.exists():

        st.error(
            f"Model file not found: {regressor_path.name}"
        )

    else:

        regressor = joblib.load(
            regressor_path
        )

        predicted_hours = float(
            regressor.predict(
                features
            )[0]
        )

        # Prevent a tiny negative floating-point
        # prediction from being displayed.
        predicted_hours = max(
            0.0,
            predicted_hours,
        )

        predicted_days = (
            predicted_hours / 24.0
        )

        st.markdown("#### Regression result")

        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Predicted time to eruption",
            f"{predicted_hours:.2f} hours",
        )

        result_col2.metric(
            "Predicted time to eruption",
            f"{predicted_days:.2f} days",
        )

        if predicted_hours == 0:

            st.warning(
                "The model predicts approximately 0 hours "
                "to eruption, corresponding to an erupting state."
            )

        else:

            st.info(
                "This is a model prediction based only on the "
                "seven earthquake features entered above. "
                "It is not a real-time volcanic warning."
            )


# GENERATED PROJECT FIGURES
st.subheader("Generated project figures")


figure_files = [
    (
        "classification_comparison.png",
        "Classification Model Comparison",
    ),
    (
        "feature_importance.png",
        "Random Forest Feature Importance",
    ),
    (
        "observed_vs_predicted.png",
        "Observed vs Predicted Time to Eruption",
    ),
]


for filename, caption in figure_files:

    path = FIGURES / filename

    if path.exists():

        st.image(
            str(path),
            caption=caption,
            use_container_width=True,
        )

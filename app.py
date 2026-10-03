from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import streamlit as st

from src.prepare_data import (
    FEATURE_COLUMNS,
    FORECAST_HORIZON_DAYS,
)


# PATHS
ROOT = Path(__file__).resolve().parent

MODELS = (
    ROOT
    / "models"
)

METRICS = (
    ROOT
    / "results"
    / "metrics"
)

FIGURES = (
    ROOT
    / "results"
    / "figures"
)


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Kīlauea Eruption Forecast",
    layout="wide",
)


# TITLE
st.title(
    "Kīlauea Eruption Forecast from Daily Earthquake Activity"
)

st.caption(
    "Academic ML mini-project using daily earthquake windows "
    "to forecast whether an eruption will start within the "
    f"next {FORECAST_HORIZON_DAYS} days."
)


# LOAD SUMMARY
summary_path = (
    METRICS
    / "summary.json"
)

if not summary_path.exists():

    st.error(
        "Training results were not found. "
        "Run `python -m src.train` first."
    )

    st.stop()


summary = json.loads(
    summary_path.read_text(
        encoding="utf-8"
    )
)


# DATASET SUMMARY
col1, col2, col3 = st.columns(3)

col1.metric(
    "Daily forecast windows",
    f"{summary['rows']:,}",
)

col2.metric(
    "Positive windows",
    f"{summary['positive_labels']:,}",
)

col3.metric(
    "Negative windows",
    f"{summary['negative_labels']:,}",
)


st.info(
    "Each row represents one calendar day. "
    f"The target is 1 when an eruption starts within "
    f"the following {FORECAST_HORIZON_DAYS} days."
)


# TIME-BASED SPLIT INFORMATION
st.subheader(
    "Chronological train / development / test split"
)

split_info = summary["split"]

split_table = pd.DataFrame(
    [
        {
            "set": "Training",
            "start": split_info[
                "train_start"
            ],
            "end": split_info[
                "train_end"
            ],
            "rows": split_info[
                "train_rows"
            ],
            "positive windows": split_info[
                "train_positive"
            ],
        },
        {
            "set": "Development",
            "start": split_info[
                "dev_start"
            ],
            "end": split_info[
                "dev_end"
            ],
            "rows": split_info[
                "dev_rows"
            ],
            "positive windows": split_info[
                "dev_positive"
            ],
        },
        {
            "set": "Testing",
            "start": split_info[
                "test_start"
            ],
            "end": split_info[
                "test_end"
            ],
            "rows": split_info[
                "test_rows"
            ],
            "positive windows": split_info[
                "test_positive"
            ],
        },
    ]
)

st.dataframe(
    split_table,
    use_container_width=True,
    hide_index=True,
)


# CLASSIFICATION RESULTS
st.subheader(
    "Daily 7-day forecast model results"
)

classification_metrics = pd.DataFrame(
    summary["classification_models"]
)

st.dataframe(
    classification_metrics,
    use_container_width=True,
    hide_index=True,
)


# SINGLE-DAY FORECAST
st.subheader(
    "Forecast from one daily earthquake window"
)

st.write(
    "Enter the earthquake activity observed during a day "
    "and the historical features available up to that day."
)


# DAILY EARTHQUAKE ACTIVITY
c1, c2, c3 = st.columns(3)

daily_eq_count = c1.number_input(
    "Earthquakes during the day",
    min_value=0,
    value=5,
    step=1,
)

prev_1d_eq_count = c2.number_input(
    "Earthquakes in previous 1 day",
    min_value=0,
    value=5,
    step=1,
)

prev_7d_eq_count = c3.number_input(
    "Earthquakes in previous 7 days",
    min_value=0,
    value=35,
    step=1,
)


c4, c5, c6 = st.columns(3)

prev_30d_eq_count = c4.number_input(
    "Earthquakes in previous 30 days",
    min_value=0,
    value=120,
    step=1,
)

mean_magnitude = c5.number_input(
    "Mean magnitude",
    min_value=0.0,
    value=2.0,
    step=0.01,
)

seismic_energy = c6.number_input(
    "Total relative seismic energy",
    min_value=0.0,
    value=10000.0,
    step=1000.0,
)


# LOCATION AND DEPTH FEATURES
c7, c8, c9 = st.columns(3)

mean_latitude = c7.number_input(
    "Mean latitude",
    min_value=-90.0,
    max_value=90.0,
    value=19.35,
    format="%.5f",
)

mean_longitude = c8.number_input(
    "Mean longitude",
    min_value=-180.0,
    max_value=180.0,
    value=-155.18,
    format="%.5f",
)

mean_depth = c9.number_input(
    "Mean depth (km)",
    min_value=0.0,
    value=7.5,
    step=0.1,
)


c10, c11, c12 = st.columns(3)

std_depth = c10.number_input(
    "Depth standard deviation",
    min_value=0.0,
    value=2.0,
    step=0.1,
)

days_since_last_eruption = c11.number_input(
    "Days since last eruption",
    min_value=-1.0,
    value=25.0,
    step=1.0,
)

last_repose_days = c12.number_input(
    "Last recorded repose length (days)",
    min_value=-1.0,
    value=24.0,
    step=0.1,
)


# DERIVED FEATURE
eq_count_ratio_1d_30d = (
    prev_1d_eq_count
    / (
        prev_30d_eq_count
        + 1.0
    )
)

st.caption(
    f"Derived 1-day/30-day earthquake-count ratio: "
    f"{eq_count_ratio_1d_30d:.4f}"
)


# CREATE INPUT DATAFRAME
features = pd.DataFrame(
    [
        {
            "daily_eq_count": daily_eq_count,
            "mean_latitude": mean_latitude,
            "mean_longitude": mean_longitude,
            "mean_depth": mean_depth,
            "std_depth": std_depth,
            "mean_magnitude": mean_magnitude,
            "seismic_energy": seismic_energy,
            "prev_1d_eq_count": prev_1d_eq_count,
            "prev_7d_eq_count": prev_7d_eq_count,
            "prev_30d_eq_count": prev_30d_eq_count,
            "eq_count_ratio_1d_30d": (
                eq_count_ratio_1d_30d
            ),
            "days_since_last_eruption": (
                days_since_last_eruption
            ),
            "last_repose_days": (
                last_repose_days
            ),
        }
    ]
)

features = features[
    FEATURE_COLUMNS
]


# MODEL SELECTION
st.markdown(
    "### Forecast model"
)

model_names = [
    "logistic_regression",
    "random_forest",
    "kmeans",
    "neural_network",
]

model_name = st.selectbox(
    "Classification model",
    model_names,
)


# PREDICTION
if st.button(
    "Forecast eruption in next 7 days",
    type="primary",
):

    model_path = (
        MODELS
        / f"{model_name}_classifier.joblib"
    )

    if not model_path.exists():

        st.error(
            f"Model file not found: "
            f"{model_path.name}"
        )

    else:

        model = joblib.load(
            model_path
        )

        probability = float(
            model.predict_proba(
                features
            )[0, 1]
        )

        prediction = int(
            model.predict(
                features
            )[0]
        )

        st.markdown(
            "#### Forecast result"
        )

        result_col1, result_col2 = (
            st.columns(2)
        )

        if prediction == 1:

            result_col1.error(
                f"Predicted: eruption starts "
                f"within the next "
                f"{FORECAST_HORIZON_DAYS} days"
            )

        else:

            result_col1.success(
                f"Predicted: no eruption start "
                f"within the next "
                f"{FORECAST_HORIZON_DAYS} days"
            )

        result_col2.metric(
            "Estimated probability",
            f"{probability:.3f}",
        )

        st.info(
            "This is an academic model prediction based on "
            "the supplied historical earthquake and eruption "
            "catalogues. It is not an operational volcanic "
            "warning."
        )


# GENERATED FIGURES
st.subheader(
    "Generated project figures"
)


figure_files = [
    (
        "classification_comparison.png",
        "Daily 7-Day Forecast Model Comparison",
    ),
    (
        "feature_importance.png",
        "Random Forest Feature Importance",
    ),
    (
        "test_forecast_timeline.png",
        "Time-Based Test Forecast",
    ),
]


for filename, caption in figure_files:

    path = (
        FIGURES
        / filename
    )

    if path.exists():

        st.image(
            str(path),
            caption=caption,
            use_container_width=True,
        )


# CONFUSION MATRICES
st.subheader(
    "Test-set confusion matrices"
)

confusion_files = [
    (
        "logistic_regression_confusion_matrix.png",
        "Logistic Regression",
    ),
    (
        "kmeans_confusion_matrix.png",
        "K-Means",
    ),
    (
        "random_forest_confusion_matrix.png",
        "Random Forest",
    ),
    (
        "neural_network_confusion_matrix.png",
        "Neural Network",
    ),
]

confusion_columns = st.columns(2)

for index, (
    filename,
    caption,
) in enumerate(
    confusion_files
):

    path = (
        FIGURES
        / filename
    )

    if path.exists():

        with confusion_columns[
            index % 2
        ]:

            st.image(
                str(path),
                caption=caption,
                use_container_width=True,
            )

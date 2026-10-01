from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
METRICS = ROOT / "results" / "metrics"

st.set_page_config(
    page_title="Kīlauea Eruption Predictor",
    layout="wide",
)

st.title("Kīlauea Eruption Prediction from Earthquake Data")
st.caption(
    "Academic ML mini-project based on the supplied paper "
    "and its Kīlauea earthquake/eruption data."
)

summary_path = METRICS / "summary.json"
if not summary_path.exists():
    st.error("Run `python -m src.train` first.")
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
metrics = pd.DataFrame(summary["classification_models"])

col1, col2, col3 = st.columns(3)
col1.metric("Earthquake records", f"{summary['rows']:,}")
col2.metric("Erupting labels", f"{summary['positive_labels']:,}")
col3.metric("Repose labels", f"{summary['negative_labels']:,}")

st.subheader("Classification results")
st.dataframe(metrics, use_container_width=True)

st.subheader("Single-earthquake prediction")

c1, c2, c3 = st.columns(3)
latitude = c1.number_input("Latitude", value=19.35, format="%.5f")
longitude = c2.number_input("Longitude", value=-155.18, format="%.5f")
depth = c3.number_input("Depth (km)", value=5.0, min_value=0.0)

c4, c5, c6 = st.columns(3)
magnitude = c4.number_input("Magnitude", value=2.0, min_value=0.0)
rate_1d = c5.number_input(
    "Earthquakes in previous 1 day", value=10, min_value=0
)
rate_7d = c6.number_input(
    "Earthquakes in previous 7 days", value=40, min_value=0
)
rate_30d = st.number_input(
    "Earthquakes in previous 30 days", value=120, min_value=0
)

features = pd.DataFrame([{
    "latitude": latitude,
    "longitude": longitude,
    "depth": depth,
    "magnitude": magnitude,
    "eq_rate_1d": rate_1d,
    "eq_rate_7d": rate_7d,
    "eq_rate_30d": rate_30d,
}])

model_name = st.selectbox(
    "Classification model",
    [
        "random_forest",
        "logistic_regression",
        "kmeans",
        "neural_network",
    ],
)

if st.button("Predict eruption state", type="primary"):
    model = joblib.load(
        MODELS / f"{model_name}_classifier.joblib"
    )
    probability = float(model.predict_proba(features)[0, 1])
    prediction = int(probability >= 0.5)

    if prediction:
        st.warning(
            f"Predicted state: ERUPTING | probability = {probability:.3f}"
        )
    else:
        st.info(
            f"Predicted state: REPOSE | eruption probability = {probability:.3f}"
        )

st.subheader("Generated project figures")

figure_dir = ROOT / "results" / "figures"
for filename in [
    "classification_comparison.png",
    "regression_comparison.png",
    "feature_importance.png",
    "observed_vs_predicted.png",
]:
    path = figure_dir / filename
    if path.exists():
        st.image(
            str(path),
            caption=filename.replace("_", " ").replace(".png", "").title(),
        )

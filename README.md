# Kīlauea Daily Eruption Forecast from Earthquake Data

Machine-learning mini-project inspired by the paper **"Predicting Eruptive Events at Volcanoes from Earthquake Data"** by Ben Mullet, Ankush Singh and Ying-Qi Wong.

This implementation uses the supplied Kīlauea earthquake and eruption catalogues, but changes the problem setup to make it a more realistic forecasting task.

## Forecasting task

Instead of making a prediction for every individual earthquake, the earthquake catalogue is aggregated into **daily windows**.

For each day, the model uses the earthquake activity observed during that day together with historical activity and eruption-history features.

The target is:

> **Will an eruption start within the next 7 days?**

The target is:

- `1` = an eruption starts during the following 7 calendar days
- `0` = no eruption starts during the following 7 calendar days

The supplied eruption catalogue contains eruption dates but not eruption start times, so the 7-day target is defined using calendar dates.

## Main changes from the supplied paper

### 1. Time-based evaluation

The daily observations are split chronologically:

- 70% training
- 15% development
- 15% testing

No random shuffling is used.

This means the model is trained on earlier historical periods and evaluated on later periods.

### 2. Daily forecasting

The original earthquake-by-earthquake prediction setup is replaced with a daily-window forecasting problem.

The model forecasts whether an eruption will start within the next 7 days.

### 3. Additional features

The project uses:

- daily earthquake count
- mean latitude
- mean longitude
- mean depth
- depth standard deviation
- mean magnitude
- total relative seismic energy
- earthquake count in previous 1 day
- earthquake count in previous 7 days
- earthquake count in previous 30 days
- ratio of previous 1-day count to previous 30-day count
- days since the most recent known eruption
- repose length recorded for the most recent eruption

Relative seismic energy is represented using:

```text
10^(1.5 × magnitude)
```

as an energy proxy.

## Models

Four classification approaches are compared:

1. Logistic Regression
2. K-Means
3. Random Forest
4. Neural Network

The K-Means classifier uses 8 clusters per class, following the general K-Means classification idea from the supplied paper.

The neural network uses four hidden layers with 128 units per layer and is implemented with scikit-learn.

## Data handling

The raw earthquake catalogue contains individual earthquake records.

The preprocessing pipeline:

1. Parses earthquake timestamps.
2. Aggregates earthquakes by calendar day.
3. Calculates daily earthquake statistics.
4. Calculates previous 1-day, 7-day and 30-day earthquake counts.
5. Calculates the earthquake-count ratio.
6. Calculates relative seismic energy.
7. Adds eruption-history features.
8. Creates the next-7-day eruption-start target.
9. Removes the final 7 calendar days because their complete future forecasting horizon is not available in the supplied eruption catalogue.

The resulting dataset is saved as:

```text
data/processed/model_dataset.csv
```

## Project structure

```text
volcano-eruption-ml/
├── app.py
├── download_data.py
├── requirements.txt
├── README.md
├── PROJECT_PLAN.md
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── puuoo_earthquakes.csv
│   │   └── PuuOo.csv
│   └── processed/
├── models/
├── results/
│   ├── figures/
│   └── metrics/
└── src/
    ├── __init__.py
    ├── models.py
    ├── prepare_data.py
    └── train.py
```

## Setup

Create the virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the project

### 1. Download the supplied data

```bash
python download_data.py
```

### 2. Build the daily forecasting dataset

```bash
python -m src.prepare_data
```

### 3. Train all models

```bash
python -m src.train
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

## Expected pipeline

```text
Raw earthquake catalogue
        +
Raw eruption catalogue
        ↓
Daily aggregation
        ↓
Feature engineering
        ↓
Next-7-day eruption target
        ↓
Chronological split
        ↓
Train / development / test
        ↓
Logistic Regression
K-Means
Random Forest
Neural Network
        ↓
Kappa / AUROC / Accuracy / Precision / Recall / F1
        ↓
Confusion matrices
Feature importance
Time-based forecast plot
```

## Interpretation

This is an academic machine-learning forecasting exercise, not an operational volcano-warning system.

The supplied dataset covers a limited historical period and contains a relatively small number of eruption events. A time-based evaluation is therefore expected to produce different results from a randomly shuffled evaluation.

The model results should be interpreted as experiments on the supplied historical data rather than as evidence of real-time eruption prediction capability.

## Attribution

The project is inspired by the supplied paper:

**Predicting Eruptive Events at Volcanoes from Earthquake Data**

The dataset and original problem formulation are based on the supplied paper and its associated PEEVED project.

The implementation in this repository changes the forecasting unit, target horizon, data splitting strategy and feature set.

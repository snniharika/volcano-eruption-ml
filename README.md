# Predicting Eruptive Events at Kīlauea from Earthquake Data

Machine-learning mini-project based on the paper **"Predicting Eruptive Events at Volcanoes from Earthquake Data"** by Ben Mullet, Ankush Singh and Ying-Qi Wong.

## Tasks

1. **Contemporaneous eruption classification:** predict whether Kīlauea is erupting when an earthquake occurs.
2. **Time-to-eruption regression:** predict the number of hours until the next eruption.

## Features

- latitude
- longitude
- depth
- magnitude
- earthquake count in the previous 1 day
- earthquake count in the previous 7 days
- earthquake count in the previous 30 days

## Setup

```bash
git clone <YOUR_PRIVATE_REPOSITORY_URL>
cd volcano-eruption-ml
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

Download the supplied data:

```bash
python download_data.py
```

Build the processed dataset:

```bash
python -m src.prepare_data
```

Train every model:

```bash
python -m src.train
```

Run the live demo:

```bash
streamlit run app.py
```

## Repository structure

```text
volcano-eruption-ml/
├── app.py
├── download_data.py
├── requirements.txt
├── README.md
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

## Interpretation

This is an academic ML forecasting exercise, not an operational volcano-warning system. The supplied paper reports weak correlation for time-to-eruption forecasting and identifies the need for additional signals such as ground deformation and gas-emission data.

## Attribution

The dataset and methodology are based on the supplied paper and the authors' public PEEVED repository. Cite the paper in the final report and presentation.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

FORECAST_HORIZON_DAYS = 7

FEATURE_COLUMNS = [
    "daily_eq_count",
    "mean_latitude",
    "mean_longitude",
    "mean_depth",
    "std_depth",
    "mean_magnitude",
    "seismic_energy",
    "prev_1d_eq_count",
    "prev_7d_eq_count",
    "prev_30d_eq_count",
    "eq_count_ratio_1d_30d",
    "days_since_last_eruption",
    "last_repose_days",
]

TARGET_COLUMN = "eruption_next_7d"


def load_earthquakes(path=None):
    """
    Load the earthquake catalogue and parse timestamps.
    """

    if path is None:
        path = RAW / "puuoo_earthquakes.csv"

    df = pd.read_csv(path)

    df.columns = [
        column.strip()
        for column in df.columns
    ]

    df["timestamp"] = pd.to_datetime(
        df["Date-time"],
        format="%m/%d/%Y %H:%M:%S",
        errors="coerce",
    )

    df = df.rename(
        columns={
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Depth": "depth",
            "Magnitude": "magnitude",
        }
    )

    required_columns = [
        "timestamp",
        "latitude",
        "longitude",
        "depth",
        "magnitude",
    ]

    df = df[required_columns].dropna()

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


def load_eruptions(path=None):
    """
    Load the historical eruption catalogue.

    Repose is kept because it is used as a historical feature.
    """

    if path is None:
        path = RAW / "PuuOo.csv"

    eruptions = pd.read_csv(path)

    eruptions.columns = [
        column.strip()
        for column in eruptions.columns
    ]

    eruptions["eruption_start"] = pd.to_datetime(
        eruptions["Date"],
        format="%m/%d/%y",
        errors="coerce",
    )

    eruptions["repose_days"] = pd.to_numeric(
        eruptions["Repose"],
        errors="coerce",
    )

    eruptions = eruptions[
        [
            "eruption_start",
            "repose_days",
        ]
    ].dropna(
        subset=["eruption_start"]
    )

    eruptions = eruptions.sort_values(
        "eruption_start"
    ).reset_index(drop=True)

    return eruptions


def aggregate_daily_earthquakes(
    earthquakes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert individual earthquakes into one row per calendar day.

    The daily row contains:
        - number of earthquakes
        - mean location
        - mean and spread of depth
        - mean magnitude
        - total relative seismic energy

    The energy feature uses 10^(1.5 * magnitude) as a
    relative seismic-energy proxy.
    """

    data = earthquakes.copy()

    data["date"] = (
        data["timestamp"]
        .dt.floor("D")
    )

    data["relative_energy"] = (
        10.0
        ** (
            1.5
            * data["magnitude"]
        )
    )

    daily = (
        data.groupby("date")
        .agg(
            daily_eq_count=(
                "magnitude",
                "size",
            ),
            mean_latitude=(
                "latitude",
                "mean",
            ),
            mean_longitude=(
                "longitude",
                "mean",
            ),
            mean_depth=(
                "depth",
                "mean",
            ),
            std_depth=(
                "depth",
                "std",
            ),
            mean_magnitude=(
                "magnitude",
                "mean",
            ),
            seismic_energy=(
                "relative_energy",
                "sum",
            ),
        )
        .reset_index()
    )

    first_date = (
        data["date"].min()
    )

    last_date = (
        data["date"].max()
    )

    all_dates = pd.DataFrame(
        {
            "date": pd.date_range(
                first_date,
                last_date,
                freq="D",
            )
        }
    )

    daily = (
        all_dates
        .merge(
            daily,
            on="date",
            how="left",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    # No earthquake means zero count and zero total
    # relative seismic energy for that day.
    daily["daily_eq_count"] = (
        daily["daily_eq_count"]
        .fillna(0)
    )

    daily["seismic_energy"] = (
        daily["seismic_energy"]
        .fillna(0)
    )

    daily["std_depth"] = (
        daily["std_depth"]
        .fillna(0)
    )

    return daily


def add_rolling_features(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add historical earthquake-activity features.

    The rolling counts are shifted by one day, so the
    previous-day/week/month activity never includes
    the day being predicted from.
    """

    data = daily.copy()

    previous_counts = (
        data["daily_eq_count"]
        .shift(1)
    )

    data["prev_1d_eq_count"] = (
        previous_counts
        .rolling(
            window=1,
            min_periods=1,
        )
        .sum()
        .fillna(0)
    )

    data["prev_7d_eq_count"] = (
        previous_counts
        .rolling(
            window=7,
            min_periods=1,
        )
        .sum()
        .fillna(0)
    )

    data["prev_30d_eq_count"] = (
        previous_counts
        .rolling(
            window=30,
            min_periods=1,
        )
        .sum()
        .fillna(0)
    )

    data["eq_count_ratio_1d_30d"] = (
        data["prev_1d_eq_count"]
        / (
            data["prev_30d_eq_count"]
            + 1.0
        )
    )

    return data


def add_eruption_history_features(
    daily: pd.DataFrame,
    eruptions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add:
        - days since the most recent known eruption start
        - repose duration recorded for that eruption

    Only eruption information available on or before the
    current daily window is used.
    """

    data = daily.copy()

    eruption_starts = (
        eruptions["eruption_start"]
        .to_numpy(
            dtype="datetime64[ns]"
        )
    )

    eruption_start_ns = (
        eruption_starts
        .astype("int64")
    )

    daily_dates = (
        data["date"]
        .to_numpy(
            dtype="datetime64[ns]"
        )
    )

    daily_ns = (
        daily_dates
        .astype("int64")
    )

    previous_indices = np.searchsorted(
        eruption_start_ns,
        daily_ns,
        side="right",
    ) - 1

    has_previous = (
        previous_indices >= 0
    )

    safe_indices = np.clip(
        previous_indices,
        0,
        len(eruption_start_ns) - 1,
    )

    data["days_since_last_eruption"] = (
        -1.0
    )

    data["last_repose_days"] = (
        -1.0
    )

    data.loc[
        has_previous,
        "days_since_last_eruption",
    ] = (
        (
            daily_ns[has_previous]
            - eruption_start_ns[
                safe_indices[has_previous]
            ]
        )
        / 86_400_000_000_000.0
    )

    repose_values = (
        eruptions["repose_days"]
        .to_numpy(dtype=float)
    )

    data.loc[
        has_previous,
        "last_repose_days",
    ] = (
        repose_values[
            safe_indices[has_previous]
        ]
    )

    return data


def add_forecast_target(
    daily: pd.DataFrame,
    eruptions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the new forecasting target.

    For each daily observation at the end of date D:

        target = 1
        if an eruption starts during the next 7
        calendar days after D.

        target = 0 otherwise.

    The eruption date itself is not included in the
    current day's future target window.
    """

    data = daily.copy()

    eruption_dates = set(
        eruptions[
            "eruption_start"
        ].dt.floor("D")
    )

    target = []

    for date in data["date"]:

        future_dates = pd.date_range(
            date
            + pd.Timedelta(days=1),
            date
            + pd.Timedelta(
                days=FORECAST_HORIZON_DAYS
            ),
            freq="D",
        )

        target.append(
            int(
                any(
                    future_date
                    in eruption_dates
                    for future_date in future_dates
                )
            )
        )

    data[TARGET_COLUMN] = np.asarray(
        target,
        dtype=int,
    )

    # The eruption catalogue only supports targets up to
    # its final known eruption date. The final 7 days cannot
    # be labeled reliably because their complete future
    # horizon is outside the supplied eruption catalogue.
    last_eruption_date = (
        eruptions["eruption_start"]
        .max()
        .floor("D")
    )

    last_usable_date = (
        last_eruption_date
        - pd.Timedelta(
            days=FORECAST_HORIZON_DAYS
        )
    )

    data = data[
        data["date"]
        <= last_usable_date
    ].copy()

    return data.reset_index(
        drop=True
    )


def build_dataset():
    """
    Build the daily-window forecasting dataset.
    """

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading earthquake catalogue..."
    )

    earthquakes = (
        load_earthquakes()
    )

    print(
        f"Earthquake records loaded: "
        f"{len(earthquakes):,}"
    )

    print(
        "Loading eruption catalogue..."
    )

    eruptions = (
        load_eruptions()
    )

    print(
        f"Eruption records loaded: "
        f"{len(eruptions):,}"
    )

    print(
        "Aggregating earthquakes into daily windows..."
    )

    data = (
        aggregate_daily_earthquakes(
            earthquakes
        )
    )

    print(
        "Creating historical activity features..."
    )

    data = (
        add_rolling_features(
            data
        )
    )

    print(
        "Creating eruption-history features..."
    )

    data = (
        add_eruption_history_features(
            data,
            eruptions,
        )
    )

    print(
        "Creating next-7-day eruption target..."
    )

    data = (
        add_forecast_target(
            data,
            eruptions,
        )
    )

    output_columns = (
        ["date"]
        + FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    output_path = (
        PROCESSED
        / "model_dataset.csv"
    )

    data[
        output_columns
    ].to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 60)
    print("DAILY FORECAST DATASET COMPLETE")
    print("=" * 60)

    print(
        f"Forecast windows: "
        f"{len(data):,}"
    )

    print(
        f"Positive windows "
        f"(eruption in next "
        f"{FORECAST_HORIZON_DAYS} days): "
        f"{data[TARGET_COLUMN].sum():,}"
    )

    print(
        f"Negative windows: "
        f"{(data[TARGET_COLUMN] == 0).sum():,}"
    )

    print(
        f"Date range: "
        f"{data['date'].min().date()} "
        f"to "
        f"{data['date'].max().date()}"
    )

    print()
    print("Features:")

    for feature in FEATURE_COLUMNS:
        print(
            f"  - {feature}"
        )

    print()
    print(
        f"Saved processed dataset to:\n"
        f"{output_path}"
    )

    return data


if __name__ == "__main__":
    build_dataset()

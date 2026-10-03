from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "depth",
    "magnitude",
    "eq_rate_1d",
    "eq_rate_7d",
    "eq_rate_30d",
]


def load_earthquakes(path=None):
    """
    Load the earthquake catalogue.

    Required columns:
        Date-time
        Latitude
        Longitude
        Depth
        Magnitude
    """

    if path is None:
        path = RAW / "puuoo_earthquakes.csv"

    df = pd.read_csv(path)

    df.columns = [column.strip() for column in df.columns]

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

    df = df[required_columns]

    df = df.dropna()

    df = df.sort_values("timestamp")

    df = df.reset_index(drop=True)

    return df


def load_eruptions(path=None):
    """
    Load the historical eruption catalogue.
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

    eruptions["length_hours"] = (
        eruptions["Length"]
        .astype(str)
        .str.extract(r"([0-9.]+)")[0]
        .astype(float)
    )

    eruptions = eruptions[
        [
            "eruption_start",
            "length_hours",
            "Repose",
            "Flow Area",
            "Flow Volume",
            "Rate",
            "Location",
        ]
    ]

    eruptions = eruptions.dropna(
        subset=[
            "eruption_start",
            "length_hours",
        ]
    )

    eruptions = eruptions.sort_values(
        "eruption_start"
    )

    eruptions = eruptions.reset_index(drop=True)

    return eruptions


def count_previous_earthquakes(
    timestamps: pd.Series,
    window_days: int,
) -> np.ndarray:
    """
    Count previous earthquakes inside a time window.

    For every earthquake at time t, count earthquakes
    in the interval:

        [t - window_days, t)

    The current earthquake itself is therefore excluded.
    """

    values = (
        timestamps
        .astype("int64")
        .to_numpy()
    )

    window_ns = (
        window_days
        * 24
        * 60
        * 60
        * 1_000_000_000
    )

    left_indices = np.searchsorted(
        values,
        values - window_ns,
        side="left",
    )

    right_indices = np.arange(
        len(values)
    )

    counts = (
        right_indices
        - left_indices
    )

    return counts


def add_earthquake_rates(
    earthquakes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add earthquake-rate/count features.

    The paper uses counts over:
        - previous day
        - previous 7 days
        - previous 30 days
    """

    df = earthquakes.copy()

    df["eq_rate_1d"] = count_previous_earthquakes(
        df["timestamp"],
        1,
    )

    df["eq_rate_7d"] = count_previous_earthquakes(
        df["timestamp"],
        7,
    )

    df["eq_rate_30d"] = count_previous_earthquakes(
        df["timestamp"],
        30,
    )

    return df


def add_labels(
    earthquakes: pd.DataFrame,
    eruptions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create:

        erupting
        time_to_eruption_hours

    Classification:
        1 = erupting
        0 = repose

    An earthquake is considered to occur during an
    eruption if it falls within:

        eruption start
        through
        eruption duration + 24 hours
    """

    df = earthquakes.copy()

    eruption_starts = (
        eruptions["eruption_start"]
        .sort_values()
        .to_numpy()
    )

    eruption_lengths = (
        eruptions["length_hours"]
        .to_numpy(dtype=float)
    )

    earthquake_times = (
        df["timestamp"]
        .to_numpy(dtype="datetime64[ns]")
    )

    eruption_start_ns = (
        eruption_starts
        .astype("datetime64[ns]")
        .astype("int64")
    )

    earthquake_time_ns = (
        earthquake_times
        .astype("datetime64[ns]")
        .astype("int64")
    )

    # -------------------------------------------------
    # Find the most recent eruption before each
    # earthquake.
    # -------------------------------------------------

    previous_indices = np.searchsorted(
        eruption_start_ns,
        earthquake_time_ns,
        side="right",
    ) - 1

    has_previous_eruption = (
        previous_indices >= 0
    )

    safe_previous_indices = np.clip(
        previous_indices,
        0,
        len(eruption_start_ns) - 1,
    )

    elapsed_hours = (
        earthquake_time_ns
        - eruption_start_ns[
            safe_previous_indices
        ]
    ) / 3_600_000_000_000.0

    # -------------------------------------------------
    # Classification label
    # -------------------------------------------------

    erupting = np.zeros(
        len(df),
        dtype=int,
    )

    valid_previous = (
        has_previous_eruption
    )

    erupting[valid_previous] = (
        elapsed_hours[valid_previous]
        <= (
            eruption_lengths[
                safe_previous_indices[
                    valid_previous
                ]
            ]
            + 24.0
        )
    ).astype(int)

    # -------------------------------------------------
    # Find the next eruption after each earthquake.
    # -------------------------------------------------

    next_indices = np.searchsorted(
        eruption_start_ns,
        earthquake_time_ns,
        side="right",
    )

    has_next_eruption = (
        next_indices
        < len(eruption_start_ns)
    )

    time_to_eruption_hours = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    time_to_eruption_hours[
        has_next_eruption
    ] = (
        eruption_start_ns[
            next_indices[
                has_next_eruption
            ]
        ]
        - earthquake_time_ns[
            has_next_eruption
        ]
    ) / 3_600_000_000_000.0

    # During an eruption, time-to-eruption is zero.
    time_to_eruption_hours[
        erupting == 1
    ] = 0.0

    df["erupting"] = erupting

    df["time_to_eruption_hours"] = (
        time_to_eruption_hours
    )

    return df


def build_dataset():
    """
    Execute the complete preprocessing pipeline.
    """

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading earthquake catalogue...")

    earthquakes = load_earthquakes()

    print(
        f"Earthquake records loaded: "
        f"{len(earthquakes):,}"
    )

    print("Loading eruption catalogue...")

    eruptions = load_eruptions()

    print(
        f"Eruption records loaded: "
        f"{len(eruptions):,}"
    )

    print("Creating earthquake-rate features...")

    data = add_earthquake_rates(
        earthquakes
    )

    print("Creating eruption labels...")

    data = add_labels(
        data,
        eruptions,
    )

    output_path = (
        PROCESSED
        / "model_dataset.csv"
    )

    output_columns = (
        FEATURE_COLUMNS
        + [
            "erupting",
            "time_to_eruption_hours",
        ]
    )

    data[
        output_columns
    ].to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 60)
    print("DATASET CREATION COMPLETE")
    print("=" * 60)

    print(
        f"Total rows: "
        f"{len(data):,}"
    )

    print(
        f"Erupting samples: "
        f"{data['erupting'].sum():,}"
    )

    print(
        f"Repose samples: "
        f"{(data['erupting'] == 0).sum():,}"
    )

    print(
        "Samples with time-to-eruption target: "
        f"{data['time_to_eruption_hours'].notna().sum():,}"
    )

    print()
    print("Features:")

    for feature in FEATURE_COLUMNS:
        print(f"  - {feature}")

    print()
    print(
        f"Saved processed dataset to:\n"
        f"{output_path}"
    )

    return data


if __name__ == "__main__":
    build_dataset()

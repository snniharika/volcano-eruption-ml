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
    path = path or (RAW / "puuoo_earthquakes.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(
        df["Date-time"], format="%m/%d/%Y %H:%M:%S"
    )
    df = df.rename(
        columns={
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Depth": "depth",
            "Magnitude": "magnitude",
        }
    )
    keep = ["timestamp", "latitude", "longitude", "depth", "magnitude"]
    return df[keep].dropna().sort_values("timestamp").reset_index(drop=True)


def load_eruptions(path=None):
    path = path or (RAW / "PuuOo.csv")
    eruptions = pd.read_csv(path)
    eruptions.columns = [c.strip() for c in eruptions.columns]
    eruptions["eruption_start"] = pd.to_datetime(
        eruptions["Date"], format="%m/%d/%y"
    )
    eruptions["length_hours"] = (
        eruptions["Length"].astype(str)
        .str.extract(r"([0-9.]+)")[0]
        .astype(float)
    )
    return eruptions[
        [
            "eruption_start",
            "length_hours",
            "Repose",
            "Flow Area",
            "Flow Volume",
            "Rate",
            "Location",
        ]
    ].sort_values("eruption_start").reset_index(drop=True)


def _count_previous(ts, window_days):
    values = ts.astype("int64").to_numpy()
    window = window_days * 24 * 60 * 60 * 1_000_000_000
    left = np.searchsorted(values, values - window, side="left")
    right = np.arange(len(values))
    return right - left


def add_earthquake_rates(df):
    out = df.copy()
    for days, name in [
        (1, "eq_rate_1d"),
        (7, "eq_rate_7d"),
        (30, "eq_rate_30d"),
    ]:
        out[name] = _count_previous(out["timestamp"], days)
    return out


def add_labels(df, eruptions):
    out = df.copy()
    starts = eruptions["eruption_start"].sort_values().to_numpy()
    lengths = eruptions["length_hours"].to_numpy(dtype=float)

    timestamps = out["timestamp"].to_numpy(dtype="datetime64[ns]")
    start_ns = starts.astype("datetime64[ns]").astype("int64")
    time_ns = timestamps.astype("datetime64[ns]").astype("int64")

    previous_idx = np.searchsorted(start_ns, time_ns, side="right") - 1
    has_previous = previous_idx >= 0
    safe_idx = np.clip(previous_idx, 0, len(starts) - 1)

    elapsed_hours = (
        time_ns - start_ns[safe_idx]
    ) / 3_600_000_000_000.0

    erupting = np.zeros(len(out), dtype=int)
    valid = has_previous
    erupting[valid] = (
        elapsed_hours[valid] <= lengths[safe_idx[valid]] + 24.0
    ).astype(int)

    next_idx = np.searchsorted(start_ns, time_ns, side="right")
    has_next = next_idx < len(start_ns)

    time_to_eruption_hours = np.full(len(out), np.nan, dtype=float)
    time_to_eruption_hours[has_next] = (
        start_ns[next_idx[has_next]] - time_ns[has_next]
    ) / 3_600_000_000_000.0
    time_to_eruption_hours[erupting == 1] = 0.0

    out["erupting"] = erupting
    out["time_to_eruption_hours"] = time_to_eruption_hours
    return out


def build_dataset():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    earthquakes = load_earthquakes()
    eruptions = load_eruptions()

    data = add_earthquake_rates(earthquakes)
    data = add_labels(data, eruptions)

    output = PROCESSED / "model_dataset.csv"
    data[
        FEATURE_COLUMNS + ["erupting", "time_to_eruption_hours"]
    ].to_csv(output, index=False)

    return data


if __name__ == "__main__":
    df = build_dataset()
    print(f"Rows: {len(df):,}")
    print(f"Positive eruption labels: {df['erupting'].sum():,}")
    print(f"Negative labels: {(df['erupting'] == 0).sum():,}")
    print(
        "Regression rows with target: "
        f"{df['time_to_eruption_hours'].notna().sum():,}"
    )
    print(f"Saved to: {PROCESSED / 'model_dataset.csv'}")

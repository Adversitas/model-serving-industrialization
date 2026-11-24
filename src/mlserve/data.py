"""Forest CoverType data (UCI, Blackard 1998): 581,012 30x30 m cells, 54 cartographic
variables, 7 forest cover types. Loaded through scikit-learn's fetcher."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
WILDERNESS = [f"Wilderness_Area{i}" for i in range(1, 5)]
SOIL = [f"Soil_Type{i}" for i in range(1, 41)]
RAW_COLUMNS = NUMERIC + WILDERNESS + SOIL
TARGET = "Cover_Type"
CLASSES = [
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
]


def load(data_dir: Path) -> pd.DataFrame:
    from sklearn.datasets import fetch_covtype

    b = fetch_covtype(data_home=str(data_dir), download_if_missing=True)
    df = pd.DataFrame(b.data, columns=RAW_COLUMNS)
    for c in WILDERNESS + SOIL:
        df[c] = df[c].astype(np.int8)
    df[TARGET] = b.target.astype(np.int64)  # 1..7
    return df


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 0):
    from sklearn.model_selection import train_test_split

    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])


def sample_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Synthetic rows with the real schema and plausible ranges (tests, no download)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Elevation": rng.integers(1850, 3850, n),
            "Aspect": rng.integers(0, 360, n),
            "Slope": rng.integers(0, 60, n),
            "Horizontal_Distance_To_Hydrology": rng.integers(0, 1300, n),
            "Vertical_Distance_To_Hydrology": rng.integers(-150, 550, n),
            "Horizontal_Distance_To_Roadways": rng.integers(0, 7000, n),
            "Hillshade_9am": rng.integers(0, 255, n),
            "Hillshade_Noon": rng.integers(0, 255, n),
            "Hillshade_3pm": rng.integers(0, 255, n),
            "Horizontal_Distance_To_Fire_Points": rng.integers(0, 7000, n),
        }
    ).astype(np.float64)
    w = rng.integers(0, 4, n)
    s = rng.integers(0, 40, n)
    for i, c in enumerate(WILDERNESS):
        df[c] = (w == i).astype(np.int8)
    for i, c in enumerate(SOIL):
        df[c] = (s == i).astype(np.int8)
    df[TARGET] = 1 + (df["Elevation"].to_numpy() // 300 + w) % 7
    return df

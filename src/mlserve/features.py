"""Production feature engineering: the prototype's logic, vectorised and validated."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import NUMERIC, RAW_COLUMNS, SOIL, WILDERNESS

FEATURES = [
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "hydro_h",
    "hydro_v",
    "hydro_dist",
    "elev_minus_hydro_v",
    "elev_minus_hydro_h",
    "road_dist",
    "fire_dist",
    "road_plus_fire",
    "road_minus_fire",
    "hillshade_9am",
    "hillshade_noon",
    "hillshade_3pm",
    "hillshade_mean",
    "wilderness",
    "soil",
]
CATEGORICAL = ["wilderness", "soil"]
RANGES = {
    "Aspect": (0, 360),
    "Slope": (0, 90),
    "Hillshade_9am": (0, 255),
    "Hillshade_Noon": (0, 255),
    "Hillshade_3pm": (0, 255),
}


class SchemaError(ValueError):
    pass


def validate(x: np.ndarray) -> None:
    """Raw matrix in RAW_COLUMNS order: shape, finiteness, ranges, one-hot blocks."""
    if x.ndim != 2 or x.shape[1] != len(RAW_COLUMNS):
        raise SchemaError(f"expected (n, {len(RAW_COLUMNS)}) raw features, got {x.shape}")
    if not np.isfinite(x).all():
        raise SchemaError("non-finite values")
    for col, (lo, hi) in RANGES.items():
        v = x[:, RAW_COLUMNS.index(col)]
        if (v < lo).any() or (v > hi).any():
            raise SchemaError(f"{col} outside [{lo}, {hi}]")
    n = len(NUMERIC)
    for name, block in (("wilderness", x[:, n : n + 4]), ("soil", x[:, n + 4 :])):
        if not np.isin(block, (0, 1)).all() or (block.sum(1) > 1).any():
            raise SchemaError(f"{name} columns must be one-hot (at most one 1)")


def engineer(x: np.ndarray) -> np.ndarray:
    """(n, 54) raw -> (n, 19) float32 features, same definitions as the prototype."""
    c = {name: x[:, i].astype(np.float64) for i, name in enumerate(NUMERIC)}
    n = len(NUMERIC)
    wild = x[:, n : n + 4]
    soil = x[:, n + 4 : n + 44]
    # prototype keeps the *last* active column; with valid one-hot rows that is the only one
    w_id = np.where(wild.any(1), 4 - np.argmax(wild[:, ::-1], axis=1), 0)
    s_id = np.where(soil.any(1), 40 - np.argmax(soil[:, ::-1], axis=1), 0)
    h, v, el = (
        c["Horizontal_Distance_To_Hydrology"],
        c["Vertical_Distance_To_Hydrology"],
        c["Elevation"],
    )
    road, fire = c["Horizontal_Distance_To_Roadways"], c["Horizontal_Distance_To_Fire_Points"]
    a = np.radians(c["Aspect"])
    hs = (c["Hillshade_9am"], c["Hillshade_Noon"], c["Hillshade_3pm"])
    cols = [
        el,
        c["Slope"],
        np.sin(a),
        np.cos(a),
        h,
        v,
        np.sqrt(h * h + v * v),
        el - v,
        el - 0.2 * h,
        road,
        fire,
        road + fire,
        road - fire,
        *hs,
        (hs[0] + hs[1] + hs[2]) / 3,
        w_id,
        s_id,
    ]
    return np.column_stack(cols).astype(np.float32)


def raw_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[RAW_COLUMNS].to_numpy(dtype=np.float64)


assert len(FEATURES) == 19 and set(CATEGORICAL) <= set(FEATURES)
assert len(WILDERNESS) == 4 and len(SOIL) == 40

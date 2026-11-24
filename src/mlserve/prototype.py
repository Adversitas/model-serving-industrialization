"""The data scientist's prototype, kept as the reference implementation.

This is the kind of code that comes out of a notebook: correct, readable, and
row-by-row. It is the specification the production code must match (see the
parity tests) and the baseline the benchmarks measure against. Do not optimise it.
"""

from __future__ import annotations

import math

import pandas as pd

from .data import SOIL, WILDERNESS


def engineer_row(row: pd.Series) -> dict:
    h = row["Horizontal_Distance_To_Hydrology"]
    v = row["Vertical_Distance_To_Hydrology"]
    wilderness = 0
    for i, c in enumerate(WILDERNESS):
        if row[c] == 1:
            wilderness = i + 1
    soil = 0
    for i, c in enumerate(SOIL):
        if row[c] == 1:
            soil = i + 1
    aspect = math.radians(row["Aspect"])
    return {
        "elevation": row["Elevation"],
        "slope": row["Slope"],
        "aspect_sin": math.sin(aspect),
        "aspect_cos": math.cos(aspect),
        "hydro_h": h,
        "hydro_v": v,
        "hydro_dist": math.sqrt(h * h + v * v),
        "elev_minus_hydro_v": row["Elevation"] - v,
        "elev_minus_hydro_h": row["Elevation"] - 0.2 * h,
        "road_dist": row["Horizontal_Distance_To_Roadways"],
        "fire_dist": row["Horizontal_Distance_To_Fire_Points"],
        "road_plus_fire": row["Horizontal_Distance_To_Roadways"]
        + row["Horizontal_Distance_To_Fire_Points"],
        "road_minus_fire": row["Horizontal_Distance_To_Roadways"]
        - row["Horizontal_Distance_To_Fire_Points"],
        "hillshade_9am": row["Hillshade_9am"],
        "hillshade_noon": row["Hillshade_Noon"],
        "hillshade_3pm": row["Hillshade_3pm"],
        "hillshade_mean": (row["Hillshade_9am"] + row["Hillshade_Noon"] + row["Hillshade_3pm"]) / 3,
        "wilderness": wilderness,
        "soil": soil,
    }


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(list(df.apply(engineer_row, axis=1)))


def predict_one_by_one(model, df: pd.DataFrame) -> list[int]:
    """How the prototype served predictions: one DataFrame row at a time."""
    out = []
    for _, row in df.iterrows():
        x = engineer(row.to_frame().T)
        out.append(int(model.predict(x.to_numpy())[0]))
    return out

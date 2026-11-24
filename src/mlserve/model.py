"""Training, ONNX export with a parity check, and the runtime predictor."""

from __future__ import annotations

import contextlib
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import features
from .data import CLASSES, TARGET


def train(train_df: pd.DataFrame, seed: int = 0, max_iter: int = 200):
    """Gradient-boosted trees. ``l2_regularization=1.0`` matters: with ~0 the multiclass
    model became unstable on this data (test accuracy peaked at 90.6 % then fell)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    x = features.engineer(features.raw_matrix(train_df))
    y = train_df[TARGET].to_numpy()
    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.1,
        max_leaf_nodes=255,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    t0 = time.perf_counter()
    model.fit(x, y)
    return model, time.perf_counter() - t0


def linear_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    xtr = features.engineer(features.raw_matrix(train_df))
    xte = features.engineer(features.raw_matrix(test_df))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
    clf.fit(xtr, train_df[TARGET])
    return scores(test_df[TARGET].to_numpy(), clf.predict(xte))


def scores(y, pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


@contextlib.contextmanager
def _int_bool_attributes():
    """skl2onnx <= 1.20 passes numpy booleans (``missing_go_to_left`` of recent
    scikit-learn trees) into int attributes, which protobuf rejects. Cast them,
    only for the duration of the export."""
    from skl2onnx.common import _container

    original = _container.make_node

    def make_node(op_type, inputs, outputs, *args, **attrs):
        for k, v in attrs.items():
            if isinstance(v, (list, tuple)) and any(isinstance(e, (bool, np.bool_)) for e in v):
                attrs[k] = [int(e) for e in v]
        return original(op_type, inputs, outputs, *args, **attrs)

    _container.make_node = make_node
    try:
        yield
    finally:
        _container.make_node = original


def export_onnx(model, path: Path) -> None:
    from skl2onnx import to_onnx

    sample = np.zeros((1, len(features.FEATURES)), dtype=np.float32)
    with _int_bool_attributes():
        onx = to_onnx(model, sample, options={id(model): {"zipmap": False}}, target_opset=17)
    path.write_bytes(onx.SerializeToString())


class Predictor:
    """ONNX Runtime session + validation + feature engineering. Thread-safe for inference."""

    def __init__(self, onnx_path: Path, threads: int = 1):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_path), opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.version = hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest()[:12]

    def predict_proba_features(self, x: np.ndarray) -> np.ndarray:
        _, proba = self.session.run(None, {self.input_name: x.astype(np.float32, copy=False)})
        return proba

    def predict_raw(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features.validate(raw)
        proba = self.predict_proba_features(features.engineer(raw))
        return proba.argmax(1) + 1, proba


@dataclass
class Parity:
    max_abs_proba_diff: float
    label_agreement: float
    n: int


def parity(model, pred: Predictor, x: np.ndarray) -> Parity:
    p_sk = model.predict_proba(x)
    p_ox = pred.predict_proba_features(x)
    return Parity(
        float(np.abs(p_sk - p_ox).max()),
        float((p_sk.argmax(1) == p_ox.argmax(1)).mean()),
        len(x),
    )


def model_card(path: Path, **info) -> None:
    import onnxruntime
    import sklearn

    card = {
        "task": "forest cover type, 7 classes",
        "classes": CLASSES,
        "features": features.FEATURES,
        "input_schema": "54 raw UCI CoverType columns, see mlserve.data.RAW_COLUMNS",
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        **info,
    }
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")

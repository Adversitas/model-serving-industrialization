"""Benchmarks: prototype vs production, sklearn vs ONNX Runtime, direct vs micro-batched API."""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import features, prototype
from .model import Predictor


def _rate(fn, n_rows: int, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return n_rows / best


def _per_call_ms(fn, n: int = 300, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


def feature_bench(df: pd.DataFrame) -> dict:
    proto_rows = df.iloc[:2000]
    raw = features.raw_matrix(df)
    p = prototype.engineer(proto_rows)[features.FEATURES].to_numpy()
    v = features.engineer(features.raw_matrix(proto_rows))
    return {
        "prototype_rows_per_s": _rate(lambda: prototype.engineer(proto_rows), len(proto_rows), 1),
        "vectorised_rows_per_s": _rate(lambda: features.engineer(raw), len(raw)),
        "max_abs_diff_vs_prototype": float(np.abs(p - v).max()),
        "max_rel_diff_vs_prototype": float((np.abs(p - v) / np.maximum(np.abs(p), 1)).max()),
    }


def inference_bench(model, pred: Predictor, df: pd.DataFrame) -> dict:
    raw = features.raw_matrix(df)
    x = features.engineer(raw)
    one_raw, one_x = raw[:1], x[:1]
    proto_rows = df.iloc[:200]
    t0 = time.perf_counter()
    prototype.predict_one_by_one(model, proto_rows)
    proto_ms = 1000 * (time.perf_counter() - t0) / len(proto_rows)
    return {
        "prototype_row_loop_ms_per_row": proto_ms,
        "sklearn_batch1_ms": _per_call_ms(lambda: model.predict_proba(one_x)),
        "onnx_batch1_ms": _per_call_ms(lambda: pred.predict_proba_features(one_x)),
        "onnx_batch1_end_to_end_ms": _per_call_ms(lambda: pred.predict_raw(one_raw)),
        "sklearn_bulk_rows_per_s": _rate(lambda: model.predict_proba(x), len(x)),
        "onnx_bulk_rows_per_s": _rate(lambda: pred.predict_proba_features(x), len(x)),
        "bulk_rows": len(x),
    }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """uvicorn in a background thread (benchmarks and integration tests)."""

    def __init__(self, app):
        import uvicorn

        self.port = _free_port()
        cfg = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        while not self.server.started:
            time.sleep(0.02)
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


class ProcessServer:
    """The API in its own Python process, so the load generator does not share its GIL."""

    def __init__(self, onnx_path: Path):
        self.port = _free_port()
        code = (
            "import sys, uvicorn; from pathlib import Path; from mlserve.api import create_app; "
            "uvicorn.run(create_app(Path(sys.argv[1])), host='127.0.0.1', "
            "port=int(sys.argv[2]), log_level='warning')"
        )
        self.cmd = [sys.executable, "-c", code, str(onnx_path), str(self.port)]
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> str:
        import httpx

        self.proc = subprocess.Popen(self.cmd)
        base = f"http://127.0.0.1:{self.port}"
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                if httpx.get(base + "/health", timeout=1).status_code == 200:
                    return base
            except httpx.HTTPError:
                time.sleep(0.2)
        self.proc.kill()
        raise RuntimeError("API process did not start")

    def __exit__(self, *exc) -> None:
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=10)


def _load_threads(url: str, payloads: list[list[float]], concurrency: int) -> dict:
    """Closed-loop load: ``concurrency`` threads, each with its own keep-alive client.

    (An asyncio httpx client was the bottleneck on Windows: it serialised requests and
    added ~7 ms each, so the server never saw concurrent traffic.)"""
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    chunks = [payloads[i::concurrency] for i in range(concurrency)]

    def worker(chunk: list[list[float]]) -> list[float]:
        out = []
        with httpx.Client(timeout=30) as client:
            client.post(url, json={"values": chunk[0]}).raise_for_status()  # warm-up
            for p in chunk:
                t0 = time.perf_counter()
                client.post(url, json={"values": p}).raise_for_status()
                out.append(time.perf_counter() - t0)
        return out

    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as ex:
        lat = [x for part in ex.map(worker, chunks) for x in part]
    wall = time.perf_counter() - t0
    ms = 1000 * np.array(lat)
    return {
        "requests": len(lat),
        "concurrency": concurrency,
        "throughput_rps": len(lat) / wall,
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
    }


def api_bench(onnx_path: Path, df: pd.DataFrame, n_requests: int = 1000, levels=(1, 8, 32)) -> list:
    import httpx

    payloads = features.raw_matrix(df.iloc[:n_requests]).tolist()
    out = []
    for endpoint in ("/predict", "/predict/batched"):
        for c in levels:
            with ProcessServer(onnx_path) as base:
                r = _load_threads(base + endpoint, payloads, c)
                st = httpx.get(base + "/stats", timeout=5).json()
            r["endpoint"] = endpoint
            r["mean_batch"] = st["mean_batch"] if endpoint.endswith("batched") else 1.0
            r["max_batch"] = st["max_batch"] if endpoint.endswith("batched") else 1
            out.append(r)
    return out

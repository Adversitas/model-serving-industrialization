"""FastAPI service: validated input, direct and micro-batched prediction endpoints.

``POST /predict``          runs the model once per HTTP request;
``POST /predict/batched``  queues the request; a background task groups whatever
                           arrived within ``max_wait_ms`` (up to ``max_batch`` rows)
                           into one ONNX call, then answers every waiter.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import CLASSES, RAW_COLUMNS
from .features import SchemaError, engineer, validate
from .model import Predictor


class Record(BaseModel):
    values: list[float] = Field(
        ..., min_length=len(RAW_COLUMNS), max_length=len(RAW_COLUMNS), description="54 raw columns"
    )


class Prediction(BaseModel):
    cover_type: int
    label: str
    probability: float
    model_version: str


class MicroBatcher:
    def __init__(self, predictor: Predictor, max_batch: int = 64, max_wait_ms: float = 2.0):
        self.predictor = predictor
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.queue: asyncio.Queue = asyncio.Queue()
        self.batches: list[int] = []
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def submit(self, feats: np.ndarray) -> np.ndarray:
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((feats, fut))
        return await fut

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            items = [await self.queue.get()]
            deadline = loop.time() + self.max_wait
            while len(items) < self.max_batch:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(self.queue.get(), timeout))
                except asyncio.TimeoutError:  # noqa: UP041 (Python 3.10 alias)
                    break
            x = np.concatenate([f for f, _ in items])
            try:
                proba = self.predictor.predict_proba_features(x)
            except Exception as exc:  # pragma: no cover - surfaced to every waiter
                for _, fut in items:
                    fut.set_exception(exc)
                continue
            self.batches.append(len(items))
            for i, (_, fut) in enumerate(items):
                fut.set_result(proba[i])


def _features(rec: Record) -> np.ndarray:
    raw = np.asarray([rec.values], dtype=np.float64)
    try:
        validate(raw)
    except SchemaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return engineer(raw)


def _answer(proba: np.ndarray, version: str) -> Prediction:
    k = int(np.argmax(proba))
    return Prediction(
        cover_type=k + 1, label=CLASSES[k], probability=float(proba[k]), model_version=version
    )


def create_app(onnx_path: Path, max_batch: int = 64, max_wait_ms: float = 2.0) -> FastAPI:
    predictor = Predictor(onnx_path)
    batcher = MicroBatcher(predictor, max_batch, max_wait_ms)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        batcher.start()
        yield
        await batcher.stop()

    app = FastAPI(title="cover-type model", version=predictor.version, lifespan=lifespan)
    app.state.batcher = batcher

    @app.get("/stats")
    def stats() -> dict:
        sizes = batcher.batches
        return {
            "batches": len(sizes),
            "mean_batch": float(np.mean(sizes)) if sizes else 0.0,
            "max_batch": int(max(sizes)) if sizes else 0,
        }

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": predictor.version}

    @app.post("/predict", response_model=Prediction)
    def predict(rec: Record) -> Prediction:
        return _answer(predictor.predict_proba_features(_features(rec))[0], predictor.version)

    @app.post("/predict/batched", response_model=Prediction)
    async def predict_batched(rec: Record) -> Prediction:
        return _answer(await batcher.submit(_features(rec)), predictor.version)

    return app

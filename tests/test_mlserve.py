import asyncio

import httpx
import numpy as np
import pytest

from mlserve import data, features, model, prototype
from mlserve.api import create_app
from mlserve.bench import Server


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    df = data.sample_frame(1500, seed=3)
    tr, te = data.split(df)
    clf, _ = model.train(tr, max_iter=15)
    path = tmp_path_factory.mktemp("art") / "model.onnx"
    model.export_onnx(clf, path)
    return clf, model.Predictor(path), path, te


def test_vectorised_features_match_prototype():
    df = data.sample_frame(400, seed=7)
    p = prototype.engineer(df)[features.FEATURES].to_numpy()
    v = features.engineer(features.raw_matrix(df))
    np.testing.assert_allclose(v, p, rtol=1e-6, atol=1e-3)


def test_rows_without_category_get_id_zero():
    df = data.sample_frame(5)
    df.loc[0, data.SOIL] = 0
    v = features.engineer(features.raw_matrix(df))
    assert v[0, features.FEATURES.index("soil")] == 0
    assert prototype.engineer_row(df.iloc[0])["soil"] == 0


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda x: x[:, :10], "expected"),
        (lambda x: np.where(np.arange(x.shape[1]) == 1, 400, x), "Aspect"),
        (lambda x: np.where(np.arange(x.shape[1]) == 0, np.nan, x), "non-finite"),
        (
            lambda x: np.where(np.arange(x.shape[1]) == 10, 1, np.where(np.arange(54) == 11, 1, x)),
            "one-hot",
        ),
    ],
)
def test_validation_rejects_bad_input(mutate, msg):
    raw = features.raw_matrix(data.sample_frame(3))
    with pytest.raises(features.SchemaError, match=msg):
        features.validate(mutate(raw))


def test_onnx_parity(trained):
    clf, pred, _, te = trained
    x = features.engineer(features.raw_matrix(te))
    par = model.parity(clf, pred, x)
    assert par.label_agreement >= 0.995 and par.max_abs_proba_diff < 1e-3


def test_predict_raw_returns_classes_1_to_7(trained):
    _, pred, _, te = trained
    labels, proba = pred.predict_raw(features.raw_matrix(te.iloc[:20]))
    assert set(labels) <= set(range(1, 8)) and np.allclose(proba.sum(1), 1, atol=1e-5)


def test_api_direct_and_batched_agree(trained):
    _, pred, path, te = trained
    rows = features.raw_matrix(te.iloc[:40]).tolist()
    app = create_app(path, max_batch=16, max_wait_ms=5)

    async def go(base):
        async with httpx.AsyncClient(base_url=base) as c:
            assert (await c.get("/health")).json()["model_version"] == pred.version
            direct = [(await c.post("/predict", json={"values": r})).json() for r in rows]
            batched = await asyncio.gather(
                *(c.post("/predict/batched", json={"values": r}) for r in rows)
            )
            bad = await c.post("/predict", json={"values": rows[0][:5]})
            return direct, [b.json() for b in batched], bad.status_code

    with Server(app) as base:
        direct, batched, bad = asyncio.run(go(base))
    assert [d["cover_type"] for d in direct] == [b["cover_type"] for b in batched]
    assert bad == 422
    assert max(app.state.batcher.batches) > 1  # concurrent requests were grouped


def test_api_rejects_invalid_one_hot(trained):
    _, _, path, te = trained
    row = features.raw_matrix(te.iloc[:1])[0]
    row[10:14] = 1  # four wilderness areas at once
    app = create_app(path)

    async def go(base):
        async with httpx.AsyncClient(base_url=base) as c:
            return (await c.post("/predict", json={"values": row.tolist()})).status_code

    with Server(app) as base:
        assert asyncio.run(go(base)) == 422

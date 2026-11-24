# Results

Train 464,809 rows, test 116,203 rows (stratified 80/20 split).

## Model quality (test set)

| Model | Accuracy % | Macro-F1 % |
|---|---|---|
| Logistic regression on engineered features | 71.42 | 50.19 |
| HistGradientBoosting (200 iterations), scikit-learn | 97.35 | 95.13 |
| Same model exported to ONNX Runtime | 97.35 | 95.13 |

ONNX parity on 116,203 test rows: label agreement 100.000 %, max |Δp| 3.67e-06. ONNX file 27.3 MB, training 75 s.

## Prototype vs production code

| Step | Prototype | Production | Speed-up |
|---|---|---|---|
| Feature engineering (rows/s) | 3,466 | 1,187,867 | 343× |
| One prediction, raw row in (ms) | 73.11 | 0.659 | 111× |

Feature parity with the prototype: max relative difference 5.7e-08 (float32 rounding).

## scikit-learn vs ONNX Runtime (same model)

| Mode | scikit-learn | ONNX Runtime |
|---|---|---|
| batch of 1, ms (median) | 70.795 | 0.183 |
| bulk 116,203 rows, rows/s | 21,187 | 8,028 |

## HTTP API under load (uvicorn, 1 worker, separate process; client on the same machine)

| Endpoint | Concurrency | Throughput (req/s) | p50 ms | p95 ms | p99 ms | Mean batch | Max batch |
|---|---|---|---|---|---|---|---|
| `/predict` | 1 | 209 | 4.5 | 5.6 | 6.2 | 1.0 | 1 |
| `/predict` | 8 | 250 | 26.6 | 37.1 | 42.7 | 1.0 | 1 |
| `/predict` | 32 | 167 | 105.6 | 124.6 | 152.5 | 1.0 | 1 |
| `/predict/batched` | 1 | 62 | 15.7 | 20.1 | 22.4 | 1.0 | 1 |
| `/predict/batched` | 8 | 293 | 21.8 | 28.1 | 30.1 | 2.6 | 7 |
| `/predict/batched` | 32 | 210 | 60.5 | 90.6 | 106.1 | 7.1 | 22 |

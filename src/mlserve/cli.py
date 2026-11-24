"""Command line.

mlserve train   [--synthetic]   train, evaluate, export ONNX, check parity, write model card
mlserve bench   [--synthetic]   feature, inference and API load benchmarks
mlserve report                  render reports/results.md from the JSON reports
mlserve serve                   run the API with uvicorn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import bench, data, features, model


def _frames(args):
    df = data.sample_frame(4000, seed=1) if args.synthetic else data.load(args.data_dir)
    return data.split(df, seed=0)


def cmd_train(args) -> dict:
    train_df, test_df = _frames(args)
    clf, fit_s = model.train(train_df, max_iter=30 if args.synthetic else 200)
    x_te = features.engineer(features.raw_matrix(test_df))
    y_te = test_df[data.TARGET].to_numpy()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    onnx_path = args.artifacts / "model.onnx"
    model.export_onnx(clf, onnx_path)
    pred = model.Predictor(onnx_path)
    par = model.parity(clf, pred, x_te)
    onnx_scores = model.scores(y_te, pred.predict_proba_features(x_te).argmax(1) + 1)
    res = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "fit_seconds": fit_s,
        "n_iter": int(clf.n_iter_),
        "hgb_sklearn": model.scores(y_te, clf.predict(x_te)),
        "hgb_onnx": onnx_scores,
        "linear_baseline": model.linear_baseline(train_df, test_df),
        "parity": par.__dict__,
        "onnx_bytes": onnx_path.stat().st_size,
        "model_version": pred.version,
    }
    model.model_card(args.artifacts / "model_card.json", metrics=res)
    import joblib

    joblib.dump(clf, args.artifacts / "model.joblib")
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "train.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def cmd_bench(args) -> dict:
    import joblib

    _, test_df = _frames(args)
    clf = joblib.load(args.artifacts / "model.joblib")
    pred = model.Predictor(args.artifacts / "model.onnx")
    res = {
        "features": bench.feature_bench(test_df),
        "inference": bench.inference_bench(clf, pred, test_df),
        "api": bench.api_bench(
            args.artifacts / "model.onnx",
            test_df,
            n_requests=200 if args.synthetic else 2000,
            levels=(1, 8) if args.synthetic else (1, 8, 32),
        ),
    }
    (args.reports / "bench.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def render(reports: Path) -> str:
    t = json.loads((reports / "train.json").read_text(encoding="utf-8"))
    b = json.loads((reports / "bench.json").read_text(encoding="utf-8"))
    f, i = b["features"], b["inference"]
    lines = [
        "# Results",
        "",
        f"Train {t['n_train']:,} rows, test {t['n_test']:,} rows (stratified 80/20 split).",
        "",
        "## Model quality (test set)",
        "",
        "| Model | Accuracy % | Macro-F1 % |",
        "|---|---|---|",
        "| Logistic regression on engineered features | "
        f"{100 * t['linear_baseline']['accuracy']:.2f} | "
        f"{100 * t['linear_baseline']['macro_f1']:.2f} |",
        f"| HistGradientBoosting ({t['n_iter']} iterations), scikit-learn | "
        f"{100 * t['hgb_sklearn']['accuracy']:.2f} | {100 * t['hgb_sklearn']['macro_f1']:.2f} |",
        f"| Same model exported to ONNX Runtime | {100 * t['hgb_onnx']['accuracy']:.2f} | "
        f"{100 * t['hgb_onnx']['macro_f1']:.2f} |",
        "",
        f"ONNX parity on {t['parity']['n']:,} test rows: label agreement "
        f"{100 * t['parity']['label_agreement']:.3f} %, "
        f"max |Δp| {t['parity']['max_abs_proba_diff']:.2e}. "
        f"ONNX file {t['onnx_bytes'] / 1e6:.1f} MB, training {t['fit_seconds']:.0f} s.",
        "",
        "## Prototype vs production code",
        "",
        "| Step | Prototype | Production | Speed-up |",
        "|---|---|---|---|",
        f"| Feature engineering (rows/s) | {f['prototype_rows_per_s']:,.0f} | "
        f"{f['vectorised_rows_per_s']:,.0f} | "
        f"{f['vectorised_rows_per_s'] / f['prototype_rows_per_s']:,.0f}× |",
        f"| One prediction, raw row in (ms) | {i['prototype_row_loop_ms_per_row']:.2f} | "
        f"{i['onnx_batch1_end_to_end_ms']:.3f} | "
        f"{i['prototype_row_loop_ms_per_row'] / i['onnx_batch1_end_to_end_ms']:,.0f}× |",
        "",
        f"Feature parity with the prototype: max relative difference "
        f"{f['max_rel_diff_vs_prototype']:.1e} (float32 rounding).",
        "",
        "## scikit-learn vs ONNX Runtime (same model)",
        "",
        "| Mode | scikit-learn | ONNX Runtime |",
        "|---|---|---|",
        f"| batch of 1, ms (median) | {i['sklearn_batch1_ms']:.3f} | {i['onnx_batch1_ms']:.3f} |",
        f"| bulk {i['bulk_rows']:,} rows, rows/s | {i['sklearn_bulk_rows_per_s']:,.0f} | "
        f"{i['onnx_bulk_rows_per_s']:,.0f} |",
        "",
        "## HTTP API under load (uvicorn, 1 worker, separate process; client on the same machine)",
        "",
        "| Endpoint | Concurrency | Throughput (req/s) | p50 ms | p95 ms | p99 ms "
        "| Mean batch | Max batch |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in b["api"]:
        lines.append(
            f"| `{a['endpoint']}` | {a['concurrency']} | {a['throughput_rps']:,.0f} | "
            f"{a['p50_ms']:.1f} | {a['p95_ms']:.1f} | {a['p99_ms']:.1f} | "
            f"{a['mean_batch']:.1f} | {a.get('max_batch', 1)} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="mlserve", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train", "bench", "report", "serve"):
        s = sub.add_parser(name)
        s.add_argument("--data-dir", type=Path, default=Path("data"))
        s.add_argument("--artifacts", type=Path, default=Path("artifacts"))
        s.add_argument("--reports", type=Path, default=Path("reports"))
        s.add_argument("--synthetic", action="store_true")
        if name == "serve":
            s.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    if args.cmd == "train":
        print(json.dumps(cmd_train(args), indent=2))
    elif args.cmd == "bench":
        cmd_bench(args)
        print(render(args.reports))
    elif args.cmd == "report":
        text = render(args.reports)
        (args.reports / "results.md").write_text(text, encoding="utf-8")
        print(text)
    else:
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(args.artifacts / "model.onnx"), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

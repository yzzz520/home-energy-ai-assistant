from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from edge_features import (
    FEATURE_NAMES,
    build_feature_matrix,
    conversion_delta,
    latency_summary,
    regression_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "时序预测"
    / "results"
    / "validation_300_households_predictions.csv"
)
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("results")
DEFAULT_ARTIFACT_DIR = Path(__file__).with_name("artifacts")


def synthetic_prediction_frame() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=60 * 48,
        freq="30min",
    )
    slot = np.arange(len(timestamps)) % 48
    hour = slot / 2
    daily_shape = (
        0.12
        + 0.22 * np.exp(-0.5 * ((hour - 18) / 2.8) ** 2)
        + 0.08 * np.exp(-0.5 * ((hour - 7) / 2.2) ** 2)
    )
    weekend_factor = np.where(
        timestamps.day_of_week >= 5,
        0.9 + 0.08 * np.sin(np.arange(len(timestamps)) / 19),
        1.0,
    )
    day_factor = 0.94 + 0.08 * np.sin(np.arange(len(timestamps)) / (48 * 4))
    base = daily_shape * weekend_factor * day_factor
    rng = np.random.default_rng(42)
    daily_naive = np.roll(base, 48)
    weekly_naive = np.roll(base, 48 * 7)
    daily_naive[: 48 * 7] = base[: 48 * 7]
    weekly_naive[: 48 * 7] = base[: 48 * 7]
    actual = np.maximum(base + rng.normal(0, 0.025, len(base)), 0)
    return pd.DataFrame(
        {
            "forecast_timestamp": timestamps,
            "actual": actual,
            "seasonal_naive_daily": daily_naive,
            "seasonal_naive_weekly": weekly_naive,
        }
    )


def load_prediction_frame(
    data_path: Path,
    *,
    synthetic: bool,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if synthetic:
        frame = synthetic_prediction_frame()
        source = "synthetic"
    else:
        frame = pd.read_csv(
            data_path,
            encoding="utf-8-sig",
            usecols=[
                "forecast_timestamp",
                "actual",
                "seasonal_naive_daily",
                "seasonal_naive_weekly",
            ],
        )
        source = str(data_path)

    frame["forecast_timestamp"] = pd.to_datetime(frame["forecast_timestamp"])
    frame = frame.dropna().sort_values("forecast_timestamp").reset_index(drop=True)
    dates = np.sort(frame["forecast_timestamp"].dt.normalize().unique())
    split_index = min(max(int(len(dates) * train_ratio), 1), len(dates) - 1)
    split_date = dates[split_index]
    train = frame.loc[frame["forecast_timestamp"].lt(split_date)].copy()
    test = frame.loc[frame["forecast_timestamp"].ge(split_date)].copy()
    return train, test, source


def sample_arrays(
    features: np.ndarray,
    target: np.ndarray,
    *,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(features) <= max_rows:
        return features, target
    indices = np.random.default_rng(seed).choice(
        len(features),
        size=max_rows,
        replace=False,
    )
    return features[indices], target[indices]


def build_model() -> keras.Model:
    return keras.Sequential(
        [
            keras.Input(shape=(len(FEATURE_NAMES),), name="features"),
            keras.layers.Dense(16, activation="relu", name="hidden_1"),
            keras.layers.Dense(8, activation="relu", name="hidden_2"),
            keras.layers.Dense(1, name="load_kwh"),
        ],
        name="edge_load_mlp",
    )


def train_model(
    train_features: np.ndarray,
    train_target: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[keras.Model, dict[str, float | int]]:
    validation_size = max(int(len(train_features) * 0.1), 96)
    fit_features = train_features[:-validation_size]
    fit_target = train_target[:-validation_size]
    validation_features = train_features[-validation_size:]
    validation_target = train_target[-validation_size:]

    keras.utils.set_random_seed(seed)
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=keras.losses.Huber(delta=0.5),
        metrics=["mae"],
    )
    history = model.fit(
        fit_features,
        fit_target,
        validation_data=(validation_features, validation_target),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_mae",
                patience=3,
                restore_best_weights=True,
            )
        ],
    )
    return model, {
        "epochs_ran": len(history.history["loss"]),
        "best_val_mae_kwh": float(min(history.history["val_mae"])),
    }


def convert_model(model: keras.Model) -> dict[str, bytes]:
    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_model = float_converter.convert()

    dynamic_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    dynamic_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    dynamic_model = dynamic_converter.convert()
    return {
        "tflite_float32": float_model,
        "tflite_dynamic_range": dynamic_model,
    }


def predict_tflite_batch(model_content: bytes, features: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(
        model_content=model_content,
        num_threads=1,
    )
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.resize_tensor_input(
        input_detail["index"],
        [len(features), len(FEATURE_NAMES)],
        strict=False,
    )
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_detail["index"], features.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_detail["index"]).reshape(-1)


def benchmark_keras_latency(
    model: keras.Model,
    sample: np.ndarray,
    *,
    warmup: int,
    repeats: int,
) -> list[float]:
    serving = tf.function(
        lambda values: model(values, training=False),
        input_signature=[tf.TensorSpec([1, len(FEATURE_NAMES)], tf.float32)],
    )
    batch = tf.convert_to_tensor(sample.reshape(1, -1), dtype=tf.float32)
    for _ in range(warmup):
        serving(batch)
    samples_ms: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        serving(batch)
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    return samples_ms


def benchmark_tflite_latency(
    model_content: bytes,
    sample: np.ndarray,
    *,
    warmup: int,
    repeats: int,
) -> list[float]:
    interpreter = tf.lite.Interpreter(
        model_content=model_content,
        num_threads=1,
    )
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.allocate_tensors()
    batch = sample.reshape(1, -1).astype(np.float32)

    def invoke() -> None:
        interpreter.set_tensor(input_detail["index"], batch)
        interpreter.invoke()
        interpreter.get_tensor(output_detail["index"])

    for _ in range(warmup):
        invoke()
    samples_ms: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        invoke()
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    return samples_ms


def model_weight_bytes(model: keras.Model) -> int:
    return int(
        sum(
            weight.size * weight.dtype.itemsize
            for weight in model.get_weights()
        )
    )


def write_markdown_report(report: dict[str, object], output_path: Path) -> None:
    lines = [
        "# Edge Model Benchmark",
        "",
        f"- Data source: `{report['data_source']}`",
        f"- Train rows used: {report['train_rows_used']:,}",
        f"- Test rows: {report['test_rows']:,}",
        f"- Parameters: {report['parameters']:,}",
        "",
        "| Variant | Size KB | MAE vs actual | MAE vs Keras | Median latency ms | P95 latency ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant_name, variant in report["variants"].items():
        lines.append(
            "| {name} | {size:.2f} | {mae:.6f} | {delta:.8f} | "
            "{median:.4f} | {p95:.4f} |".format(
                name=variant_name,
                size=variant["size_bytes"] / 1024,
                mae=variant["metrics_vs_actual"]["mae_kwh"],
                delta=variant["conversion_error_vs_keras"]["mae_kwh"],
                median=variant["latency"]["median_ms"],
                p95=variant["latency"]["p95_ms"],
            )
        )
    lines.extend(
        [
            "",
            "Latency is measured on the current CPU with batch size 1 and one thread.",
            "It is a local engineering baseline, not a phone or embedded-device benchmark.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if not args.synthetic and not args.data_path.exists():
        raise FileNotFoundError(f"prediction data not found: {args.data_path}")

    train_frame, test_frame, source = load_prediction_frame(
        args.data_path,
        synthetic=args.synthetic,
        train_ratio=args.train_ratio,
    )
    train_features, train_target = build_feature_matrix(train_frame)
    test_features, test_target = build_feature_matrix(test_frame)
    if train_target is None or test_target is None:
        raise ValueError("target column is required")

    train_features, train_target = sample_arrays(
        train_features,
        train_target,
        max_rows=args.max_train_rows,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    model, training = train_model(
        train_features,
        train_target,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    keras_path = args.artifact_dir / "load_mlp.keras"
    model.save(keras_path)

    converted_models = convert_model(model)
    converted_paths = {
        "tflite_float32": args.artifact_dir / "load_mlp_float32.tflite",
        "tflite_dynamic_range": args.artifact_dir / "load_mlp_dynamic_range.tflite",
    }
    for name, content in converted_models.items():
        converted_paths[name].write_bytes(content)

    keras_prediction = model.predict(
        test_features,
        batch_size=8192,
        verbose=0,
    ).reshape(-1)
    variants: dict[str, object] = {
        "keras": {
            "size_bytes": keras_path.stat().st_size,
            "weights_bytes": model_weight_bytes(model),
            "metrics_vs_actual": regression_metrics(
                test_target,
                keras_prediction,
            ),
            "conversion_error_vs_keras": conversion_delta(
                keras_prediction,
                keras_prediction,
            ),
            "latency": latency_summary(
                benchmark_keras_latency(
                    model,
                    test_features[0],
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            ),
        }
    }
    for name, content in converted_models.items():
        prediction = predict_tflite_batch(content, test_features)
        variants[name] = {
            "size_bytes": converted_paths[name].stat().st_size,
            "metrics_vs_actual": regression_metrics(test_target, prediction),
            "conversion_error_vs_keras": conversion_delta(
                keras_prediction,
                prediction,
            ),
            "latency": latency_summary(
                benchmark_tflite_latency(
                    content,
                    test_features[0],
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            ),
            "compression_ratio_vs_keras": (
                converted_paths[name].stat().st_size
                / keras_path.stat().st_size
            ),
        }

    report: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "platform": platform.platform(),
        "data_source": source,
        "train_rows_used": len(train_features),
        "test_rows": len(test_features),
        "parameters": int(model.count_params()),
        "training": training,
        "latency_protocol": {
            "batch_size": 1,
            "threads": 1,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "artifacts": {
            "keras": str(keras_path),
            **{
                name: str(path)
                for name, path in converted_paths.items()
            },
        },
        "variants": variants,
    }
    report_path = args.output_dir / "edge_benchmark.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(report, args.output_dir / "edge_benchmark.md")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and benchmark a small TFLite load model."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.67)
    parser.add_argument("--max-train-rows", type=int, default=250_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_benchmark(args)
    print(
        f"[OK] parameters={report['parameters']:,}, "
        f"test_rows={report['test_rows']:,}"
    )
    for name, variant in report["variants"].items():
        print(
            f"[{name}] size={variant['size_bytes'] / 1024:.2f} KiB, "
            f"MAE={variant['metrics_vs_actual']['mae_kwh']:.6f}, "
            f"median={variant['latency']['median_ms']:.4f} ms"
        )
    print(f"[OK] {args.output_dir / 'edge_benchmark.json'}")


if __name__ == "__main__":
    main()

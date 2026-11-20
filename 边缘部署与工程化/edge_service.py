from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from edge_features import interval_feature_vector


PROJECT_DIR = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_DIR / "static"
DEFAULT_MODEL_PATH = PROJECT_DIR / "artifacts" / "load_mlp_float32.tflite"


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    slot_index: int = Field(ge=0, le=47)
    weekday: int = Field(ge=0, le=6)
    seasonal_naive_daily_kwh: float = Field(ge=0, le=10)
    seasonal_naive_weekly_kwh: float = Field(ge=0, le=10)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    load_kwh: float
    inference_ms: float
    model_version: str = "edge_load_mlp_tflite_v1"


def load_interpreter_class():
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter
    except ImportError:
        try:
            import tensorflow as tf

            return tf.lite.Interpreter
        except ImportError as exc:
            raise RuntimeError(
                "Install ai-edge-litert or tensorflow to run the edge model"
            ) from exc


class TFLiteRunner:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {self.model_path}")
        interpreter_class = load_interpreter_class()
        model_content = self.model_path.read_bytes()
        self.interpreter = interpreter_class(
            model_content=model_content,
            num_threads=1,
        )
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.lock = threading.Lock()

    def predict(self, features: np.ndarray) -> float:
        sample = features.astype(np.float32).reshape(1, -1)
        with self.lock:
            self.interpreter.set_tensor(self.input_detail["index"], sample)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_detail["index"])
        return float(np.asarray(output).reshape(-1)[0])


class RunnerProtocol:
    def predict(self, features: np.ndarray) -> float:
        raise NotImplementedError


def create_app(
    *,
    runner: RunnerProtocol | None = None,
    model_path: str | Path | None = None,
) -> FastAPI:
    edge_runner = runner or TFLiteRunner(
        model_path or os.getenv("TFLITE_MODEL_PATH", DEFAULT_MODEL_PATH)
    )
    app = FastAPI(
        title="家庭负荷边缘预测 API",
        version="1.0.0",
        description="A small TFLite model for edge load inference.",
        docs_url=None,
        redoc_url=None,
    )
    app.state.runner = edge_runner
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/docs", include_in_schema=False)
    def docs() -> FileResponse:
        return FileResponse(STATIC_DIR / "docs.html")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": "tflite"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        features = interval_feature_vector(
            slot_index=request.slot_index,
            weekday=request.weekday,
            seasonal_naive_daily=request.seasonal_naive_daily_kwh,
            seasonal_naive_weekly=request.seasonal_naive_weekly_kwh,
        )
        started = time.perf_counter()
        prediction = edge_runner.predict(features)
        inference_ms = (time.perf_counter() - started) * 1000
        if not np.isfinite(prediction) or prediction < 0:
            raise HTTPException(
                status_code=500,
                detail="model returned an invalid load prediction",
            )
        return PredictionResponse(
            load_kwh=prediction,
            inference_ms=round(inference_ms, 4),
        )

    return app

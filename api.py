import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/plant_disease_model.keras"))
LABELS_PATH = Path(os.getenv("LABELS_PATH", "artifacts/class_names.json"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
TOP_K = int(os.getenv("TOP_K", "3"))

app = FastAPI(title="Plant Disease Predictor", version="1.0.0")

_model: tf.keras.Model | None = None
_class_names: List[str] | None = None


def _load_artifacts() -> tuple[tf.keras.Model, List[str]]:
    global _model, _class_names

    if _model is not None and _class_names is not None:
        return _model, _class_names

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

    _model = tf.keras.models.load_model(MODEL_PATH)
    with LABELS_PATH.open("r", encoding="utf-8") as f:
        _class_names = json.load(f)

    return _model, _class_names


def _prepare_image_bytes(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid image file") from exc

    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_arr = np.asarray(image, dtype=np.float32)
    image_arr = preprocess_input(image_arr)
    image_arr = np.expand_dims(image_arr, axis=0)
    return image_arr


@app.get("/health")
def health() -> Dict[str, Any]:
    model_exists = MODEL_PATH.exists()
    labels_exists = LABELS_PATH.exists()
    return {
        "status": "ok" if model_exists and labels_exists else "missing_artifacts",
        "model_path": str(MODEL_PATH),
        "labels_path": str(LABELS_PATH),
        "model_exists": model_exists,
        "labels_exists": labels_exists,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        model, class_names = _load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        image_batch = _prepare_image_bytes(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    probs = model.predict(image_batch, verbose=0)[0]
    best_idx = int(np.argmax(probs))

    top_k = min(TOP_K, len(class_names))
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_predictions = [
        {
            "class": class_names[int(i)],
            "confidence": float(probs[int(i)]),
        }
        for i in top_indices
    ]

    return {
        "predicted_class": class_names[best_idx],
        "confidence": float(probs[best_idx]),
        "top_predictions": top_predictions,
    }

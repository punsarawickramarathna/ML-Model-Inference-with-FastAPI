from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional
import numpy as np
import joblib
import json
import logging
import os
from fastapi.responses import HTMLResponse
from fastapi import Form

APP_TITLE = "ML Model API - Iris Classifier"
DESCRIPTION = "FastAPI app for Iris species prediction using a trained scikit-learn model."
VERSION = "1.0.0"

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
META_PATH = os.getenv("META_PATH", "model_meta.json")

# Logging (Bonus)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iris_api")

# Pydantic input/output schemas
class IrisInput(BaseModel):
    # Using simple names; map to sklearn ordering:
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")

class PredictionOutput(BaseModel):
    prediction: str
    confidence: Optional[float] = Field(
        default=None, description="Max class probability (0-1)"
    )
    probabilities: Optional[Dict[str, float]] = Field(
        default=None, description="Per-class probabilities"
    )

class BatchPredictionInput(BaseModel):
    items: List[IrisInput]

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]

# App
app = FastAPI(title=APP_TITLE, description=DESCRIPTION, version=VERSION)

model = None
meta = None
target_names: List[str] = []

# Startup: load model + metadata once
@app.on_event("startup")
def load_assets():
    global model, meta, target_names
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        model = None

    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        target_names = meta.get("target_names", [])
        logger.info(f"Loaded metadata from {META_PATH}")
    except Exception as e:
        logger.exception(f"Failed to load metadata: {e}")
        meta = None
        target_names = []

# Health check
@app.get("/", tags=["health"])
def health_check():
    return {
        "status": "healthy",
        "message": "ML Model API is running",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "meta_loaded": meta is not None,
    }

# Single prediction
@app.post("/predict", response_model=PredictionOutput, tags=["inference"])
def predict(input_data: IrisInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Train and save model.pkl first.")

    try:
        features = np.array([[  # order must match training
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width,
        ]], dtype=float)

        pred_idx = int(model.predict(features)[0])

        # If we have target_names, map index -> label
        if target_names and 0 <= pred_idx < len(target_names):
            label = target_names[pred_idx]
        else:
            label = str(pred_idx)

        # Confidence + probabilities if available
        probs = None
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]  # 1D array
            if target_names and len(target_names) == len(proba):
                probs = {target_names[i]: float(p) for i, p in enumerate(proba)}
            else:
                probs = {str(i): float(p) for i, p in enumerate(proba)}
            confidence = float(np.max(proba))

        return PredictionOutput(
            prediction=label,
            confidence=confidence,
            probabilities=probs
        )
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(e))


# Batch prediction (Bonus)
@app.post("/predict-batch", response_model=BatchPredictionOutput, tags=["inference"])
def predict_batch(payload: BatchPredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Train and save model.pkl first.")

    try:
        features = np.array([
            [
                item.sepal_length,
                item.sepal_width,
                item.petal_length,
                item.petal_width,
            ] for item in payload.items
        ], dtype=float)

        pred_indices = model.predict(features)
        outputs: List[PredictionOutput] = []

        has_proba = hasattr(model, "predict_proba")
        probas = model.predict_proba(features) if has_proba else None

        for i, pred_idx in enumerate(pred_indices):
            pred_idx = int(pred_idx)
            if target_names and 0 <= pred_idx < len(target_names):
                label = target_names[pred_idx]
            else:
                label = str(pred_idx)

            probs = None
            confidence = None
            if has_proba and probas is not None:
                row = probas[i]
                if target_names and len(target_names) == len(row):
                    probs = {target_names[j]: float(p) for j, p in enumerate(row)}
                else:
                    probs = {str(j): float(p) for j, p in enumerate(row)}
                confidence = float(np.max(row))

            outputs.append(PredictionOutput(
                prediction=label,
                confidence=confidence,
                probabilities=probs
            ))

        return BatchPredictionOutput(predictions=outputs)
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=400, detail=str(e))

# Model info / metadata
@app.get("/model-info", tags=["info"])
def model_info():
    if meta is None:
        # Fallback if no separate meta file
        info = {
            "model_type": "Unknown (metadata file missing)",
            "problem_type": "classification",
            "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "notes": "Run train_model.py to regenerate metadata."
        }
        return JSONResponse(content=info)

    return JSONResponse(content=meta)

@app.get("/form", response_class=HTMLResponse)
def form():
    return """
    <html>
        <body>
            <h2>Iris Prediction Form</h2>
            <form action="/predict-form" method="post">
                Sepal Length: <input type="text" name="sepal_length"><br>
                Sepal Width: <input type="text" name="sepal_width"><br>
                Petal Length: <input type="text" name="petal_length"><br>
                Petal Width: <input type="text" name="petal_width"><br>
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """

@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    confidence = float(np.max(model.predict_proba(features)))
    return f"<h2>Prediction: {prediction}</h2><h3>Confidence: {confidence:.2f}</h3>"
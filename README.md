# Iris Prediction API

A FastAPI-based REST API for predicting Iris flower species using a trained `RandomForestClassifier`.  
Includes logging, batch prediction, confidence scores, and a simple HTML form for testing.

---

## Features

- Single prediction endpoint (`/predict`)
- Batch prediction endpoint (`/predict-batch`)
- Returns prediction **confidence scores**
- Logging of predictions and errors
- HTML form for browser-based testing (`/form`)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/punsarawickramarathna/ML-Model-Inference-with-FastAPI.git
cd iris-prediction-api
````

2. Create a virtual environment:

```bash
python -m myenv
```

3. Install dependencies:

```bash
pip install fastapi uvicorn scikit-learn pandas numpy joblib
```

---

## Running the API

```bash
uvicorn main:app --reload
```

* API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Test form: [http://127.0.0.1:8000/form](http://127.0.0.1:8000/form)

---

## API Endpoints

### 1. Single Prediction

* **URL:** `/predict`
* **Method:** `POST`
* **Body Example:**

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

* **Response Example:**

```json
{
  "prediction": "setosa",
  "confidence": 0.98
}
```

---

### 2. Batch Prediction

* **URL:** `/predict-batch`
* **Method:** `POST`
* **Body Example:**

```json
{
  "inputs": [
    {"sepal_length":5.1, "sepal_width":3.5, "petal_length":1.4, "petal_width":0.2},
    {"sepal_length":6.2, "sepal_width":2.8, "petal_length":4.8, "petal_width":1.8}
  ]
}
```

* **Response Example:**

```json
{
  "predictions": ["setosa", "virginica"],
  "confidences": [0.98, 0.92]
}
```

---

### 3. HTML Form Testing

* **URL:** `/form` (GET)
* Submit values for `sepal_length`, `sepal_width`, `petal_length`, `petal_width`.
* Form sends data to `/predict-form` (POST) and displays prediction with confidence.

---

## Logging

* Logs all predictions, batch predictions, and errors.
* Format:

```
2025-08-18 10:30:25,123 - INFO - Prediction: setosa, Confidence: 0.98
```

---

## Requirements

* Python 3.8+
* FastAPI
* uvicorn
* scikit-learn
* numpy
* pydantic

---

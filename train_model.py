import json
import joblib
from datetime import datetime
from typing import Dict, Any

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import platform
import sklearn

MODEL_PATH = "model.pkl"
META_PATH = "model_meta.json"

def main():
    # 1) Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)  
    target_names = list(iris.target_names)    

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Build pipeline
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=300, random_state=42))
    ])

    # 4) Train
    pipe.fit(X_train, y_train)

    # 5) Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    # 6) Save model
    joblib.dump(pipe, MODEL_PATH)

    # 7) Save metadata for /model-info
    meta: Dict[str, Any] = {
        "model_type": "LogisticRegression (with StandardScaler in Pipeline)",
        "problem_type": "classification",
        "framework": "scikit-learn",
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "features": feature_names,
        "target_names": target_names,
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "metrics": {"test_accuracy": float(acc)},
        "notes": "Simple baseline model for Iris classification."
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {MODEL_PATH} and {META_PATH}")

if __name__ == "__main__":
    main()

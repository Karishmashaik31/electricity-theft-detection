"""Train the electricity theft classifier.

If you have a real labelled dataset (for example the SGCC dataset), put it in
backend/data/dataset.csv with one consumer per row: the first columns are the
daily readings and the LAST column is the label (1 = theft, 0 = normal), then run
    python train_model.py --csv data/dataset.csv
Without a CSV, a synthetic dataset is generated so the project runs out of the box.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

from features import FEATURE_NAMES, extract_features

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

DAYS = 30


def _normal_consumer(rng):
    base = rng.uniform(3, 25)
    weekly = 1 + 0.15 * np.sin(2 * np.pi * np.arange(DAYS) / 7 + rng.uniform(0, 6))
    noise = rng.normal(1, rng.uniform(0.08, 0.25), DAYS)
    x = base * weekly * noise
    if rng.random() < 0.1:  # honest vacation: a few low days
        start = rng.integers(0, DAYS - 6)
        x[start:start + rng.integers(3, 7)] *= rng.uniform(0.15, 0.4)
    return np.clip(x, 0, None)


def _theft_consumer(rng):
    x = _normal_consumer(rng)
    kind = rng.integers(0, 4)
    if kind == 0:    # meter tampered for the whole period
        x *= rng.uniform(0.2, 0.6)
    elif kind == 1:  # tampering starts part-way through
        start = rng.integers(5, DAYS - 5)
        x[start:] *= rng.uniform(0.1, 0.5)
    elif kind == 2:  # meter bypassed on random days
        mask = rng.random(DAYS) < rng.uniform(0.25, 0.5)
        x[mask] *= rng.uniform(0.0, 0.15)
    else:            # random daily reduction
        x *= rng.uniform(0.1, 0.8, DAYS)
    return np.clip(x, 0, None)


def make_synthetic(n=4000, theft_share=0.3, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        is_theft = rng.random() < theft_share
        readings = _theft_consumer(rng) if is_theft else _normal_consumer(rng)
        X.append(extract_features(readings))
        y.append(int(is_theft))
    return np.array(X), np.array(y)


def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    readings, y = data[:, :-1], data[:, -1].astype(int)
    readings = np.nan_to_num(readings, nan=0.0)
    return np.array([extract_features(r) for r in readings]), y


def train(csv_path=None):
    X, y = load_csv(csv_path) if csv_path else make_synthetic()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    metrics = {
        "model": "Random Forest",
        "data": "csv" if csv_path else "synthetic",
        "samples": int(len(y)),
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "precision": round(precision_score(y_te, pred), 4),
        "recall": round(recall_score(y_te, pred), 4),
        "f1": round(f1_score(y_te, pred), 4),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "feature_importance": {
            n: round(float(v), 4)
            for n, v in zip(FEATURE_NAMES, model.feature_importances_)
        },
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="labelled dataset (last column = label)")
    args = ap.parse_args()
    m = train(args.csv)
    print(json.dumps({k: v for k, v in m.items() if k != "feature_importance"}, indent=2))
    print("Saved model to", MODEL_PATH)

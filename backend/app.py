"""Flask backend for the electricity theft detection project."""
import csv
import io
import json
import math
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import joblib
from flask import Flask, jsonify, request

from features import FEATURE_NAMES, explain, extract_features
from train_model import METRICS_PATH, MODEL_PATH, train

FRONTEND = BASE.parent / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")

MIN_DAYS, MAX_DAYS = 14, 366

if not MODEL_PATH.exists():
    print("No trained model found, training one now...")
    train()
model = joblib.load(MODEL_PATH)


def risk_level(p):
    if p >= 0.7:
        return "High"
    if p >= 0.4:
        return "Medium"
    return "Low"


def parse_readings(values):
    """Validate a list of daily readings. Raises ValueError with a clear message."""
    try:
        nums = [float(v) for v in values]
    except (TypeError, ValueError):
        raise ValueError("Readings must be numbers.")
    if any(math.isnan(v) or math.isinf(v) for v in nums):
        raise ValueError("Readings contain invalid values.")
    if any(v < 0 for v in nums):
        raise ValueError("Readings cannot be negative.")
    if not MIN_DAYS <= len(nums) <= MAX_DAYS:
        raise ValueError(f"Provide between {MIN_DAYS} and {MAX_DAYS} daily readings "
                         f"(got {len(nums)}).")
    return nums


def score(readings):
    feats = extract_features(readings)
    p = float(model.predict_proba([feats])[0][1])
    return p, feats


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.get("/api/metrics")
def metrics():
    return jsonify(json.loads(METRICS_PATH.read_text()))


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    try:
        readings = parse_readings(body.get("readings", []))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    p, feats = score(readings)
    return jsonify(
        probability=round(p, 4),
        is_theft=p >= 0.5,
        risk=risk_level(p),
        days=len(readings),
        features={n: round(v, 3) for n, v in zip(FEATURE_NAMES, feats)},
        reasons=explain(feats) if p >= 0.4 else [],
    )


@app.post("/api/predict-batch")
def predict_batch():
    """CSV upload: one consumer per row, first column = consumer id, rest = daily kWh."""
    file = request.files.get("file")
    if not file:
        return jsonify(error="Upload a CSV file in the 'file' field."), 400
    try:
        text = file.read().decode("utf-8-sig")
    except UnicodeDecodeError:
        return jsonify(error="File must be UTF-8 text."), 400

    results, skipped = [], 0
    for i, row in enumerate(csv.reader(io.StringIO(text))):
        row = [c.strip() for c in row if c.strip() != ""]
        if len(row) < 2:
            continue
        try:
            readings = parse_readings(row[1:])
        except ValueError:
            if i > 0:
                skipped += 1  # first row is treated as a header
            continue
        p, _ = score(readings)
        results.append({"consumer": row[0], "probability": round(p, 4),
                        "risk": risk_level(p), "is_theft": p >= 0.5})

    if not results:
        return jsonify(error="No valid rows found. Each row needs an id followed by "
                             f"{MIN_DAYS}+ daily readings."), 400
    results.sort(key=lambda r: r["probability"], reverse=True)
    return jsonify(
        total=len(results),
        flagged=sum(r["is_theft"] for r in results),
        skipped=skipped,
        results=results,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

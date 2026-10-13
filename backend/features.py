"""Feature extraction shared by training and the API."""
import numpy as np

EPS = 1e-6

FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "cv",             # coefficient of variation (std / mean)
    "zero_ratio",     # share of days with (almost) zero consumption
    "low_ratio",      # share of days below half of the average
    "drop_ratio",     # mean of 2nd half / mean of 1st half
    "max_drop",       # biggest day-to-day fall, relative to the mean
]


def extract_features(readings):
    """Turn a list of daily kWh readings into a fixed-length feature vector."""
    x = np.asarray(readings, dtype=float)
    n = len(x)
    mean = x.mean()
    std = x.std()
    half = n // 2
    first, second = x[:half], x[half:]
    diffs = np.diff(x)

    return [
        float(mean),
        float(std),
        float(x.min()),
        float(x.max()),
        float(np.median(x)),
        float(std / (mean + EPS)),
        float(np.mean(x <= 0.05)),
        float(np.mean(x < 0.5 * mean)),
        float(second.mean() / (first.mean() + EPS)),
        float(max(0.0, -diffs.min()) / (mean + EPS)) if n > 1 else 0.0,
    ]


def explain(feats):
    """Plain-language reasons based on the feature values."""
    f = dict(zip(FEATURE_NAMES, feats))
    reasons = []
    if f["drop_ratio"] < 0.6:
        reasons.append("Consumption fell sharply in the second half of the period.")
    if f["zero_ratio"] > 0.1:
        reasons.append("Many days show zero or near-zero consumption.")
    if f["low_ratio"] > 0.3:
        reasons.append("A large share of days is far below the average usage.")
    if f["cv"] > 0.6:
        reasons.append("Daily usage is unusually irregular.")
    if f["max_drop"] > 1.5:
        reasons.append("There is an abrupt one-day fall in usage.")
    return reasons

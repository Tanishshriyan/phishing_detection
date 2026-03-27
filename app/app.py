"""Flask dashboard and APIs for phishing URL detection."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from flask import Flask, jsonify, render_template, request

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import load_model_artifact, predict_url

MODEL_PATH = ROOT_DIR / "models" / "model.pkl"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"

APP_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
    static_folder=str(APP_DIR / "static"),
)

_model_artifact: dict[str, Any] | None = None


def _ensure_model_artifact() -> dict[str, Any]:
    global _model_artifact

    if _model_artifact is None:
        _model_artifact = load_model_artifact(MODEL_PATH)
    return _model_artifact


@app.get("/")
def dashboard() -> str:
    return render_template("dashboard.html")


@app.get("/api/health")
def health_check():
    try:
        artifact = _ensure_model_artifact()
        return jsonify(
            {
                "ok": True,
                "model_loaded": artifact is not None,
                "trained_at_utc": artifact.get("trained_at_utc"),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "model_loaded": False, "error": str(exc)}), 500


@app.get("/api/status")
def model_status():
    try:
        artifact = _ensure_model_artifact()
        return jsonify(
            {
                "ok": True,
                "dataset_path": artifact.get("dataset_path"),
                "dataset_rows": artifact.get("dataset_rows"),
                "generated_dataset": artifact.get("generated_dataset"),
                "train_rows": artifact.get("train_rows"),
                "test_rows": artifact.get("test_rows"),
                "trained_at_utc": artifact.get("trained_at_utc"),
                "metrics": artifact.get("metrics", {}),
                "model_path": str(MODEL_PATH),
                "metrics_path": str(METRICS_PATH) if METRICS_PATH.exists() else None,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/predict")
def predict_endpoint():
    payload = request.get_json(silent=True) or {}
    url = (payload.get("url") or "").strip()

    if not url:
        return jsonify({"ok": False, "error": "URL is required."}), 400

    try:
        artifact = _ensure_model_artifact()
        result = predict_url(url=url, artifact=artifact)
        return jsonify({"ok": True, "result": result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    _ensure_model_artifact()
    app.run(host="127.0.0.1", port=5000, debug=False)

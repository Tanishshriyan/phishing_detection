"""Flask dashboard and APIs for phishing URL detection."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

ROOT_DIR = Path(__file__).resolve().parent
APP_DIR = ROOT_DIR / "app"
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"
MODEL_PATH = ROOT_DIR / "models" / "model.pkl"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import load_model_artifact, predict_url

_model_artifact: dict[str, Any] | None = None


def _ensure_model_artifact() -> dict[str, Any]:
    global _model_artifact

    if _model_artifact is None:
        _model_artifact = load_model_artifact(MODEL_PATH)
    return _model_artifact


def _extract_requested_url() -> str:
    payload = request.get_json(silent=True) or {}
    url = ""

    if isinstance(payload, dict):
        url = str(payload.get("url") or "").strip()

    if not url:
        url = str(request.form.get("url") or request.args.get("url") or "").strip()

    if url and "://" not in url:
        url = f"http://{url}"

    return url


def create_app() -> Flask:
    flask_app = Flask(
        __name__,
        template_folder=str(TEMPLATES_DIR),
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
    )
    CORS(flask_app)

    @flask_app.get("/")
    def dashboard() -> str:
        return render_template("dashboard.html")

    @flask_app.get("/api/health")
    def health_check():
        return jsonify({"ok": True, "status": "ok"}), 200

    @flask_app.get("/api/status")
    def model_status():
        try:
            artifact = _ensure_model_artifact()
            return jsonify(
                {
                    "ok": True,
                    "dataset_path": artifact.get("dataset_path"),
                    "dataset_rows": artifact.get("dataset_rows"),
                    "curated_rows": artifact.get("curated_rows"),
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
            return jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "model_path": str(MODEL_PATH),
                }
            ), 500

    @flask_app.post("/api/predict")
    def predict_api():
        url = _extract_requested_url()
        if not url:
            return jsonify({"ok": False, "error": "URL is required."}), 400

        try:
            artifact = _ensure_model_artifact()
            result = predict_url(url=url, artifact=artifact)
            return jsonify({"ok": True, "result": result})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @flask_app.post("/predict")
    def predict_legacy():
        url = _extract_requested_url()
        if not url:
            return jsonify({"error": "Invalid input, 'url' required"}), 400

        try:
            artifact = _ensure_model_artifact()
            result = predict_url(url=url, artifact=artifact)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return flask_app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)

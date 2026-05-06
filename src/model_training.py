"""Train and persist the phishing detection model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_preprocessing import load_raw_dataset
from src.feature_extraction import GENERIC_HOSTING_DOMAINS, extract_features_dataframe

LEGIT_DOMAINS = [
    "google.com",
    "amazon.com",
    "microsoft.com",
    "wikipedia.org",
    "openai.com",
    "github.com",
    "apple.com",
    "paypal.com",
    "bankofamerica.com",
    "chase.com",
    "netflix.com",
    "dropbox.com",
]

LEGIT_PATHS = [
    "/",
    "/home",
    "/account",
    "/products",
    "/docs",
    "/help/contact",
    "/pricing",
    "/about-us",
]

PHISH_BRANDS = [
    "paypal",
    "netflix",
    "banking",
    "appleid",
    "microsoft",
    "coinbase",
    "dropbox",
    "support",
]

PHISH_TLDS = ["xyz", "top", "click", "work", "info", "live", "cc", "biz", "shop"]
PHISH_PATHS = [
    "/login",
    "/verify-account",
    "/security-check",
    "/update-payment",
    "/confirm-password",
    "/recover/access",
]

CURATED_LEGIT_URLS = [
    "https://portfolio-demo.onrender.com",
    "https://docs-preview.vercel.app/help",
    "https://static-site.netlify.app/about",
    "https://project-notes.pages.dev/security/phishing-awareness",
    "https://team-dashboard.web.app/login",
    "https://openai.com/research",
    "https://github.com/security/advisories",
]

CURATED_PHISHING_URLS = [
    "http://phishing-detection-76wr.onrender.com",
    "http://phishing-check-login.onrender.com/verify-account",
    "https://scam-wallet-verify.vercel.app/connect",
    "http://fraud-alert-payment.netlify.app/update-payment",
    "https://malware-security-check.pages.dev/login",
    "http://paypal-login-security.example.com/verify-account",
    "https://www.apple.com.co",
]


def _project_root() -> Path:
    return ROOT_DIR


def _random_token(rng: np.random.Generator, min_len: int = 5, max_len: int = 12) -> str:
    length = int(rng.integers(min_len, max_len + 1))
    chars = np.array(list("abcdefghijklmnopqrstuvwxyz0123456789"))
    picked = rng.choice(chars, size=length, replace=True)
    return "".join(picked.tolist())


def generate_synthetic_dataset(num_samples: int = 2200, random_state: int = 42) -> pd.DataFrame:
    """Creates a fallback dataset if no raw dataset exists."""
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    phishing_count = num_samples // 2
    legit_count = num_samples - phishing_count

    for _ in range(legit_count):
        domain = rng.choice(LEGIT_DOMAINS)
        subdomain = rng.choice(["www", "app", "mail", "portal", "shop", ""])
        host = f"{subdomain}.{domain}" if subdomain else domain
        scheme = "https" if rng.random() < 0.9 else "http"
        path = rng.choice(LEGIT_PATHS)
        query = ""
        if rng.random() < 0.35:
            query = f"?ref={_random_token(rng, 4, 8)}"
        rows.append({"url": f"{scheme}://{host}{path}{query}", "label": 0})

    for _ in range(phishing_count):
        if rng.random() < 0.2:
            host = ".".join(str(int(rng.integers(10, 250))) for _ in range(4))
        else:
            brand = rng.choice(PHISH_BRANDS)
            bait = rng.choice(["secure", "verify", "update", "signin", "alert"])
            noise = _random_token(rng, 3, 8)
            tld = rng.choice(PHISH_TLDS)
            sub = rng.choice(["", "login", "auth", "pay"])
            prefix = f"{sub}." if sub else ""
            host = f"{prefix}{brand}-{bait}-{noise}.{tld}"

        scheme = "http" if rng.random() < 0.85 else "https"
        path = rng.choice(PHISH_PATHS)
        query = f"?session={_random_token(rng, 10, 16)}&token={_random_token(rng, 6, 10)}"
        url = f"{scheme}://{host}{path}{query}"
        if rng.random() < 0.25:
            url = f"{scheme}://{host}@secure-gateway-check.com{path}{query}"
        rows.append({"url": url, "label": 1})

    data = pd.DataFrame(rows)
    return data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _curated_training_examples() -> pd.DataFrame:
    rows = [{"url": url, "label": 0} for url in CURATED_LEGIT_URLS]
    rows.extend({"url": url, "label": 1} for url in CURATED_PHISHING_URLS)

    generic_hosts = sorted(GENERIC_HOSTING_DOMAINS)
    for host in generic_hosts:
        rows.append({"url": f"https://demo-app.{host}/status", "label": 0})
        rows.append({"url": f"http://phishing-update-demo.{host}/verify-account", "label": 1})

    return pd.DataFrame(rows)


def _append_curated_examples(dataset: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    curated = _curated_training_examples()
    combined = pd.concat([dataset, curated], ignore_index=True)
    combined = combined.drop_duplicates(subset=["url"], keep="last").reset_index(drop=True)
    return combined, int(len(curated))


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_builtin(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _evaluate_model(
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_test.nunique() > 1 else None,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "classification_report": classification_report(
            y_test, y_pred, target_names=["legitimate", "phishing"], output_dict=True, zero_division=0
        ),
    }
    return _to_builtin(metrics)


def train_model(
    data_path: str | Path | None = None,
    model_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    synthetic_samples: int = 2200,
) -> dict[str, Any]:
    """Trains the phishing model and saves artifact + metrics."""
    root = _project_root()
    model_output = Path(model_path) if model_path else root / "models" / "model.pkl"
    metrics_output = Path(metrics_path) if metrics_path else root / "models" / "metrics.json"
    model_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    generated_dataset = False
    try:
        dataset, resolved_path = load_raw_dataset(data_path=data_path, return_path=True)
    except (FileNotFoundError, KeyError, ValueError):
        generated_dataset = True
        dataset = generate_synthetic_dataset(num_samples=synthetic_samples, random_state=random_state)
        resolved_path = root / "data" / "raw" / "synthetic_urls.csv"
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(resolved_path, index=False)

    dataset, curated_rows = _append_curated_examples(dataset)
    x = extract_features_dataframe(dataset["url"].tolist())
    y = dataset["label"].astype(int)
    if x.empty:
        raise ValueError("Feature extraction produced empty training data.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=1,
        class_weight="balanced_subsample",
        random_state=random_state,
    )
    model.fit(x_train, y_train)

    metrics = _evaluate_model(model, x_test, y_test)
    artifact: dict[str, Any] = {
        "model": model,
        "feature_names": list(x.columns),
        "metrics": metrics,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(resolved_path),
        "dataset_rows": int(len(dataset)),
        "curated_rows": curated_rows,
        "generated_dataset": generated_dataset,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }

    joblib.dump(artifact, model_output)
    with metrics_output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_path": str(model_output),
                "trained_at_utc": artifact["trained_at_utc"],
                "dataset_path": artifact["dataset_path"],
                "dataset_rows": artifact["dataset_rows"],
                "curated_rows": artifact["curated_rows"],
                "generated_dataset": artifact["generated_dataset"],
                "metrics": artifact["metrics"],
            },
            handle,
            indent=2,
        )
    return artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train phishing URL classifier.")
    parser.add_argument("--data-path", type=str, default=None, help="Optional CSV dataset path.")
    parser.add_argument("--model-path", type=str, default=None, help="Output model artifact path.")
    parser.add_argument("--metrics-path", type=str, default=None, help="Output metrics JSON path.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=2200,
        help="Synthetic sample count used when no dataset is found.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    trained_artifact = train_model(
        data_path=args.data_path,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        test_size=args.test_size,
        random_state=args.random_state,
        synthetic_samples=args.synthetic_samples,
    )

    payload = {
        "dataset_path": trained_artifact["dataset_path"],
        "dataset_rows": trained_artifact["dataset_rows"],
        "curated_rows": trained_artifact["curated_rows"],
        "generated_dataset": trained_artifact["generated_dataset"],
        "model_metrics": trained_artifact["metrics"],
    }
    print(json.dumps(payload, indent=2))

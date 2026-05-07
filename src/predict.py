"""Prediction helpers for phishing URL classification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.feature_extraction import extract_features_dataframe, extract_url_features


def _project_root() -> Path:
    return ROOT_DIR


DEFAULT_MODEL_PATH = _project_root() / "models" / "model.pkl"
DEFAULT_DECISION_THRESHOLD = 0.45

BRAND_OFFICIAL_DOMAINS: dict[str, tuple[str, ...]] = {
    "netflix": ("netflix.com",),
    "paypal": ("paypal.com",),
    "google": ("google.com", "gmail.com", "youtube.com"),
    "microsoft": ("microsoft.com", "live.com", "office.com", "outlook.com"),
    "apple": ("apple.com", "icloud.com"),
    "amazon": ("amazon.com", "amazon.in", "amazon.co.uk"),
    "facebook": ("facebook.com", "fb.com", "messenger.com"),
    "instagram": ("instagram.com",),
    "whatsapp": ("whatsapp.com",),
    "bankofamerica": ("bankofamerica.com",),
    "chase": ("chase.com",),
}


def load_model_artifact(model_path: str | Path | None = None) -> dict[str, Any]:
    """Loads a trained model artifact from disk."""
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Model artifact not found or empty: {path}")

    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Invalid model artifact format.")
    return artifact


def _align_features(features: pd.DataFrame, feature_names: list[str] | None) -> pd.DataFrame:
    if not feature_names:
        return features
    aligned = features.copy()
    for column in feature_names:
        if column not in aligned.columns:
            aligned[column] = 0.0
    return aligned.reindex(columns=feature_names, fill_value=0.0)


def _normalize_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return "http://invalid.local"
    if "://" not in value:
        return f"http://{value}"
    return value


def _extract_hostname(url: str) -> str:
    parsed = urlparse(_normalize_url(url))
    return (parsed.hostname or "").lower()


def _extract_scheme(url: str) -> str:
    parsed = urlparse(_normalize_url(url))
    return (parsed.scheme or "").lower()


def _matches_official_domain(hostname: str, official_domains: tuple[str, ...]) -> bool:
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in official_domains)


def _official_brand_hits(hostname: str) -> list[str]:
    hits: list[str] = []
    for brand, official_domains in BRAND_OFFICIAL_DOMAINS.items():
        if _matches_official_domain(hostname, official_domains):
            hits.append(brand)
    return hits


def _brand_impersonation_hits(hostname: str, lowered_url: str) -> list[str]:
    hits: list[str] = []
    for brand, official_domains in BRAND_OFFICIAL_DOMAINS.items():
        if brand in lowered_url and not _matches_official_domain(hostname, official_domains):
            hits.append(brand)
    return hits


def _embedded_official_domain_hits(hostname: str) -> list[str]:
    """Detects hosts like 'apple.com.co' that embed a trusted domain plus extra suffix."""
    hits: list[str] = []
    for brand, official_domains in BRAND_OFFICIAL_DOMAINS.items():
        for domain in official_domains:
            if _matches_official_domain(hostname, (domain,)):
                continue
            if hostname.startswith(f"{domain}.") or f".{domain}." in hostname:
                hits.append(brand)
                break
    return hits


def _heuristic_assessment(url: str, features: dict[str, float]) -> tuple[float, list[str]]:
    lowered = _normalize_url(url).lower()
    hostname = _extract_hostname(url)
    scheme = _extract_scheme(url)
    score = 0.0
    signals: list[str] = []

    official_brand_hits = _official_brand_hits(hostname)
    brand_hits = _brand_impersonation_hits(hostname, lowered)
    embedded_domain_hits = _embedded_official_domain_hits(hostname)
    suspicious_hits = features.get("suspicious_keyword_hits", 0.0)
    host_threat_hits = features.get("host_threat_keyword_hits", 0.0)
    uses_generic_hosting = features.get("has_generic_hosting_domain", 0.0) >= 1
    brand_typo_hits = features.get("brand_typo_hits", 0.0)

    if brand_hits:
        score += 3.0
        signals.append(f"brand-impersonation:{','.join(brand_hits[:3])}")
    if embedded_domain_hits:
        score += 2.6
        signals.append(f"embedded-official-domain:{','.join(embedded_domain_hits[:3])}")
    if brand_typo_hits >= 1 and not official_brand_hits:
        score += 2.8
        signals.append("typosquatted-brand-host")
    if uses_generic_hosting and host_threat_hits >= 1:
        score += 2.6
        signals.append("generic-hosting-phishing-keywords")
        if scheme == "http":
            score += 1.2
            signals.append("http-on-hosted-app")

    if features.get("has_ip_address", 0) >= 1:
        score += 3.0
        signals.append("ip-address-host")
    if features.get("has_suspicious_tld", 0) >= 1:
        score += 2.0
        signals.append("risky-tld")
    if features.get("count_at", 0) > 0:
        score += 2.5
        signals.append("at-symbol-obfuscation")
    if features.get("has_shortener", 0) >= 1:
        score += 1.5
        signals.append("url-shortener")
        if suspicious_hits >= 1:
            score += 2.2
            signals.append("shortener-with-phishing-keywords")
        if features.get("contains_login_hint", 0) >= 1 or features.get("contains_security_hint", 0) >= 1:
            score += 1.8
            signals.append("shortener-credential-lure")
    if suspicious_hits >= 1:
        score += 1.2
        signals.append("phishing-keywords")
    if suspicious_hits >= 2:
        score += 0.8
    if features.get("uses_https", 1) < 1:
        score += 0.75
        if official_brand_hits:
            score += 3.5
            signals.append("http-on-sensitive-brand-domain")
        elif suspicious_hits >= 1:
            score += 1.6
            signals.append("http-with-phishing-keywords")
    if features.get("num_subdomains", 0) >= 3:
        score += 1.0
        signals.append("deep-subdomain-chain")
    if features.get("digit_ratio", 0) > 0.22:
        score += 0.75
    if features.get("count_hyphen", 0) >= 2:
        score += 1.2
        signals.append("excessive-hyphenation")
    if features.get("url_length", 0) >= 75:
        score += 0.6
    if any(
        token in lowered
        for token in ("verify-account", "confirm-password", "update-payment", "billing-update", "secure-update")
    ):
        score += 1.5
        signals.append("credential-harvest-pattern")

    hard_risk = (
        features.get("count_at", 0) > 0
        or bool(embedded_domain_hits)
        or (brand_hits and suspicious_hits >= 1)
        or (features.get("has_ip_address", 0) >= 1 and suspicious_hits >= 1)
        or (brand_typo_hits >= 1 and suspicious_hits >= 1)
        or (
            features.get("has_shortener", 0) >= 1
            and (suspicious_hits >= 1 or features.get("contains_login_hint", 0) >= 1 or features.get("contains_security_hint", 0) >= 1)
        )
        or (
            uses_generic_hosting
            and host_threat_hits >= 1
            and (scheme == "http" or features.get("count_hyphen", 0) >= 2 or suspicious_hits >= 2)
        )
        or (scheme == "http" and len(official_brand_hits) > 0)
    )

    probability = min(score / 8.8, 1.0)
    if hard_risk:
        probability = max(probability, 0.9)
    return probability, signals[:6]


def predict_url(
    url: str,
    model_path: str | Path | None = None,
    artifact: dict[str, Any] | None = None,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> dict[str, Any]:
    """Predicts whether a URL is phishing."""
    loaded = artifact if artifact is not None else load_model_artifact(model_path)
    model = loaded["model"]
    feature_names = loaded.get("feature_names")
    features = extract_features_dataframe([url])
    features = _align_features(features, feature_names)

    if hasattr(model, "predict_proba"):
        model_probability = float(model.predict_proba(features)[0][1])
    else:
        model_probability = float(model.predict(features)[0])

    raw_features = extract_url_features(url)
    heuristic_probability, signals = _heuristic_assessment(url, raw_features)
    phishing_probability = max(model_probability, heuristic_probability)

    threshold = max(0.01, min(float(threshold), 0.99))
    is_phishing = int(phishing_probability >= threshold)
    confidence = phishing_probability if is_phishing else 1.0 - phishing_probability

    if phishing_probability >= 0.8:
        risk_level = "high"
    elif phishing_probability >= 0.55:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "url": url,
        "prediction": "phishing" if is_phishing else "legitimate",
        "is_phishing": bool(is_phishing),
        "model_probability": round(model_probability, 6),
        "heuristic_probability": round(heuristic_probability, 6),
        "phishing_probability": round(phishing_probability, 6),
        "confidence": round(confidence, 6),
        "risk_level": risk_level,
        "signals": signals,
        "threshold": threshold,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict phishing risk from URL.")
    parser.add_argument("--url", type=str, default=None, help="URL to classify.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model artifact.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DECISION_THRESHOLD,
        help="Decision threshold (default tuned for high phishing recall).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    input_url = args.url or input("Enter URL: ").strip()
    result = predict_url(input_url, model_path=args.model_path, threshold=args.threshold)
    print(json.dumps(result, indent=2))

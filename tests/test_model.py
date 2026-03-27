import shutil
from pathlib import Path

import pandas as pd

from src.feature_extraction import extract_url_features
from src.model_training import train_model
from src.predict import load_model_artifact, predict_url


def test_extract_url_features_shape():
    features = extract_url_features("https://secure-login.example.com/account?user=alex")
    assert isinstance(features, dict)
    assert len(features) >= 25
    assert features["url_length"] > 0
    assert features["uses_https"] in (0.0, 1.0)


def test_train_and_predict_roundtrip():
    rows = [
        {"url": "https://google.com", "label": 0},
        {"url": "https://github.com/openai", "label": 0},
        {"url": "https://wikipedia.org/wiki/Security", "label": 0},
        {"url": "https://amazon.com/products?id=12", "label": 0},
        {"url": "http://paypal-verify-login.xyz/login?token=abc123", "label": 1},
        {"url": "http://secure-bank-update.top/confirm-password", "label": 1},
        {"url": "http://account-verify-alert.live/verify-account?session=abc", "label": 1},
        {"url": "http://45.12.77.122/update-payment?key=kkk", "label": 1},
    ]
    tmp_path = Path("test_artifacts")
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True, exist_ok=True)

    dataset_path = tmp_path / "mini_dataset.csv"
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"
    pd.DataFrame(rows).to_csv(dataset_path, index=False)

    artifact = train_model(
        data_path=dataset_path,
        model_path=model_path,
        metrics_path=metrics_path,
        test_size=0.25,
        random_state=12,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert "metrics" in artifact
    assert 0.0 <= artifact["metrics"]["accuracy"] <= 1.0

    loaded = load_model_artifact(model_path)
    result = predict_url("http://signin-update-account.xyz/login", artifact=loaded)
    assert "prediction" in result
    assert 0.0 <= result["phishing_probability"] <= 1.0

    netflix_spoof = predict_url("https://netflix-billing-update-secure.ml", artifact=loaded)
    assert netflix_spoof["is_phishing"] is True

    official_netflix = predict_url("https://www.netflix.com/in/", artifact=loaded)
    assert official_netflix["is_phishing"] is False

    insecure_paypal = predict_url("http://www.paypal.com", artifact=loaded)
    assert insecure_paypal["is_phishing"] is True

    embedded_brand_domain = predict_url("https://www.apple.com.co", artifact=loaded)
    assert embedded_brand_domain["is_phishing"] is True

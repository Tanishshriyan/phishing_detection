# Phishing Detection System (ML + Flask Dashboard)

A phishing URL detection project with:
- URL feature extraction
- Trained ML model artifact
- Rule-enhanced phishing recall (brand spoofing + risky URL patterns)
- Flask APIs for prediction/status/health
- Frontend dashboard for instant URL checks

## Features
- Binary URL classification (`legitimate` vs `phishing`)
- Strong unsafe URL detection using ML + high-recall heuristic signals
- Pre-trained model usage in dashboard (no in-app retraining controls)
- Metrics shown in dashboard from `models/metrics.json`

## Project Structure
```text
phishing-detection/
|- app/
|  |- static/
|  |  |- dashboard.js
|  |  `- styles.css
|  `- templates/
|     `- dashboard.html
|- data/
|  |- raw/
|  `- processed/
|- models/
|  |- model.pkl
|  `- metrics.json
|- src/
|  |- __init__.py
|  |- data_preprocessing.py
|  |- feature_extraction.py
|  |- model_training.py
|  `- predict.py
|- tests/
|  `- test_model.py
|- Procfile
|- app.py
|- requirements.txt
`- README.md
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train Model (if needed)
```bash
python src/model_training.py
```

Optional custom dataset:
```bash
python src/model_training.py --data-path data/raw/your_dataset.csv --test-size 0.25
```

## Run Dashboard
```bash
python app.py
```

Open:
- `http://127.0.0.1:5000/`

## API Endpoints
- `GET /api/health`
- `GET /api/status`
- `POST /api/predict`
- `POST /predict`

Example `POST /api/predict` body:
```json
{
  "url": "https://netflix-billing-update-secure.ml"
}
```

## Testing
```bash
python -m pytest -q
```

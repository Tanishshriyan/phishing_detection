from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

MODEL_LOADED = False
artifact = None

try:
    from src.predict import load_model_artifact, predict_url
    MODEL_PATH = ROOT_DIR / "models" / "model.pkl"
    artifact = load_model_artifact(MODEL_PATH)
    MODEL_LOADED = True
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    MODEL_LOADED = False
    artifact = None

app = Flask(__name__)
CORS(app)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "Invalid input, 'url' required"}), 400
    
    url = data['url'].strip()
    if not url:
        return jsonify({"error": "URL cannot be empty"}), 400
    
    # Basic URL validation
    if '://' not in url:
        url = 'http://' + url
    
    try:
        if MODEL_LOADED:
            result = predict_url(url, artifact=artifact)
            return jsonify({
                "url": result["url"],
                "prediction": result["prediction"],
                "confidence": result["confidence"]
            })
        else:
            # Dummy prediction
            return jsonify({
                "url": url,
                "prediction": "legitimate",
                "confidence": 0.5
            })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

@app.errorhandler(500)
def internal_error(error):
    print(f"500 error: {error}")
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
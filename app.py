from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import joblib

# Load saved models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clf = joblib.load('saved_model/logistic_regression.joblib')
tokenizer = AutoTokenizer.from_pretrained('saved_model/tokenizer')
bert_model = AutoModel.from_pretrained('saved_model/bert_model').to(device)

app = Flask(__name__)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        emb = get_embedding(text).reshape(1, -1)
        pred = clf.predict(emb)[0]
        prob = clf.predict_proba(emb)[0][pred]
        label = "SCAM" if pred == 1 else "LEGIT"

        if pred == 1:
            threat_level = round(prob * 9.5, 1)
            category = (
                "Stable" if threat_level <= 3.0 else
                "Suspicious" if threat_level <= 6.5 else
                "Critical"
            )
        else:
            threat_level = round(prob * 3.0, 1)
            category = "Stable"

        return jsonify({
            'label': label,
            'confidence': f"{prob*100:.1f}%",
            'threat_level': threat_level,
            'threat_category': category
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
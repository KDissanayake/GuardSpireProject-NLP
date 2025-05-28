import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import shutil

# Load dataset
df = pd.read_csv('synthetic_scam_dataset.csv')
texts = df['text'].tolist()
labels = df['label'].tolist()
    
# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# Load DistilBERT
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name).to(device)

# Function to get embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

# Create embeddings
print("🔨 Creating embeddings...")
embeddings = np.array([get_embedding(text) for text in texts])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.33, random_state=42, stratify=labels)

# Train classifier
print("⚙️ Training Logistic Regression...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legit', 'Scam']))

# Clean and save models
save_dir = 'saved_model'
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)

print("💾 Saving updated model...")
joblib.dump(clf, os.path.join(save_dir, 'logistic_regression.joblib'))
tokenizer.save_pretrained(os.path.join(save_dir, 'tokenizer'))
bert_model.save_pretrained(os.path.join(save_dir, 'bert_model'))

print(f"✅ Updated model saved to ./{save_dir}")

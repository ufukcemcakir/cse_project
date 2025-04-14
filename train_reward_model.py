# train_reward_model.py
import json
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def vectorize(concepts, papers):
    texts = concepts + [p["title"] + " " + p.get("abstract", "") for p in papers]
    vecs = model.encode(texts)
    return np.mean(vecs, axis=0)

def load_training_data(file_path="llm_training_data.jsonl"):
    X, y = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            x = vectorize(entry["concepts"], entry["papers"])
            X.append(x)
            y.append(entry["score"])
    return np.array(X), np.array(y)

def train():
    X, y = load_training_data()
    model = Ridge()
    model.fit(X, y)
    joblib.dump(model, "reward_model.joblib")
    print("✅ Reward model trained and saved.")

if __name__ == "__main__":
    train()

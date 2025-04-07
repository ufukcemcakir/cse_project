import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# --- Load Dataset ---
df = pd.read_csv("concept_pairs.csv")

# Optional: drop duplicates/self-pairs again if needed
df = df[df["concept_A"] != df["concept_B"]]
df = df.drop_duplicates()

# --- Load embedding model (lightweight + effective) ---
print("🔧 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Encode concept pairs into feature vectors ---
def encode_pair(concept_a, concept_b):
    emb_a = model.encode(concept_a, convert_to_tensor=False)
    emb_b = model.encode(concept_b, convert_to_tensor=False)
    return list(emb_a) + list(emb_b)

print("🔍 Encoding pairs...")
X = [encode_pair(a, b) for a, b in zip(df["concept_A"], df["concept_B"])]
y = df["label"]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Classifier ---
print("🤖 Training classifier...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Save model for later use ---
joblib.dump(clf, "prereq_classifier.joblib")
print("✅ Classifier saved to prereq_classifier.joblib")

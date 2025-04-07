import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# --- Load Data ---
df = pd.read_csv("concept_pairs.csv")
df = df[df["concept_A"] != df["concept_B"]].drop_duplicates()

# --- Load Saved Classifier and Embedding Model ---
clf = joblib.load("prereq_classifier.joblib")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Encode Concept Pairs ---
def encode_pair(a, b):
    vec_a = model.encode(a, convert_to_tensor=False)
    vec_b = model.encode(b, convert_to_tensor=False)
    return list(vec_a) + list(vec_b)

print("🔁 Encoding test pairs...")
X = [encode_pair(a, b) for a, b in zip(df["concept_A"], df["concept_B"])]
y = df["label"]

# --- Train/Test Split (Same as before) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Predict and Evaluate ---
y_pred = clf.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧩 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# --- Plot Confusion Matrix ---
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Prereq", "Prereq"], yticklabels=["No Prereq", "Prereq"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

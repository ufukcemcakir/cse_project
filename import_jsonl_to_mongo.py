import json
from pymongo import MongoClient

# Configuration
JSONL_FILE = "local_papers_with_refs.jsonl"
DB_NAME = "paper_graph"
COLLECTION_NAME = "papers"

# Connect to local MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Optional: Clear existing documents to avoid duplicates
collection.delete_many({})

# Load and insert JSONL data
with open(JSONL_FILE, "r", encoding="utf-8") as file:
    for i, line in enumerate(file, start=1):
        try:
            doc = json.loads(line)
            collection.insert_one(doc)
        except json.JSONDecodeError:
            print(f"❌ Skipping invalid JSON at line {i}")

print(f"✅ Import complete. Inserted {collection.count_documents({})} documents into '{DB_NAME}.{COLLECTION_NAME}'")

import json
import pandas as pd
from itertools import product
from sentence_transformers import SentenceTransformer, util
from concept_extractor import extract_concepts  # ← use improved extractor!

# --- CONFIG ---
INPUT_FILE = "local_papers_with_refs.jsonl"
OUTPUT_FILE = "concept_pairs_minilm.csv"
SIMILARITY_THRESHOLD = 0.6

# --- Load lightweight MiniLM model ---
print("⚡ Using all-MiniLM-L6-v2 for fast similarity...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Semantic similarity filter ---
def are_semantically_related(a, b, threshold=SIMILARITY_THRESHOLD):
    emb1 = embed_model.encode(a, convert_to_tensor=True)
    emb2 = embed_model.encode(b, convert_to_tensor=True)
    sim = util.cos_sim(emb1, emb2)
    return sim.item() >= threshold

# --- Process papers and generate pairs ---
def process_papers(filepath):
    concept_pairs = []

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            paper = json.loads(line)
            main_concepts = extract_concepts(paper.get("abstract", ""))

            if not main_concepts:
                continue

            for ref in paper.get("references", []):
                ref_concepts = extract_concepts(ref.get("abstract", ""))

                for prereq, target in product(ref_concepts, main_concepts):
                    if prereq.lower() == target.lower():
                        continue
                    if are_semantically_related(prereq, target):
                        concept_pairs.append({
                            "concept_a": prereq,
                            "concept_b": target,
                            "label": 1
                        })

            if i % 5 == 0:
                print(f"✅ Processed {i} papers... {len(concept_pairs)} pairs so far.")

    return concept_pairs

# --- Run and save ---
print("🔍 Extracting concept pairs using MiniLM...")
pairs = process_papers(INPUT_FILE)
df = pd.DataFrame(pairs)
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n🎉 Saved {len(df)} filtered concept pairs to {OUTPUT_FILE}")

import requests
import time
import spacy
import pandas as pd
from itertools import product
from sentence_transformers import SentenceTransformer, util

# Load SciSpaCy for concept extraction
nlp = spacy.load("en_core_web_sm")

# Load SciBERT model for similarity filtering
bert_model = SentenceTransformer("allenai-specter")

# Extract concepts from text
def extract_concepts(text):
    doc = nlp(text)
    return list(set(ent.text.strip() for ent in doc.ents if len(ent.text.split()) <= 4))

# Measure similarity between concepts using embeddings
def are_similar(concept1, concept2, threshold=0.6):
    emb1 = bert_model.encode(concept1, convert_to_tensor=True)
    emb2 = bert_model.encode(concept2, convert_to_tensor=True)
    sim = util.cos_sim(emb1, emb2)
    return sim.item() > threshold

# Search papers from Semantic Scholar API
def search_papers(query, limit=5):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,paperId"
    }
    response = requests.get(url, params=params)
    return response.json().get("data", [])

# Get references for a given paper (used as possible prerequisites)
def get_references(paper_id):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields=title,abstract,paperId"
    response = requests.get(url)
    time.sleep(1.2)  # To respect rate limits
    return [ref['citedPaper'] for ref in response.json().get("data", []) if ref.get("citedPaper")]

# Build concept pairs from citations and filter by semantic similarity
def build_concept_pairs(paper_limit=5):
    print("🔍 Searching for papers...")
    papers = search_papers("machine learning", limit=paper_limit)

    dataset = []

    for paper in papers:
        title = paper.get("title")
        abstract = paper.get("abstract", "")
        paper_id = paper.get("paperId")

        if not abstract:
            continue

        source_concepts = extract_concepts(abstract)
        print(f"\n📄 {title} — Extracted {len(source_concepts)} target concepts.")

        references = get_references(paper_id)

        for ref in references:
            ref_abstract = ref.get("abstract", "")
            if not ref_abstract:
                continue

            prereq_concepts = extract_concepts(ref_abstract)

            filtered_pairs = 0
            for prereq, target in product(prereq_concepts, source_concepts):
                if prereq.lower() == target.lower():
                    continue
                if are_similar(prereq, target, threshold=0.6):
                    dataset.append({
                        "concept_a": prereq,
                        "concept_b": target,
                        "label": 1
                    })
                    filtered_pairs += 1
            print(f"  🔗 {filtered_pairs} concept pairs kept from 1 reference.")

    return dataset

# Run the builder
dataset = build_concept_pairs(paper_limit=5)

# Save to CSV
df = pd.DataFrame(dataset)
df.to_csv("filtered_prerequisite_pairs.csv", index=False)
print(f"\n✅ Saved {len(df)} filtered prerequisite pairs to filtered_prerequisite_pairs.csv")

import json

# Load the stored papers
with open("Climate_papers.json", "r") as f:
    papers = json.load(f)

# Extract abstracts
abstracts = [paper.get("abstract", "") for paper in papers if paper.get("abstract")]
print(f"Loaded {len(abstracts)} abstracts.")

import spacy

# Load SciSpaCy model
nlp = spacy.load("en_core_web_sm")

def extract_terms(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

# Test on a sample abstract
sample_abstract = abstracts[0]
terms = extract_terms(sample_abstract)
# print("Extracted Knowledge Terms:", terms)

# from keybert import KeyBERT

# kw_model = KeyBERT()

# def extract_keywords(text, num_terms=5):
#     keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words="english")
#     return [kw[0] for kw in keywords[:num_terms]]

# # Test on a sample abstract
# sample_abstract = abstracts[0]
# terms = extract_keywords(sample_abstract)
# print("Extracted Key Terms:", terms)

import pandas as pd

all_terms = []
for i, abstract in enumerate(abstracts):
    terms = extract_terms(abstract)
    all_terms.append({"abstract": abstract, "terms": terms})
    if i % 50 == 0:
        print(f"Processed {i}/{len(abstracts)} abstracts...")

df = pd.DataFrame(all_terms)
df.to_csv("extracted_terms.csv", index=False)

print("Saved extracted terms to extracted_terms.csv")

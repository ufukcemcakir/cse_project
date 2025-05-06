# high_score_path_generator.py

import os
import json
import time
import random
import requests
from dotenv import load_dotenv

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OUTPUT_FILE = "llm_training_data.jsonl"
MISTRAL_REQUEST_INTERVAL = 2.0

# Concepts to base the prompts on
CONCEPT_SETS = [
    ["transformer models", "attention mechanism", "positional encoding"],
    ["graph neural networks", "message passing", "node embeddings"],
    ["reinforcement learning", "q-learning", "policy gradients"],
    ["language models", "pretraining", "causal masking"],
    ["zero-shot learning", "semantic similarity", "vector search"],
    ["generative adversarial networks", "discriminator", "latent space"],
    ["autoencoders", "dimensionality reduction", "reconstruction loss"],
    ["multi-modal learning", "vision-language models", "CLIP"],
    ["explainable ai", "shap values", "model interpretability"],
    ["self-supervised learning", "contrastive loss", "positive pairs"]
]

def mistral_generate_reading_list(concepts):
    concept_str = ", ".join(concepts)
    prompt = (
        f"You are a machine learning tutor. Generate a high-quality reading path "
        f"for a student who wants to learn about: {concept_str}. "
        f"List 5 research paper titles in a progressive learning order, focusing on clarity and conceptual depth. "
        f"Only return a list of paper titles, do not add commentary."
    )

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4
    }

    try:
        response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return parse_paper_list(content)

    except Exception as e:
        print(f"⚠️ Mistral request failed: {e}")
        return []

def parse_paper_list(text):
    papers = []
    for line in text.strip().split("\n"):
        if "-" in line:
            title = line.split("-", 1)[-1].strip()
        else:
            title = line.strip("1234567890. ").strip()
        if title:
            papers.append({
                "title": title,
                "source": "mistral-generated",
                "url": "https://example.com/fake-paper"
            })
    return papers

def append_to_file(example):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        json.dump(example, f)
        f.write("\n")

def main():
    print("🚀 Generating high-quality reading paths using Mistral...\n")
    for i in range(30):
        concept_set = random.choice(CONCEPT_SETS)
        papers = mistral_generate_reading_list(concept_set)
        if papers:
            example = {
                "concepts": concept_set,
                "papers": papers,
                "score": round(random.uniform(8.5, 9.5), 1)
            }
            append_to_file(example)
            print(f"✅ {i+1}/30 saved: {concept_set}")
        else:
            print(f"❌ {i+1}/30 failed for {concept_set}")
        time.sleep(MISTRAL_REQUEST_INTERVAL)

if __name__ == "__main__":
    main()

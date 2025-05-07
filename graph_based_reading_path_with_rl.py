# graph_based_reading_path_with_rl.py

import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from final_generator import extract_concepts_from_abstract
from graph_enhanced_generator import get_papers_for_concepts
from reinforcement_learning import PolicyNet  # Must match your model definition

# --- Config ---
TOP_K = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

# --- Load trained REINFORCE model ---
policy_model_path = "policy_net_rl.pt"
policy_net = PolicyNet(input_dim=384, hidden_dim=128).to(DEVICE)
policy_net.load_state_dict(torch.load(policy_model_path, map_location=DEVICE))
policy_net.eval()


def graph_based_reading_path_with_rl(abstract, top_k_per_concept=TOP_K):
    # --- Extract concepts from abstract ---
    concepts = extract_concepts_from_abstract(abstract)
    concepts = [c for c in concepts if len(c) > 2]
    print("📚 Extracted Concepts:", concepts)

    papers_by_concept = get_papers_for_concepts(concepts)
    final_reading_path = []

    for concept in concepts:
        candidates = papers_by_concept.get(concept, [])
        if not candidates:
            continue

        titles = [p["title"] for p in candidates]
        embeddings = EMBEDDER.encode(titles, convert_to_tensor=True, device=DEVICE)

        if len(embeddings.shape) == 1:
            embeddings = embeddings.unsqueeze(0)

        with torch.no_grad():
            scores = policy_net(embeddings)

        # Make sure scores is always iterable
        scores = scores.cpu().numpy().flatten()

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top_k = [paper for _, paper in ranked[:top_k_per_concept]]
        final_reading_path.extend(top_k)

    return final_reading_path

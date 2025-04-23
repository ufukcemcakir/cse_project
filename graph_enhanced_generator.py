# graph_enhanced_generator.py

import json
import networkx as nx
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from final_generator import extract_concepts_from_abstract

# Load papers from your .jsonl file
PAPER_FILE = "local_papers_with_refs.jsonl"
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

# --- Step 1: Build Citation Graph & Concept Mapping ---
def load_papers(filepath):
    papers = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                papers.append(json.loads(line))
            except:
                continue
    return papers

def build_citation_graph(papers):
    G = nx.DiGraph()
    for paper in papers:
        paper_id = paper.get("paperId")
        if not paper_id:
            continue
        G.add_node(paper_id, title=paper["title"], paper=paper)
        for ref in paper.get("references", []):
            ref_id = ref.get("paperId")
            if ref_id:
                G.add_edge(paper_id, ref_id)
    return G

def build_concept_to_papers(papers):
    mapping = defaultdict(list)
    for paper in papers:
        abstract = paper.get("abstract")
        if not abstract:
            continue  # skip papers with no abstract
        concepts = extract_concepts_from_abstract(abstract)
        paper["concepts"] = concepts
        for concept in concepts:
            mapping[concept].append(paper)
    return mapping

# ✅ Exposed helper for RL loop
def get_papers_for_concepts(concepts):
    papers = load_papers(PAPER_FILE)
    concept_map = build_concept_to_papers(papers)
    return {c: concept_map.get(c, []) for c in concepts}

# --- Step 2: Scoring and Recommendation ---
def rank_papers(concepts, concept_map, citation_graph, top_k=5):
    scores = []
    concept_emb = EMBEDDER.encode(concepts, convert_to_tensor=True)
    seen = set()

    for concept in concepts:
        papers = concept_map.get(concept, [])
        print(f"  - {concept}: {len(papers)} papers found")
        for paper in papers:
            pid = paper.get("paperId")
            if not pid or pid in seen:
                continue
            seen.add(pid)

            # Semantic similarity
            sim_score = util.cos_sim(EMBEDDER.encode(paper.get("abstract", ""), convert_to_tensor=True), concept_emb).mean().item()

            # Graph features
            degree = citation_graph.degree(pid)
            pagerank = nx.pagerank(citation_graph).get(pid, 0)

            # Metadata
            year = paper.get("year", 2020)
            age_score = (year - 2000) / 25  # Normalize year to 0–1

            final_score = 0.5 * sim_score + 0.3 * pagerank + 0.2 * age_score

            scores.append((final_score, paper))

    if not scores:
        print("⚠️ No papers scored — falling back to top PageRank papers")
        pr = nx.pagerank(citation_graph)
        top_ids = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for pid, score in top_ids:
            paper = citation_graph.nodes[pid].get("paper")
            if paper:
                scores.append((score, paper))

    ranked = sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
    return [p for _, p in ranked]

# --- Step 3: Main Generator ---
def graph_based_reading_path(abstract, max_results=5):
    papers = load_papers(PAPER_FILE)
    citation_graph = build_citation_graph(papers)
    concept_map = build_concept_to_papers(papers)

    concepts = extract_concepts_from_abstract(abstract)
    print("📚 Extracted Concepts:", concepts)

    ranked_papers = rank_papers(concepts, concept_map, citation_graph, top_k=max_results)
    return ranked_papers

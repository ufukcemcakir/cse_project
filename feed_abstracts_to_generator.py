import json
import datetime
import os
import time
import random
from tqdm import tqdm

from final_generator import extract_concepts_from_abstract
from graph_enhanced_generator import get_papers_for_concepts
from llm_evaluator import evaluate_reading_path_with_rubric as evaluate_reading_path
from reward_logger import save_training_example

OUTPUT_FILE = "reading_results.txt"
TRAINING_FILE = "llm_training_data.jsonl"
MISTRAL_REQUEST_INTERVAL = 2.0
NUM_VARIANTS = 5
TOP_K_PER_CONCEPT = 5
MAX_ABSTRACTS = None

def load_abstracts(file_path, limit=None):
    abstracts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            try:
                paper = json.loads(line)
                abstract = paper.get("abstract")
                title = paper.get("title", "Untitled")
                if abstract:
                    abstracts.append((title, abstract))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid line {i + 1}")
    return abstracts

def log_to_file(text):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def generate_variants(concepts, papers_by_concept, num_variants=5, top_k=5):
    candidates = []
    for _ in range(num_variants):
        path = []
        seen_titles = set()
        for c in concepts:
            if c in papers_by_concept:
                sampled = random.sample(papers_by_concept[c], min(len(papers_by_concept[c]), top_k))
                for paper in sampled:
                    if paper["title"] not in seen_titles:
                        seen_titles.add(paper["title"])
                        path.append(paper)
        candidates.append(path)
    return candidates

def rl_guided_reading_path(abstract, n_variants=NUM_VARIANTS, top_k=TOP_K_PER_CONCEPT):
    concepts = extract_concepts_from_abstract(abstract)
    concepts = [c for c in concepts if len(c) > 2]
    papers_by_concept = get_papers_for_concepts(concepts)
    candidates = generate_variants(concepts, papers_by_concept, n_variants, top_k)

    best_score = -1
    best_path = []

    for idx, path in enumerate(candidates):
        print(f"\n🔁 Evaluating path variant {idx+1}/{n_variants}...")
        score = evaluate_reading_path(path, abstract)
        time.sleep(MISTRAL_REQUEST_INTERVAL)

        if score is not None:
            print(f"✅ Mistral Score: {score}")
            if score > best_score:
                best_score = score
                best_path = path
        else:
            print("⚠️ Mistral scoring failed for this variant.")

    if best_path:
        save_training_example(concepts, best_path, best_score)
        print(f"🏆 Best score: {best_score}")
    else:
        print("⚠️ No valid reading path found.")

    return best_path

def feed_abstracts_to_generator(abstracts):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(f"\n===== 🔎 Run at {timestamp} =====")
    results = []

    with open(TRAINING_FILE, "w", encoding="utf-8") as f:
        pass

    for idx, (title, abstract) in enumerate(tqdm(abstracts, desc="Evaluating abstracts"), 1):
        header = f"\n=== Abstract {idx}: {title} ==="
        print(header)
        log_to_file(header)

        papers = rl_guided_reading_path(abstract)
        abstract_result = {"abstract_title": title, "papers": []}

        if not papers:
            msg = "No relevant papers found."
            print(msg)
            log_to_file(msg)
            results.append(abstract_result)
            continue

        any_printed = False
        for paper in papers:
            paper_title = paper.get("title")
            url = paper.get("url") or paper.get("fullTextUrl")
            source = paper.get("source", "unknown")
            if paper_title and url:
                line = f"- {paper_title} ({source})\n  {url}"
                print(line)
                log_to_file(line)
                any_printed = True
                abstract_result["papers"].append(paper)

        if not any_printed:
            msg = f"⚠️ No papers with usable links found among {len(papers)} generated."
            print(msg)
            log_to_file(msg)

        results.append(abstract_result)

    return results

def save_reading_paths_to_file(results, output_file="reading_paths_output.json"):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    file_path = "local_papers_with_refs.jsonl"
    abstracts = load_abstracts(file_path, limit=MAX_ABSTRACTS)
    results = feed_abstracts_to_generator(abstracts)
    save_reading_paths_to_file(results)

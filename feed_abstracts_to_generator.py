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

# === CONFIG ===
INPUT_FILE = "local_papers_with_refs_enriched.jsonl"
OUTPUT_FILE = "reading_results_enriched.txt"
TRAINING_FILE = "llm_training_data_enriched.jsonl"
MISTRAL_REQUEST_INTERVAL = 2.0
NUM_VARIANTS = 5
TOP_K_PER_CONCEPT = 5
MAX_ABSTRACTS = None
PAUSE_INTERVAL = 25  # Pause after every 25 abstracts
PAUSE_DURATION = 60  # seconds


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


def generate_quality_variants(concepts, papers_by_concept, top_k=5):
    high_quality = []
    low_quality = []
    seen_titles_hq = set()
    seen_titles_lq = set()

    # --- High quality: pick best top_k from each concept
    for c in concepts:
        papers = papers_by_concept.get(c, [])
        ranked = sorted(papers, key=lambda p: p.get("score", 0), reverse=True)[:top_k]
        for paper in ranked:
            if paper["title"] not in seen_titles_hq:
                high_quality.append(paper)
                seen_titles_hq.add(paper["title"])

    # --- Low quality: random unrelated or repeated papers
    all_papers = [p for papers in papers_by_concept.values() for p in papers]
    if all_papers:
        for _ in range(min(len(concepts) * top_k, 10)):
            p = random.choice(all_papers)
            if p["title"] not in seen_titles_lq:
                low_quality.append(p)
                seen_titles_lq.add(p["title"])
        # Intentionally inject some repeated or unrelated entries
        if len(all_papers) > 0:
            for _ in range(2):
                low_quality.append(random.choice(all_papers))

    return high_quality, low_quality



def rl_guided_reading_path(abstract):
    concepts = extract_concepts_from_abstract(abstract)
    concepts = [c for c in concepts if len(c) > 2]
    papers_by_concept = get_papers_for_concepts(concepts)

    high, low = generate_quality_variants(concepts, papers_by_concept)

    scored_variants = []

    for label, variant in [("high", high), ("low", low)]:
        print(f"\n🔁 Evaluating {label}-quality path...")
        score = evaluate_reading_path(variant, abstract)
        time.sleep(MISTRAL_REQUEST_INTERVAL)
        if score is not None:
            print(f"✅ Mistral Score ({label}): {score}")
            save_training_example(concepts, variant, score, quality=label)
            scored_variants.append((score, label))
        else:
            print(f"⚠️ Mistral scoring failed for {label} variant.")

    if scored_variants:
        best_label = max(scored_variants)[1]
        print(f"🏆 Best path type: {best_label}")
        return high if best_label == "high" else low
    else:
        return []


def feed_abstracts_to_generator(abstracts):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(f"\n===== 🔎 Run at {timestamp} =====")
    results = []

    with open(TRAINING_FILE, "w", encoding="utf-8") as f:
        pass  # Clear file

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

        # Pause every N abstracts
        if idx % PAUSE_INTERVAL == 0:
            print(f"⏸️ Pausing for {PAUSE_DURATION} seconds after {idx} abstracts...")
            time.sleep(PAUSE_DURATION)

    return results


def save_reading_paths_to_file(results, output_file="reading_paths_output_enriched.json"):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    abstracts = load_abstracts(INPUT_FILE, limit=MAX_ABSTRACTS)
    results = feed_abstracts_to_generator(abstracts)
    save_reading_paths_to_file(results)

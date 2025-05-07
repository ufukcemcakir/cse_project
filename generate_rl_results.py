import json
import datetime
import os
import time
from tqdm import tqdm

from final_generator import extract_concepts_from_abstract
from graph_based_reading_path_with_rl import graph_based_reading_path_with_rl

# === CONFIG ===
INPUT_FILE = "local_papers_with_abstracts_only.jsonl"
TEXT_LOG_FILE = "reading_results_enriched.txt"
STRUCTURED_LOG_FILE = "rl_results.jsonl"
MAX_ABSTRACTS = 1
PAUSE_INTERVAL = 25
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

def log_to_text(text):
    with open(TEXT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def log_to_jsonl(entry):
    with open(STRUCTURED_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def feed_abstracts_to_generator(abstracts):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_text(f"\n===== 🔎 Run at {timestamp} =====")

    for idx, (title, abstract) in enumerate(tqdm(abstracts, desc="Generating RL paths"), 1):
        header = f"\n=== Abstract {idx}: {title} ==="
        print(header)
        log_to_text(header)

        papers = graph_based_reading_path_with_rl(abstract)
        abstract_result = {"abstract_title": title, "papers": []}

        if not papers:
            msg = "No relevant papers found."
            print(msg)
            log_to_text(msg)
            log_to_jsonl(abstract_result)
            continue

        for paper in papers:
            paper_title = paper.get("title")
            url = paper.get("url") or paper.get("fullTextUrl")
            source = paper.get("source", "unknown")
            if paper_title and url:
                line = f"- {paper_title} ({source})\n  {url}"
                print(line)
                log_to_text(line)
                abstract_result["papers"].append(paper)

        if not abstract_result["papers"]:
            msg = f"⚠️ No papers with usable links found among {len(papers)} generated."
            print(msg)
            log_to_text(msg)

        log_to_jsonl(abstract_result)

        if idx % PAUSE_INTERVAL == 0:
            print(f"⏸️ Pausing for {PAUSE_DURATION} seconds after {idx} abstracts...")
            time.sleep(PAUSE_DURATION)

if __name__ == "__main__":
    abstracts = load_abstracts(INPUT_FILE, limit=MAX_ABSTRACTS)
    print(f"✅ Loaded {len(abstracts)} abstracts from {INPUT_FILE}")
    feed_abstracts_to_generator(abstracts)

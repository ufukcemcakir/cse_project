import json
import time
from tqdm import tqdm
from final_generator import extract_concepts_from_abstract
from graph_based_reading_path_with_rl import graph_based_reading_path_with_rl
from llm_evaluator import evaluate_reading_path_with_rubric as evaluate_reading_path

INPUT_FILE = "local_papers_with_abstracts_only.jsonl"
OUTPUT_FILE = "rl_vs_mistral_scores.json"
MISTRAL_REQUEST_INTERVAL = 2.0
MAX_ABSTRACTS = None  # Set a limit for testing

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
                continue
    return abstracts

def evaluate_and_log(abstracts):
    results = []

    for idx, (title, abstract) in enumerate(tqdm(abstracts, desc="Evaluating RL-generated paths"), 1):
        print(f"\n=== Abstract {idx}: {title} ===")
        concepts = extract_concepts_from_abstract(abstract)
        papers = graph_based_reading_path_with_rl(abstract)

        if not papers:
            print("⚠️ No papers generated.")
            continue

        score = evaluate_reading_path(papers, abstract)
        time.sleep(MISTRAL_REQUEST_INTERVAL)

        if score is not None:
            print(f"🧠 Mistral Score: {score}/10")
            results.append({
                "title": title,
                "concepts": concepts,
                "papers": papers,
                "mistral_score": score
            })

    return results

if __name__ == "__main__":
    abstracts = load_abstracts(INPUT_FILE, limit=MAX_ABSTRACTS)
    evaluations = evaluate_and_log(abstracts)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(evaluations, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Results saved to {OUTPUT_FILE}")

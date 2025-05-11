import json
from tqdm import tqdm
from final_generator import extract_concepts_from_abstract
from graph_enhanced_generator import get_papers_for_concepts  # your old non-RL generator

INPUT_FILE = "local_papers_with_abstracts_only.jsonl"
OUTPUT_FILE = "non_rl_results.jsonl"

MAX_ABSTRACTS = None  # Match the same limit as RL run

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

def generate_reading_paths(abstracts):
    results = []
    for title, abstract in tqdm(abstracts, desc="Generating non-RL paths"):
        concepts = extract_concepts_from_abstract(abstract)
        papers = get_papers_for_concepts(concepts)
        results.append({
            "abstract_title": title,
            "papers": papers
        })
    return results

if __name__ == "__main__":
    abstracts = load_abstracts(INPUT_FILE, limit=MAX_ABSTRACTS)
    results = generate_reading_paths(abstracts)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in results:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


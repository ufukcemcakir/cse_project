#--feed_abstracts_to_generator.py--
import json
import datetime
from graph_enhanced_generator import graph_based_reading_path
from llm_evaluator import evaluate_reading_path
from reward_logger import save_training_example
from final_generator import extract_concepts_from_abstract

OUTPUT_FILE = "reading_results.txt"
MAX_RESULTS_PER_CONCEPT = 5  # Adjustable global setting


def load_abstracts(file_path, limit=MAX_RESULTS_PER_CONCEPT):
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


def feed_abstracts_to_generator(abstracts, generator_function):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(f"\n===== 🔎 Run at {timestamp} =====")

    results = []
    for idx, (title, abstract) in enumerate(abstracts, 1):
        header = f"\n=== Abstract {idx}/{len(abstracts)}: {title} ==="
        print(header)
        log_to_file(header)

        papers = generator_function(abstract, max_results=MAX_RESULTS_PER_CONCEPT)
        # --- LLM Scoring + Logging ---
        concepts = extract_concepts_from_abstract(abstract)
        score = evaluate_reading_path(papers)

        if score is not None:
            save_training_example(concepts, papers, score)
            print(f"🧠 LLM Score: {score}/10")
            log_to_file(f"🧠 LLM Score: {score}/10")

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
    from graph_enhanced_generator import graph_based_reading_path
    from final_generator import generate_reading_path_from_abstract

    file_path = "local_papers_with_refs.jsonl"
    abstracts = load_abstracts(file_path)

    USE_GRAPH = True  # ← toggle here

    generator = graph_based_reading_path if USE_GRAPH else generate_reading_path_from_abstract
    results = feed_abstracts_to_generator(abstracts, generator)
    save_reading_paths_to_file(results)


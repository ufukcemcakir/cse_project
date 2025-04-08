# --- feed_abstracts_to_generator.py ---
import json
import datetime

OUTPUT_FILE = "reading_results.txt"
FULLTEXT_JSONL_FILE = "local_papers_with_fulltext.jsonl"
TEXTBOOK_FILE = "textbook_database.json"
MAX_RESULTS_PER_CONCEPT = 3

def load_papers(file_path, limit=MAX_RESULTS_PER_CONCEPT):
    papers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            try:
                paper = json.loads(line)
                fulltext = paper.get("fullText")
                title = paper.get("title", "Untitled")
                if fulltext:
                    papers.append((title, fulltext))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid line {i + 1}")
    return papers

def log_to_file(text):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def load_textbook_database():
    try:
        with open(TEXTBOOK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️ No textbook database found.")
        return {}

def feed_papers_to_generator(papers, generator_function, max_results=5):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(f"\n===== 🔎 Run at {timestamp} =====")
    results = []

    # Load textbooks once and pass to generator
    textbook_db = load_textbook_database()

    for idx, (title, fulltext) in enumerate(papers, 1):
        header = f"\n=== Paper {idx}/{len(papers)}: {title} ==="
        print(header)
        log_to_file(header)

        papers = generator_function(fulltext, max_results=max_results, textbook_db=textbook_db)
        if not papers:
            msg = "No relevant papers found."
            print(msg)
            log_to_file(msg)
            continue

        result_entry = {"paper_title": title, "papers": papers}
        results.append(result_entry)

        for paper in papers:
            paper_title = paper.get("title")
            url = paper.get("url")
            source = paper.get("source", "unknown")
            if paper_title and url:
                line = f"- {paper_title} ({source})\n  {url}"
                print(line)
                log_to_file(line)

    return results

# Usage
if __name__ == "__main__":
    from final_generator import generate_reading_path_from_text

    papers = load_papers(FULLTEXT_JSONL_FILE)
    results = feed_papers_to_generator(papers, generate_reading_path_from_text, max_results=MAX_RESULTS_PER_CONCEPT)

    with open("reading_paths_output_fulltext.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
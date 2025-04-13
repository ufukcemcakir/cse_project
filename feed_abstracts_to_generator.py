import json
import datetime
import os
from dotenv import load_dotenv
from final_generator import generate_reading_path_from_text, score_reading_path_with_mistral

OUTPUT_FILE = "reading_results.txt"
REWARD_LOG_FILE = "reading_reward_dataset.jsonl"
FULLTEXT_JSONL_FILE = "local_papers_with_fulltext.jsonl"
TEXTBOOK_FILE = "textbook_database.json"
MAX_RESULTS_PER_CONCEPT = 3

# Load environment variables
load_dotenv()

def load_papers(file_path, limit=None):
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
                print(f"⚠️ Skipping invalid line {i + 1}")
    return papers

def load_textbook_database():
    try:
        with open(TEXTBOOK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️ No textbook database found.")
        return []

def log_to_file(text, file=OUTPUT_FILE):
    with open(file, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def log_reward_example(input_text, paper_titles, score, file=REWARD_LOG_FILE):
    with open(file, "a", encoding="utf-8") as f:
        json.dump({
            "input": input_text[:1000],  # Avoid huge logs
            "output_titles": paper_titles,
            "reward": score
        }, f, ensure_ascii=False)
        f.write("\n")

def feed_papers_to_generator(papers, generator_function, max_results=5):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(f"\n===== 🔎 Run at {timestamp} =====")

    results = []
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

        for paper in papers:
            print(f"- {paper['title']} ({paper['source']})\n  {paper['url']}")
            log_to_file(f"- {paper['title']} ({paper['source']})\n  {paper['url']}")

        score = score_reading_path_with_mistral(title, papers)
        if score is not None:
            print(f"\n🧠 Mistral Score: {score}/10")
            log_to_file(f"\n🧠 Mistral Score: {score}/10")

            # Log training example
            paper_titles = [p["title"] for p in papers]
            log_reward_example(fulltext, paper_titles, score)

        results.append({
            "paper_title": title,
            "papers": papers,
            "mistral_score": score
        })

    return results

# Usage
if __name__ == "__main__":
    papers = load_papers(FULLTEXT_JSONL_FILE)
    results = feed_papers_to_generator(
        papers, generator_function=generate_reading_path_from_text, max_results=MAX_RESULTS_PER_CONCEPT
    )

    with open("reading_paths_output_fulltext.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

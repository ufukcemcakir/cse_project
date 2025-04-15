import json
import os

def is_duplicate_entry(new_example, existing_examples):
    new_titles = sorted([p["title"] for p in new_example["papers"]])
    new_concepts = sorted(new_example["concepts"])

    for entry in existing_examples:
        entry_titles = sorted([p["title"] for p in entry["papers"]])
        entry_concepts = sorted(entry["concepts"])
        if new_titles == entry_titles and new_concepts == entry_concepts:
            return True
    return False

def save_training_example(concepts, papers, score, output_file="llm_training_data.jsonl"):
    new_example = {
        "concepts": concepts,
        "papers": papers,
        "score": score
    }

    # Load existing examples (if file exists)
    existing = []
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if is_duplicate_entry(new_example, existing):
        print("⚠️ Duplicate reading path detected — skipping save.")
        return

    with open(output_file, "a", encoding="utf-8") as f:
        json.dump(new_example, f)
        f.write("\n")
    print("✅ New example saved.")

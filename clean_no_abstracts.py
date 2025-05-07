# clean_no_abstracts.py

import json

INPUT_FILE = "local_papers_with_refs_enriched.jsonl"
OUTPUT_FILE = "local_papers_with_abstracts_only.jsonl"

def remove_entries_without_abstract(input_path, output_path):
    kept = 0
    removed = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                entry = json.loads(line)
                if entry.get("abstract"):
                    json.dump(entry, outfile)
                    outfile.write("\n")
                    kept += 1
                else:
                    removed += 1
            except json.JSONDecodeError:
                removed += 1  # Skip malformed entries too

    print(f"✅ Done. Kept: {kept}, Removed: {removed}")

if __name__ == "__main__":
    remove_entries_without_abstract(INPUT_FILE, OUTPUT_FILE)

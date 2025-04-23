import json
import os
import requests
import time
import shutil
from tqdm import tqdm

INPUT_FILE = "local_papers_with_refs_backup.jsonl"
OUTPUT_FILE = "local_papers_with_refs.jsonl"
OPENALEX_BASE = "https://api.openalex.org"


def get_openalex_id(title):
    try:
        url = f"{OPENALEX_BASE}/works?filter=title.search:{requests.utils.quote(title)}&per-page=1"
        resp = requests.get(url)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            return results[0].get("id")
    except Exception as e:
        print(f"⚠️ OpenAlex lookup failed for title '{title}': {e}")
    return None


def get_referenced_works(openalex_id):
    try:
        work_id = openalex_id.split('/')[-1]
        url = f"{OPENALEX_BASE}/works/{work_id}"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        references = data.get("referenced_works", [])
        ref_entries = []
        for ref_id in references:
            time.sleep(0.3)  # rate limit protection
            ref_url = f"{OPENALEX_BASE}/works/{ref_id.split('/')[-1]}"
            ref_resp = requests.get(ref_url)
            if ref_resp.status_code == 200:
                ref_data = ref_resp.json()
                ref_entries.append({
                    "title": ref_data.get("title"),
                    "year": ref_data.get("publication_year"),
                    "url": ref_data.get("id"),
                    "doi": ref_data.get("doi"),
                    "authors": [a["author"]["display_name"] for a in ref_data.get("authorships", []) if "author" in a]
                })
        return ref_entries
    except Exception as e:
        print(f"⚠️ Could not fetch references for {openalex_id}: {e}")
        return []


def enrich_and_merge_arxiv_papers(input_path, output_path):
    if not os.path.exists(input_path):
        print("❌ Input file not found.")
        return

    enriched_lines = []
    seen_titles = set()

    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    for line in tqdm(lines, desc="🔄 Enriching Papers", unit="paper"):
        try:
            paper = json.loads(line)
            title_key = paper.get("title", "").strip().lower()
            if title_key in seen_titles:
                print(f"⚠️ Duplicate found, skipping: {paper['title']}")
                continue
            seen_titles.add(title_key)

            if paper.get("source") == "arxiv":
                if "references" in paper and isinstance(paper["references"], list) and len(paper["references"]) > 0:
                    print(f"⏭️ Skipping already enriched: {paper['title']}")
                else:
                    print(f"🔎 Enriching: {paper['title']}")
                    openalex_id = get_openalex_id(paper["title"])
                    if openalex_id:
                        paper["openalex_id"] = openalex_id
                        paper["references"] = get_referenced_works(openalex_id)
                        print(f"✅ Added {len(paper['references'])} references")
                    else:
                        print("⚠️ No OpenAlex match found.")

            enriched_lines.append(json.dumps(paper))
        except Exception as e:
            print(f"⚠️ Failed to process a paper: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in enriched_lines:
            f.write(line + '\n')

    print(f"✅ Enrichment complete. Output written to {output_path}")


if __name__ == "__main__":
    enrich_and_merge_arxiv_papers(INPUT_FILE, OUTPUT_FILE)

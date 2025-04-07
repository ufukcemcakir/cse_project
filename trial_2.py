import requests
import json
import time
import os

QUERY = "artificial intelligence"
LIMIT = 10
PAPER_FIELDS = "title,abstract,year,paperId,fieldsOfStudy,openAccessPdf"
REFERENCE_FIELDS = "title,abstract,paperId,openAccessPdf"
OUTPUT_FILE = "local_papers_with_refs.jsonl"
OFFSET_FILE = "offset_tracker_refs.txt"

# Load/save offset for pagination
def load_offset():
    return int(open(OFFSET_FILE).read()) if os.path.exists(OFFSET_FILE) else 0

def save_offset(offset):
    with open(OFFSET_FILE, "w") as f:
        f.write(str(offset))

# Search for papers
def fetch_papers(query, limit, offset):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "offset": offset,
        "fields": PAPER_FIELDS
    }
    print(f"🔍 Fetching {limit} papers from offset {offset}...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get("data", [])

# Fetch references for a given paper
def fetch_references(paper_id):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields={REFERENCE_FIELDS}"
    try:
        response = requests.get(url)
        time.sleep(1)  # be nice to the API
        response.raise_for_status()
        references = []
        for ref in response.json().get("data", []):
            cited = ref.get("citedPaper")
            if cited:
                pdf_url = cited.get("openAccessPdf", {}).get("url")
                if pdf_url:
                    cited["fullTextUrl"] = pdf_url
                references.append(cited)
        return references
    except Exception as e:
        print(f"⚠️ Failed to get references for {paper_id}: {e}")
        return []

# Append data to a JSONL file
def append_to_file(papers, filepath):
    with open(filepath, "a", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper) + "\n")

# Main script
def main():
    offset = load_offset()
    papers = fetch_papers(QUERY, LIMIT, offset)

    if not papers:
        print("✅ No new papers found.")
        return

    enriched = []
    for paper in papers:
        paper_id = paper.get("paperId")
        if not paper_id:
            continue

        references = fetch_references(paper_id)
        paper["references"] = references

        pdf_url = paper.get("openAccessPdf", {}).get("url")
        if pdf_url:
            paper["fullTextUrl"] = pdf_url

        enriched.append(paper)
        print(f"📄 {paper.get('title', 'Unknown')} — {len(references)} refs")

    append_to_file(enriched, OUTPUT_FILE)
    print(f"✅ Saved {len(enriched)} enriched papers to {OUTPUT_FILE}")

    offset += LIMIT
    save_offset(offset)
    time.sleep(2)

if __name__ == "__main__":
    main()

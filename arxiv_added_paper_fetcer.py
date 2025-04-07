import requests
import json
import time
import os
import feedparser
import urllib.parse

# --- Config ---
QUERY = "artificial intelligence"
LIMIT = 10
OUTPUT_FILE = "local_papers_with_refs.jsonl"
OFFSET_FILE = "offset_tracker_refs.txt"

# Fields for Semantic Scholar
PAPER_FIELDS = "title,abstract,year,paperId,fieldsOfStudy,openAccessPdf"
REFERENCE_FIELDS = "title,abstract,paperId,openAccessPdf"

# --- Offset Handling ---
def load_offset():
    return int(open(OFFSET_FILE).read()) if os.path.exists(OFFSET_FILE) else 0

def save_offset(offset):
    with open(OFFSET_FILE, "w") as f:
        f.write(str(offset))

# --- Semantic Scholar ---
def fetch_semantic_scholar_papers(query, limit, offset):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "offset": offset,
        "fields": PAPER_FIELDS
    }
    print(f"🔍 [SS] Fetching {limit} papers from offset {offset}")
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get("data", [])

def fetch_references(paper_id):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields={REFERENCE_FIELDS}"
    try:
        response = requests.get(url)
        time.sleep(1)
        response.raise_for_status()
        refs = []
        for ref in response.json().get("data", []):
            cited = ref.get("citedPaper") or {}
            if cited.get("openAccessPdf", {}).get("url"):
                cited["fullTextUrl"] = cited["openAccessPdf"]["url"]
                refs.append(cited)
        return refs
    except Exception as e:
        print(f"⚠️ Failed to get refs for {paper_id}: {e}")
        return []

# --- arXiv ---
def fetch_arxiv_papers(query, max_results=10):
    base_url = "http://export.arxiv.org/api/query"
    search_query = f"all:{query}"
    encoded_query = urllib.parse.quote(f"all:{query}")
    url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_results}"
    print(f"🔍 [arXiv] Querying: {search_query}")
    try:
        feed = feedparser.parse(url)
        papers = []
        for entry in feed.entries:
            title = entry.title.strip()
            abstract = entry.summary.strip()
            link = entry.link
            if title and abstract and link:
                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "fullTextUrl": link,
                    "source": "arxiv"
                })
        return papers
    except Exception as e:
        print(f"⚠️ arXiv error: {e}")
        return []

# --- Save to File ---
def append_to_file(papers, filepath):
    with open(filepath, "a", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper) + "\n")

# --- Main Runner ---
def main():
    offset = load_offset()
    final_papers = []

    # --- Semantic Scholar ---
    ss_papers = fetch_semantic_scholar_papers(QUERY, LIMIT, offset)
    for paper in ss_papers:
        pdf_url = paper.get("openAccessPdf", {}).get("url")
        if not pdf_url:
            continue

        paper["fullTextUrl"] = pdf_url
        paper["source"] = "semantic_scholar"
        paper_id = paper.get("paperId")
        paper["references"] = fetch_references(paper_id) if paper_id else []
        final_papers.append(paper)
        print(f"✅ [SS] {paper['title']}")

    # --- arXiv ---
    arxiv_papers = fetch_arxiv_papers(QUERY, max_results=LIMIT)
    for paper in arxiv_papers:
        final_papers.append(paper)
        print(f"✅ [arXiv] {paper['title']}")

    if final_papers:
        append_to_file(final_papers, OUTPUT_FILE)
        print(f"💾 Saved {len(final_papers)} papers to {OUTPUT_FILE}")
    else:
        print("⚠️ No full-text papers found.")

    save_offset(offset + LIMIT)
    time.sleep(2)

if __name__ == "__main__":
    main()

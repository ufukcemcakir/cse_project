import json
import requests
import time

INPUT_FILE = "local_papers_with_refs.jsonl"
OUTPUT_FILE = "local_papers_with_fulltext.jsonl"


def download_full_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"⚠️ Failed to download full text from {url}: {e}")
        return None


def enrich_with_fulltext(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, start=1):
            try:
                paper = json.loads(line)
                url = paper.get("fullTextUrl") or paper.get("openAccessPdf", {}).get("url")

                if not url:
                    print(f"⏭️ Skipping paper {i} — no fullTextUrl")
                    continue

                full_text = download_full_text(url)
                if not full_text or len(full_text.strip()) < 500:
                    print(f"⏭️ Skipping paper {i} — content too short or empty")
                    continue

                paper["fullText"] = full_text
                fout.write(json.dumps(paper) + "\n")
                print(f"✅ Saved full text for paper {i}: {paper.get('title', 'Untitled')}")

                time.sleep(1)
            except json.JSONDecodeError:
                print(f"❌ Skipping malformed JSON at line {i}")


if __name__ == "__main__":
    enrich_with_fulltext(INPUT_FILE, OUTPUT_FILE)

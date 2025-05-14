import requests
import os
import time
from dotenv import load_dotenv

# === Load Mistral API key ===
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# === Configurable parameters ===
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 2  # seconds
DEFAULT_DELAY_BETWEEN_CALLS = 5  # seconds

def evaluate_reading_path_with_rubric(papers, abstract=None, delay=DEFAULT_DELAY_BETWEEN_CALLS):
    if MISTRAL_API_KEY is None:
        print("❌ MISTRAL_API_KEY is not set from .env")
        return None

    if not papers or not isinstance(papers, list):
        print("⚠️ No papers provided or papers are not a list.")
        return None

    valid_items = []
    for i, paper in enumerate(papers):
        if isinstance(paper, dict) and "title" in paper and isinstance(paper["title"], str):
            valid_items.append(f"- {paper['title']}")
        else:
            print(f"⚠️ Skipping invalid paper at index {i}: {paper}")
    
    if not valid_items:
        print("⚠️ No valid paper titles found to evaluate.")
        return None

    items = "\n".join(valid_items)

    prompt = f"""
You are an academic assistant helping evaluate reading paths for learning.

Rate the following list from 0 to 10 based on how helpful it is for understanding the topic described in the abstract.

Use this scoring rubric:
- 9–10: Excellent. Strong progression, clear build-up of concepts.
- 7–8: Good. Mostly logical, but a few unclear links.
- 5–6: Average. Some relevance, limited structure.
- 3–4: Weak. Disconnected or lacks clarity.
- 0–2: Poor. Unrelated or confusing.

Abstract (for context):
{abstract if abstract else '[no abstract provided]'}

Reading path:
{items}

Only respond with a number between 0 and 10.
""".strip()

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            print("📬 Raw response from Mistral:", result)

            score_str = result["choices"][0]["message"]["content"].strip()
            score = float(score_str)
            time.sleep(delay)  # Delay after successful request
            return max(0, min(10, score))

        except requests.exceptions.HTTPError as http_err:
            print(f"❌ HTTP error on attempt {attempt+1}: {http_err}")
            print("🔍 Full response:", getattr(response, 'text', '[no response text]'))

        except ValueError as parse_err:
            print(f"❌ Could not parse score on attempt {attempt+1}: {parse_err}")
            print("🔍 Returned content:", result['choices'][0]['message']['content'])

        except Exception as e:
            print(f"⚠️ Unexpected error in LLM scoring on attempt {attempt+1}: {e}")

        wait = RETRY_BACKOFF_BASE * (2 ** attempt)
        print(f"⏳ Retrying after {wait} seconds...")
        time.sleep(wait)

    print("❌ All retries failed. Returning None.")
    return None

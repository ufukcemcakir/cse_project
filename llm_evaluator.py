# llm_evaluator.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def evaluate_reading_path_with_rubric(papers, abstract=None):
    if MISTRAL_API_KEY is None:
        print("❌ MISTRAL_API_KEY is not set from .env")
        return None

    if not papers:
        print("⚠️ No papers provided to LLM evaluator.")
        return None

    items = "\n".join(f"- {paper['title']}" for paper in papers)

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

    try:
        response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        print("📬 Raw response from Mistral:", result)

        score_str = result["choices"][0]["message"]["content"].strip()
        score = float(score_str)
        return max(0, min(10, score))

    except requests.exceptions.HTTPError as http_err:
        print("❌ HTTP error:", http_err)
        print("🔍 Full response:", response.text)
        return None
    except ValueError as parse_err:
        print("❌ Could not parse score:", parse_err)
        print("🔍 Returned content:", result['choices'][0]['message']['content'])
        return None
    except Exception as e:
        print("⚠️ Unexpected error in LLM scoring:", e)
        return None

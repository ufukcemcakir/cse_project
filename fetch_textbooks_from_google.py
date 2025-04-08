import requests
import json
import time

TOPIC_QUERIES = [
    "artificial intelligence textbook",
    "machine learning textbook",
    "natural language processing textbook",
    "computer vision textbook",
    "deep learning textbook",
    "computational theory textbook",
    "data mining textbook"
]

OUTPUT_FILE = "textbook_database.json"
GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"

def fetch_books_for_topic(topic, max_results=5):
    params = {
        "q": topic,
        "maxResults": max_results,
        "printType": "books"
    }
    try:
        response = requests.get(GOOGLE_BOOKS_URL, params=params)
        response.raise_for_status()
        data = response.json()
        books = []
        for item in data.get("items", []):
            info = item.get("volumeInfo", {})
            title = info.get("title")
            url = info.get("infoLink")
            authors = info.get("authors", [])
            if title and url:
                books.append({
                    "title": title,
                    "authors": authors,
                    "url": url,
                    "source": "textbook"
                })
        return books
    except Exception as e:
        print(f"⚠️ Failed to fetch books for topic '{topic}': {e}")
        return []

def main():
    all_books = []
    for topic in TOPIC_QUERIES:
        print(f"🔍 Searching for: {topic}")
        books = fetch_books_for_topic(topic, max_results=5)
        all_books.extend(books)
        time.sleep(1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_books, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved {len(all_books)} textbooks to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

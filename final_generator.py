import spacy
import joblib
import requests
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import time
import feedparser
import urllib.parse

# Load models
clf = joblib.load("prereq_classifier.joblib")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

# Config
MAX_RESULTS_PER_CONCEPT = 10

GENERIC_CONCEPTS = {"early", "key", "based", "systems", "tasks", "methods", "models", "approach", "approaches", "research"}

def is_valid_concept(concept):
    return len(concept) > 3 and concept.lower() not in GENERIC_CONCEPTS

def extract_concepts_from_abstract(abstract, top_n=7):
    doc = nlp(abstract)
    chunks = [chunk.text.lower().strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform([' '.join(chunks)])
    scores = tfidf_matrix.toarray().flatten()
    feature_names = tfidf.get_feature_names_out()
    scored = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return [phrase for phrase, score in scored[:top_n] if score > 0]

def encode_pair(a, b):
    vec_a = embed_model.encode(a)
    vec_b = embed_model.encode(b)
    return list(vec_a) + list(vec_b)

search_cache = {}

def search_semantic_scholar(concept, max_results=MAX_RESULTS_PER_CONCEPT):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"User-Agent": "ConceptGraphFetcher/1.0"}
    queries = [f"introduction to {concept}", f"{concept} review", f"{concept} tutorial", concept]

    for query in queries:
        params = {"query": query, "limit": max_results, "fields": "title,url,authors,year,abstract"}
        try:
            response = requests.get(url, params=params, headers=headers)
            time.sleep(1)
            response.raise_for_status()
            papers = response.json().get("data", [])
            results = []
            for paper in papers:
                title = paper.get("title")
                url = paper.get("url")
                abstract = paper.get("abstract")
                if title and url and abstract:
                    results.append({
                        "title": title,
                        "url": url,
                        "abstract": abstract,
                        "source": "semantic_scholar"
                    })
            if results:
                return results
        except Exception:
            continue
    return []

def search_arxiv(concept, max_results=MAX_RESULTS_PER_CONCEPT):
    encoded = urllib.parse.quote(f"all:{concept}")
    url = f"http://export.arxiv.org/api/query?search_query={encoded}&start=0&max_results={max_results}"
    try:
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries:
            title = entry.title
            url = entry.link
            abstract = entry.summary
            if title and url and abstract:
                results.append({
                    "title": title,
                    "url": url,
                    "abstract": abstract,
                    "source": "arxiv"
                })
        return results
    except Exception:
        return []

def search_papers_for_concept(concept, max_results=MAX_RESULTS_PER_CONCEPT):
    if concept in search_cache:
        return search_cache[concept]

    results = search_semantic_scholar(concept, max_results)
    if not results:
        results = search_arxiv(concept, max_results)

    # Filter results using cosine similarity
    filtered = []
    concept_emb = embed_model.encode(concept, convert_to_tensor=True)
    for paper in results:
        score = util.cos_sim(concept_emb, embed_model.encode(paper["abstract"], convert_to_tensor=True)).item()
        if score > 0.3:
            paper["score"] = round(score, 3)
            paper["matched_concept"] = concept
            filtered.append(paper)

    # De-duplicate by title
    seen_titles = set()
    deduped = []
    for paper in filtered:
        if paper["title"] not in seen_titles:
            seen_titles.add(paper["title"])
            deduped.append(paper)

    search_cache[concept] = deduped
    return deduped

def generate_reading_path_from_abstract(abstract, max_results=MAX_RESULTS_PER_CONCEPT):
    concepts = extract_concepts_from_abstract(abstract, top_n=7)
    concepts = [c for c in concepts if is_valid_concept(c)]

    if len(concepts) <= 1:
        return [paper for c in concepts for paper in search_papers_for_concept(c, max_results=max_results)]

    G = nx.DiGraph()
    for concept in concepts:
        G.add_node(concept)

    for i in range(len(concepts)):
        for j in range(len(concepts)):
            if i == j:
                continue
            pair = encode_pair(concepts[i], concepts[j])
            pred = clf.predict([pair])[0]
            if pred == 1:
                G.add_edge(concepts[i], concepts[j])

    try:
        sorted_concepts = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        sorted_concepts = concepts

    reading_plan = []
    seen = set()
    for concept in sorted_concepts:
        if not is_valid_concept(concept):
            continue
        for paper in search_papers_for_concept(concept, max_results=max_results):
            key = paper["title"]
            if key not in seen:
                seen.add(key)
                reading_plan.append(paper)

    return reading_plan

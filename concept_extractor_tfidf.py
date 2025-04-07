import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Extract noun chunks from a single document
def extract_noun_chunks(text):
    doc = nlp(text)
    return [chunk.text.lower().strip() for chunk in doc.noun_chunks if 1 < len(chunk.text.strip()) < 50]

# Fit TF-IDF model over a list of abstracts
def build_tfidf_model(documents):
    all_phrases = [' '.join(extract_noun_chunks(doc)) for doc in documents]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3), max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(all_phrases)
    return vectorizer, tfidf_matrix

# Extract top phrases from a new document using fitted TF-IDF
def extract_top_concepts(text, vectorizer, top_n=5):
    phrases = [' '.join(extract_noun_chunks(text))]
    tfidf_vector = vectorizer.transform(phrases)
    scores = tfidf_vector.toarray().flatten()
    feature_names = vectorizer.get_feature_names_out()
    phrase_scores = list(zip(feature_names, scores))
    sorted_phrases = sorted(phrase_scores, key=lambda x: x[1], reverse=True)
    return [phrase for phrase, score in sorted_phrases[:top_n] if score > 0]
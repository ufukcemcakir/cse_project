import spacy

# Load the SciSpaCy large model
print("🔧 Loading en_core_web_lg...")
nlp = spacy.load("en_core_web_trf")

# OPTIONAL: Enable TextRank
USE_TEXTRANK = False
if USE_TEXTRANK:
    import pytextrank
    nlp.add_pipe("textrank")

# Main extractor function
def extract_concepts(text, top_n=5, min_len=4, max_words=4):
    if not text or len(text) < 30:
        return []

    concepts = set()

    # --- Option 1: Use SciSpaCy entities ---
    doc = nlp(text)
    for ent in doc.ents:
        phrase = ent.text.strip()
        if len(phrase) >= min_len and len(phrase.split()) <= max_words:
            concepts.add(phrase)

    # --- Option 2: Add top-ranked phrases from TextRank ---
    if USE_TEXTRANK:
        for phrase in doc._.phrases[:top_n]:
            if len(phrase.text.split()) <= max_words:
                concepts.add(phrase.text.strip())

    return list(concepts)

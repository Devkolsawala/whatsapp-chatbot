import json
import re
import os
from collections import defaultdict
from math import log

SOURCE_FILE = 'whatsapp_faq_multilingual.json'
INDEX_FILE = 'faq_index.json'

def normalize(text):
    if not isinstance(text, str): return []
    text = text.lower()
    text = re.sub(r"[^\w\s\u0900-\u097F]", "", text)
    return text.split()

# Load the original multilingual questions
with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# Build document list
documents = []
word_doc_freq = defaultdict(int)  # for IDF

for entry in raw_data:
    doc_id = entry['id']
    all_keywords = set()

    # Collect paraphrases + base questions
    for lang, question in entry.get('question', {}).items():
        all_keywords.update(normalize(question))

    for lang, paraphrases in entry.get('paraphrases', {}).items():
        for phr in paraphrases:
            all_keywords.update(normalize(phr))

    doc_keywords = list(all_keywords)
    for word in doc_keywords:
        word_doc_freq[word] += 1

    documents.append({
        "id": doc_id,
        "keywords": doc_keywords,
        "answers": entry.get('answer', {})
    })

# Compute IDF scores
total_docs = len(documents)
idf_scores = {}
for word, doc_count in word_doc_freq.items():
    idf_scores[word] = log((total_docs + 1) / (doc_count + 1)) + 1

# Save the index
final_index = {
    "documents": documents,
    "idf_scores": idf_scores
}

with open(INDEX_FILE, 'w', encoding='utf-8') as f:
    json.dump(final_index, f, ensure_ascii=False, indent=2)

print(f"Index built successfully and saved to '{INDEX_FILE}'")

import json
import re
import logging

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
LANGUAGES = ["en", "hi", "id"]
FAQ_JSON_FILE = 'whatsapp_faq_multilingual.json'
OUTPUT_INDEX_FILE = 'faq_index.json'

# Define common "stop words" that should be ignored during search
STOP_WORDS = {
    'en': {'a', 'an', 'the', 'is', 'in', 'it', 'i', 'to', 'for', 'of', 'how', 'do', 'can', 'what', 'why', 'my'},
    'hi': {'एक', 'में', 'है', 'की', 'मैं', 'यह', 'से', 'क्या', 'कैसे', 'क्यों', 'मेरा', 'मेरे'},
    'id': {'di', 'ke', 'dari', 'dan', 'ini', 'itu', 'saya', 'untuk', 'bagaimana', 'apa', 'kenapa', 'bisa'}
}

def normalize_and_tokenize(text, lang):
    """Normalizes text and splits it into important keywords."""
    if not isinstance(text, str): return []
    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r"[^\w\s\u0900-\u097F]", "", text)
    # Tokenize (split into words)
    words = text.split()
    # Remove stop words
    important_words = [word for word in words if word not in STOP_WORDS.get(lang, set())]
    return important_words

if __name__ == '__main__':
    logger.info("Starting search index creation...")
    
    try:
        with open(FAQ_JSON_FILE, 'r', encoding='utf-8') as f:
            faqs = json.load(f)
    except FileNotFoundError:
        logger.error(f"FATAL: {FAQ_JSON_FILE} not found.")
        exit()

    # The final index will hold the processed questions and answers
    search_data = {lang: [] for lang in LANGUAGES}

    for faq_item in faqs:
        for lang in LANGUAGES:
            original_question = faq_item['question'][lang]
            answer = faq_item['answer'][lang]
            
            # Combine original question and paraphrases for indexing
            all_phrases = [original_question] + faq_item.get('paraphrases', {}).get(lang, [])
            
            # Create a unique set of keywords for this entire FAQ entry
            keywords = set()
            for phrase in all_phrases:
                keywords.update(normalize_and_tokenize(phrase, lang))

            search_data[lang].append({
                "question": original_question,
                "answer": answer,
                "keywords": list(keywords) # Store as a list in the JSON
            })
            
    # Save the processed data to a new index file
    with open(OUTPUT_INDEX_FILE, 'w', encoding='utf-8') as f_out:
        json.dump(search_data, f_out, ensure_ascii=False, indent=2)

    logger.info(f"Successfully created search index file: '{OUTPUT_INDEX_FILE}'")
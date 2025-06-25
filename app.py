import json
from langdetect import detect, DetectorFactory
import re
from flask import Flask, request, jsonify, render_template_string
import logging
from rapidfuzz import process, fuzz

app = Flask(__name__)

# --- CONFIGURATION ---
# Set up logging to monitor the app's behavior and troubleshoot issues.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure consistent language detection results.
DetectorFactory.seed = 0

# The confidence score required for a match to be considered valid.
SIMILARITY_THRESHOLD = 70
LANGUAGES = ["en", "hi", "id"]

# --- TEXT NORMALIZATION ---
def normalize(text):
    """
    Cleans and standardizes text for more accurate matching.
    - Converts to lowercase.
    - Removes punctuation (except for specific language characters).
    - Corrects common, hardcoded typos.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Preserve Devanagari (Hindi) and standard alphanumeric characters.
    text = re.sub(r"[^\w\s\u0900-\u097F]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Manual typo corrections for better matching
    manual_typos = {
        "helo": "hello", "watsap": "whatsapp", "whastapp": "whatsapp", "downlaod": "download",
        "हेलो": "नमस्ते", "वाट्सप": "व्हाट्सएप", "डाउनलोड्ड": "डाउनलोड",
        "halo": "hai", "undh": "unduh", "wa": "whatsapp",
        "exir": "exit", "exittt": "exit", "quittt": "quit", "byee": "bye"  # Added exit-related typos
    }
    words = text.split()
    corrected_words = [manual_typos.get(word, word) for word in words]
    return " ".join(corrected_words)

# --- GREETING AND EXIT HANDLING ---
GREETINGS = {
    "en": {"hi", "hello", "hey", "good morning", "good evening"},
    "hi": {"नमस्ते", "हाय", "हेलो", "शुभ प्रभात", "शुभ संध्या"},
    "id": {"hai", "halo", "selamat pagi", "selamat malam"}
}
EXIT_COMMANDS = {"exit", "quit", "bye"}  # Base exit commands
GREETING_RESPONSES = {
    "en": "Hello! How can I assist you with WhatsApp statuses?",
    "hi": "नमस्ते! मैं व्हाट्सएप स्टेटस के बारे में कैसे मदद कर सकता हूँ?",
    "id": "Hai! Bagaimana saya bisa membantu dengan status WhatsApp?"
}

def detect_language_and_check_intent(text):
    """
    Detects the language of the input text and checks if it's a greeting or exit intent.
    """
    normalized_text = normalize(text)
    normalized_words = set(normalized_text.split())
    
    # Check for exit intent with fuzzy matching
    for lang in LANGUAGES:
        if any(word in EXIT_COMMANDS for word in normalized_words):
            best_exit_match = process.extractOne(normalized_text, EXIT_COMMANDS, scorer=fuzz.partial_ratio, score_cutoff=60)
            if best_exit_match:
                return lang, False, True  # Language, not a greeting, is an exit
    
    # Check for greetings
    for lang, greetings in GREETINGS.items():
        if normalized_words.intersection(greetings):
            return lang, True, False

    # If not a greeting or exit, perform language detection
    try:
        lang = detect(text)
        return lang if lang in LANGUAGES else "en", False, False
    except:
        return "en", False, False  # Default to English if detection fails

# --- FAQ DATA LOADING AND PREPARATION ---
faq_answers = {lang: {} for lang in LANGUAGES}
search_corpus = {lang: [] for lang in LANGUAGES}

def load_faq_data():
    """
    Loads FAQs from the JSON file and prepares them for searching.
    It builds a search corpus including main questions and their paraphrases.
    """
    try:
        with open('whatsapp_faq_multilingual.json', 'r', encoding='utf-8') as f:
            faqs = json.load(f)
        
        for item in faqs:
            for lang in LANGUAGES:
                original_question = item["question"][lang]
                answer = item["answer"][lang]
                
                # Store the answer keyed by the original question.
                faq_answers[lang][original_question] = answer
                
                # Add the original question to the search corpus.
                search_corpus[lang].append((normalize(original_question), original_question))
                
                # Add all paraphrases to the search corpus.
                paraphrases = item.get("paraphrases", {}).get(lang, [])
                if isinstance(paraphrases, list):
                    for p in paraphrases:
                        search_corpus[lang].append((normalize(p), original_question))

        logger.info("FAQ data loaded and prepared successfully.")

    except FileNotFoundError:
        logger.error("FATAL: whatsapp_faq_multilingual.json not found. The chatbot cannot function without it.")
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"FATAL: Error processing JSON file: {e}")

# --- CORE MATCHING LOGIC ---
def find_best_match(question, lang):
    """
    Finds the best matching question from the corpus using fuzzy string matching.
    Uses multiple scorers for better accuracy.
    """
    normalized_question = normalize(question)
    
    if not search_corpus[lang]:
        return None, 0.0

    # Extract just the phrases to search against
    phrases = [item[0] for item in search_corpus[lang]]
    
    # Use process.extractOne with a combination of scorers
    best_match_result = process.extractOne(
        normalized_question,
        phrases,
        scorer=lambda s1, s2: max(fuzz.partial_ratio(s1, s2), fuzz.token_sort_ratio(s1, s2)),
        score_cutoff=SIMILARITY_THRESHOLD
    )

    if best_match_result:
        best_match_phrase, similarity, index = best_match_result
        # Get the original question associated with the matched phrase
        original_question = search_corpus[lang][index][1]
        logger.info(f"Matched '{question}' to '{original_question}' with similarity {similarity:.2f}")
        return original_question, similarity
    logger.warning(f"No good match found for '{question}' in language '{lang}'.")
    return None, 0.0

# --- FLASK ROUTES ---
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>WhatsApp Status FAQ Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .chat-container { max-width: 600px; margin: auto; }
        .message { margin: 10px; padding: 10px; border-radius: 5px; }
        .user { background: #e0f7fa; text-align: right; }
        .bot { background: #f1f8e9; text-align: left; }
        .input-box { margin-top: 20px; }
        input[type="text"] { width: 80%; padding: 10px; }
        button { padding: 10px; }
    </style>
    <script>
        function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            if (!message) return;
            const chat = document.getElementById('chat');
            chat.innerHTML += `<div class="message user">${message}</div>`;
            fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message})
            })
            .then(response => response.json())
            .then(data => {
                chat.innerHTML += `<div class="message bot">${data.response}</div>`;
                chat.scrollTop = chat.scrollHeight;
            });
            input.value = '';
        }
        function checkEnter(event) {
            if (event.key === 'Enter') sendMessage();
        }
    </script>
</head>
<body>
    <div class="chat-container">
        <h2>WhatsApp Status FAQ Chatbot</h2>
        <div id="chat" style="height: 400px; overflow-y: auto; border: 1px solid #ccc;"></div>
        <div class="input-box">
            <input type="text" id="user-input" placeholder="Ask about WhatsApp statuses..." onkeypress="checkEnter(event)">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input or not user_input.strip():
        return jsonify({"response": "Please enter a question."})

    # Check for exit intent
    lang, is_greeting, is_exit = detect_language_and_check_intent(user_input)
    if is_exit:
        return jsonify({"response": "Goodbye!"})

    if is_greeting:
        return jsonify({"response": GREETING_RESPONSES[lang]})

    best_match_question, similarity = find_best_match(user_input, lang)

    if best_match_question:
        response = faq_answers[lang][best_match_question]
    else:
        response = "I'm sorry, I didn’t understand that. Could you please rephrase your question?"

    return jsonify({"response": response})

if __name__ == '__main__':
    load_faq_data()  # Load the data when the app starts
    app.run(host='0.0.0.0', port=5000)
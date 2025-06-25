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
SIMILARITY_THRESHOLD = 75
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
        "halo": "hai", "undh": "unduh", "wa": "whatsapp"
    }
    words = text.split()
    corrected_words = [manual_typos.get(word, word) for word in words]
    return " ".join(corrected_words)

# --- GREETING HANDLING ---
GREETINGS = {
    "en": {"hi", "hello", "hey", "good morning", "good evening"},
    "hi": {"नमस्ते", "हाय", "हेलो", "शुभ प्रभात", "शुभ संध्या"},
    "id": {"hai", "halo", "selamat pagi", "selamat malam"}
}
GREETING_RESPONSES = {
    "en": "Hello! How can I assist you with WhatsApp statuses?",
    "hi": "नमस्ते! मैं व्हाट्सएप स्टेटस के बारे में कैसे मदद कर सकता हूँ?",
    "id": "Hai! Bagaimana saya bisa membantu dengan status WhatsApp?"
}

def detect_language_and_check_greeting(text):
    """
    Detects the language of the input text and checks if it's a greeting.
    """
    normalized_words = set(normalize(text).split())
    
    # Check for greetings first
    for lang, greetings in GREETINGS.items():
        if normalized_words.intersection(greetings):
            return lang, True

    # If not a greeting, perform language detection
    try:
        lang = detect(text)
        return lang if lang in LANGUAGES else "en", False
    except:
        return "en", False # Default to English if detection fails

# --- FAQ DATA LOADING AND PREPARATION ---
# These global variables will hold the prepared FAQ data.
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
                # Ensure paraphrases is a list, default to empty list if key is missing or not a list.
                paraphrases = item.get("paraphrases", {}).get(lang, [])
                if isinstance(paraphrases, list):
                    for p in paraphrases:
                        search_corpus[lang].append((normalize(p), original_question))

        logger.info("FAQ data loaded and prepared successfully.")

    except FileNotFoundError:
        logger.error("FATAL: whatsapp_faq_multilingual.json not found. The chatbot cannot function without it.")
        # In a real application, you might want to exit or handle this more gracefully.
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"FATAL: Error processing JSON file: {e}")

# --- CORE MATCHING LOGIC ---
def find_best_match(question, lang):
    """
    Finds the best matching question from the corpus using fuzzy string matching.
    It now searches both questions and paraphrases.
    """
    normalized_question = normalize(question)
    
    # Extract just the phrases to search against
    phrases = [item[0] for item in search_corpus[lang]]
    
    if not phrases:
        return None, 0.0

    # Use rapidfuzz's process.extractOne to find the best match
    best_match_result = process.extractOne(
        normalized_question, 
        phrases, 
        scorer=fuzz.partial_ratio, 
        score_cutoff=SIMILARITY_THRESHOLD
    )

    if best_match_result:
        best_match_phrase, similarity, index = best_match_result
        # Get the original question associated with the matched phrase
        original_question = search_corpus[lang][index][1]
        return original_question, similarity
        
    return None, 0.0

# --- FLASK ROUTES ---
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>FAQ Chatbot</title>
    <style>
        body { font-family: sans-serif; }
        #chatbox { width: 400px; height: 500px; border: 1px solid #ccc; display: flex; flex-direction: column; }
        #messages { flex-grow: 1; padding: 10px; overflow-y: auto; }
        #userInput { display: flex; padding: 10px; }
        #userInput input { flex-grow: 1; border: 1px solid #ccc; padding: 8px; }
        #userInput button { padding: 8px 12px; border: none; background-color: #007bff; color: white; cursor: pointer; }
        .user-message { text-align: right; color: blue; }
        .bot-message { text-align: left; color: green; }
    </style>
</head>
<body>
    <h1>WhatsApp Status FAQ Chatbot</h1>
    <div id="chatbox">
        <div id="messages"></div>
        <div id="userInput">
            <input type="text" id="message" placeholder="Ask a question..." autocomplete="off"/>
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
    <script>
        function sendMessage() {
            const input = document.getElementById('message');
            const message = input.value;
            if (!message) return;

            const messagesDiv = document.getElementById('messages');
            messagesDiv.innerHTML += `<div class="user-message"><p><b>You:</b> ${message}</p></div>`;

            fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                messagesDiv.innerHTML += `<div class="bot-message"><p><b>Bot:</b> ${data.response}</p></div>`;
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            });
            input.value = '';
        }
        document.getElementById('message').addEventListener('keyup', function(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
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

    # Check for exit commands
    if user_input.lower() in ["exit", "quit", "bye"]:
        return jsonify({"response": "Goodbye!"})

    lang, is_greeting = detect_language_and_check_greeting(user_input)

    if is_greeting:
        return jsonify({"response": GREETING_RESPONSES[lang]})

    best_match_question, similarity = find_best_match(user_input, lang)

    if best_match_question:
        response = faq_answers[lang][best_match_question]
        logger.info(f"Matched '{user_input}' to '{best_match_question}' with similarity {similarity:.2f}")
    else:
        response = "I'm sorry, I didn’t understand that. Could you please rephrase your question?"
        logger.warning(f"No good match found for '{user_input}' in language '{lang}'.")

    return jsonify({"response": response})

if __name__ == '__main__':
    load_faq_data() # Load the data when the app starts
    app.run(host='0.0.0.0', port=5000)
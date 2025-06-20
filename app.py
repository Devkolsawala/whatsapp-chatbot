import json
from langdetect import detect, DetectorFactory
import re
from flask import Flask, request, jsonify, render_template_string
import logging
from rapidfuzz import fuzz

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DetectorFactory.seed = 0

# Text normalization
def normalize(text):
    if isinstance(text, str):
        text = text.lower()
        # Use Unicode range for Devanagari (Hindi) characters: \u0900-\u097F
        text = re.sub(r"[^\w\s\u0900-\u097F\!\?]", "", text)  # Preserve Hindi characters
        text = re.sub(r"[!?]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        manual_typos = {
            "helo": "hello", "watsap": "whatsapp", "whastapp": "whatsapp", "downlaod": "download",
            "हेलो": "नमस्ते", "वाट्सप": "व्हाट्सएप", "डाउनलोड्ड": "डाउनलोड",
            "halo": "hai", "undh": "unduh", "wa": "whatsapp"
        }
        words = text.split()
        corrected_words = [manual_typos.get(word, word) for word in words]
        return " ".join(corrected_words).strip()
    return text

# Language detection and greeting handling
english_greetings = {"hi", "hello", "hey", "good morning", "good evening"}
hindi_greetings = {"नमस्ते", "हाय", "हेलो", "शुभ प्रभात", "शुभ संध्या"}
indonesian_greetings = {"hai", "halo", "selamat pagi", "selamat malam"}

greeting_responses = {
    "en": "Hello! How can I assist you with WhatsApp statuses?",
    "hi": "नमस्ते! मैं व्हाट्सएप स्टेटस के बारे में कैसे मदद कर सकता हूँ?",
    "id": "Hai! Bagaimana saya bisa membantu dengan status WhatsApp?"
}

def detect_language_and_check_greeting(text):
    normalized_text = normalize(text)
    words = normalized_text.split()
    is_greeting = False
    lang = "en"  # Default language

    # Check for greetings in short inputs
    if len(words) <= 3:
        if any(word in hindi_greetings for word in words):
            is_greeting = True
            lang = "hi"
        elif any(word in indonesian_greetings for word in words):
            is_greeting = True
            lang = "id"
        elif any(word in english_greetings for word in words):
            is_greeting = True
            lang = "en"

    # If not a greeting, detect language
    if not is_greeting and normalized_text.strip():
        try:
            lang = detect(text)
            lang = lang if lang in ["hi", "id"] else "en"
        except:
            lang = "en"

    return lang, is_greeting

# Load FAQ data
try:
    with open('whatsapp_faq_multilingual.json', 'r', encoding='utf-8') as f:
        faqs = json.load(f)
except FileNotFoundError:
    logger.error("whatsapp_faq_multilingual.json not found")
    faqs = []  # Fallback to empty list if file is missing

# Prepare FAQ questions for fuzzy matching
faq_questions = {lang: [] for lang in ["en", "hi", "id"]}
faq_answers = {lang: {} for lang in ["en", "hi", "id"]}
for idx, item in enumerate(faqs):
    for lang in ["en", "hi", "id"]:
        faq_questions[lang].append(item["question"][lang])
        faq_answers[lang][item["question"][lang]] = item["answer"][lang]

# Function to find best match for a question with grammatical tolerance
def find_best_match(question, lang):
    normalized_question = normalize(question)
    if not faq_questions[lang]:
        return None, 0.0
    best_match = max(faq_questions[lang], key=lambda x: fuzz.partial_ratio(normalized_question, normalize(x)), default=None)
    similarity = fuzz.partial_ratio(normalized_question, normalize(best_match)) if best_match else 0.0
    # Lower threshold to 60 to handle grammatical errors (e.g., missing articles, wrong verb forms)
    return best_match, similarity if similarity >= 60 else 0.0

# HTML template
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
    if not user_input:
        return jsonify({"response": "Please enter a question."})

    if user_input.lower() in ["exit", "quit"]:
        return jsonify({"response": "Goodbye!"})

    lang, is_greeting = detect_language_and_check_greeting(user_input)
    if is_greeting:
        return jsonify({"response": greeting_responses[lang]})

    normalized_input = normalize(user_input)
    if not normalized_input.strip():
        return jsonify({"response": "Please enter a valid question."})

    # Find the best matching question with tolerance for grammatical errors
    best_match, similarity = find_best_match(user_input, lang)
    if best_match and similarity >= 60:  # Lower threshold for grammatical flexibility
        response = faq_answers[lang][best_match]
    else:
        response = "I'm not sure how to answer that. Try rephrasing your question."

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
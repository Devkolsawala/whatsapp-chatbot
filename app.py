import json
import re
from langdetect import detect, DetectorFactory
from rapidfuzz import process, fuzz
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 0

# Load dataset
dataset_path = 'whatsapp_faq_multilingual.json'
try:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        faqs = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset '{dataset_path}' not found.")

# Preprocess dataset
questions = {"en": [], "hi": [], "id": []}
answers = {"en": {}, "hi": {}, "id": {}}
for idx, item in enumerate(faqs):
    for lang in ["en", "hi", "id"]:
        qlist = [item["question"][lang]] + item["paraphrases"][lang]
        questions[lang].extend(qlist)
        answers[lang][idx] = item["answer"][lang]

# Normalize text
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s?!]", "", text)
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

# Detect language
def detect_language(text):
    normalized_text = normalize(text)
    words = normalized_text.split()
    if len(words) <= 3:
        hindi_greetings = {"नमस्ते", "हाय", "हेलो", "शुभ प्रभात", "शुभ संध्या"}
        indonesian_greetings = {"hai", "halo", "selamat pagi", "selamat malam"}
        english_greetings = {"hi", "hello", "hey", "good morning", "good evening"}
        if any(word in hindi_greetings for word in words):
            return "hi"
        if any(word in indonesian_greetings for word in words):
            return "id"
        if any(word in english_greetings for word in words):
            return "en"
    try:
        lang = detect(text)
        return lang if lang in ["hi", "id"] else "en"
    except:
        return "en"

# Find closest question
def find_closest_question(user_input, lang_questions, lang):
    normalized_input = normalize(user_input)
    choices = [normalize(q) for q in lang_questions]
    best_match, score, _ = process.extractOne(normalized_input, choices, scorer=fuzz.token_set_ratio)
    if score >= 80:
        return lang_questions[choices.index(best_match)]
    return None

# HTML template for the web interface
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

    lang = detect_language(user_input)
    closest_question = find_closest_question(user_input, questions[lang], lang)

    if closest_question:
        for idx, item in enumerate(faqs):
            if closest_question in [item["question"][lang]] + item["paraphrases"][lang]:
                return jsonify({"response": answers[lang][idx]})
    else:
        return jsonify({"response": "I'm not sure how to answer that. Try rephrasing your question."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
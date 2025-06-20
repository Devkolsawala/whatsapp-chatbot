import json
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from langdetect import detect, DetectorFactory
import re
from flask import Flask, request, jsonify, render_template_string
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DetectorFactory.seed = 0

# Load tokenizer and TFLite model
logger.info("Loading tokenizer and TFLite model...")
tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
try:
    interpreter = tf.lite.Interpreter(model_path='finetuned_tinybert_multilingual.tflite')
    interpreter.allocate_tensors()
except Exception as e:
    logger.error(f"Failed to load TFLite model: {e}")
    exit(1)

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

# TFLite classification
def classify(text):
    inputs = tokenizer(
        text, padding='max_length', truncation=True, max_length=128,
        return_tensors="np", return_attention_mask=True
    )
    input_ids = inputs["input_ids"].astype(np.int32)
    attention_mask = inputs["attention_mask"].astype(np.int32)
    interpreter.set_tensor(input_details[0]['index'], input_ids)
    interpreter.set_tensor(input_details[1]['index'], attention_mask)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    scale, zero_point = output_details[0]['quantization']
    output = (output.astype(np.float32) - zero_point) * scale
    return output

# Load FAQ data
try:
    with open('whatsapp_faq_multilingual.json', 'r', encoding='utf-8') as f:
        faqs = json.load(f)
except FileNotFoundError:
    logger.error("whatsapp_faq_multilingual.json not found")
    exit(1)

# Process FAQ data
label_to_answer = {"en": {}, "hi": {}, "id": {}}
for idx, item in enumerate(faqs):
    for lang in ["en", "hi", "id"]:
        label_to_answer[lang][idx] = item["answer"][lang]

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
    probabilities = classify(normalized_input)
    predicted_label = np.argmax(probabilities, axis=1)[0]
    confidence = probabilities[0, predicted_label]
    if confidence >= 0.7:
        response = label_to_answer[lang].get(predicted_label, "Unknown category")
    else:
        response = "I'm not sure how to answer that. Try rephrasing your question."
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
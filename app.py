import json
import re
from flask import Flask, request, jsonify, render_template_string
import logging
from langdetect import detect, DetectorFactory
from rapidfuzz import process, fuzz

app = Flask(__name__)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DetectorFactory.seed = 0
INDEX_FILE = 'faq_index.json'
LANGUAGES = ["en", "hi", "id"]
FUZZY_MATCH_THRESHOLD = 80 # A good starting point for typo tolerance
MINIMUM_SCORE_THRESHOLD = 0.25 # Threshold for a match to be considered valid

# --- LOAD THE UNIFIED SEARCH INDEX ---
try:
    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        search_index = json.load(f)
    logger.info("Final search index loaded successfully.")
except FileNotFoundError:
    logger.error(f"FATAL: '{INDEX_FILE}' not found. Please run build_index.py first.")
    exit()

def normalize_and_tokenize_query(text):
    if not isinstance(text, str): return []
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s\u0900-\u097F]", "", text)
    return text.split()

def find_best_match(user_query):
    """Searches the unified index using IDF and fuzzy matching, returning the best document."""
    user_keywords = normalize_and_tokenize_query(user_query)
    if not user_keywords:
        return None

    documents = search_index.get('documents', [])
    idf_scores = search_index.get('idf_scores', {})
    
    best_score = 0
    best_match_doc = None

    for doc in documents:
        doc_keywords = doc['keywords']
        current_doc_score = 0
        
        for user_word in user_keywords:
            # Find the best fuzzy match for the user's word in this document's keywords
            # We check against all keywords since the document represents a single intent
            best_match = process.extractOne(user_word, doc_keywords, scorer=fuzz.ratio)
            
            if best_match and best_match[1] >= FUZZY_MATCH_THRESHOLD:
                matched_keyword = best_match[0]
                # The score is a combination of how similar the word is (fuzzy score) and how important it is (IDF score)
                keyword_importance = idf_scores.get(matched_keyword, 0.05) # Give low importance to unknown words
                current_doc_score += (best_match[1] / 100) * keyword_importance

        if current_doc_score > best_score:
            best_score = current_doc_score
            best_match_doc = doc
            
    if best_match_doc and best_score >= MINIMUM_SCORE_THRESHOLD:
        logger.info(f"Found match for '{user_query}' in doc '{best_match_doc['id']}' with score {best_score:.2f}")
        return best_match_doc
    else:
        logger.warning(f"No good match found for query '{user_query}' (Best score: {best_score:.2f})")
        return None

# --- HTML & FLASK ROUTES ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FAQ Chatbot</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');
        body {
            font-family: 'Roboto', sans-serif; margin: 0; background-color: #e5ddd5;
            display: flex; justify-content: center; align-items: center; height: 100vh;
        }
        .chat-container {
            width: 100%; max-width: 450px; height: 90vh; max-height: 700px;
            display: flex; flex-direction: column; background-color: #f0f0f0;
            border-radius: 8px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); overflow: hidden;
        }
        .chat-header {
            background-color: #075E54; color: white; padding: 15px 20px;
            font-weight: 500; font-size: 1.1rem; text-align: center;
        }
        .chat-box {
            flex-grow: 1; padding: 20px; overflow-y: auto;
            background-image: url('https://user-images.githubusercontent.com/15075759/28719144-86dc0f70-73b1-11e7-911d-60d70fcded21.png');
        }
        .message { display: flex; margin-bottom: 15px; max-width: 80%; }
        .message-content { padding: 10px 15px; border-radius: 12px; line-height: 1.4; white-space: pre-wrap; }
        .user { margin-left: auto; flex-direction: row-reverse; }
        .user .message-content { background-color: #dcf8c6; color: #303030; border-top-right-radius: 0; }
        .bot .message-content { background-color: #ffffff; color: #303030; border-top-left-radius: 0; }
        .typing-indicator {
            display: none; padding: 10px 15px; background-color: #ffffff;
            border-radius: 12px; border-top-left-radius: 0; align-items: center;
        }
        .typing-indicator span {
            height: 8px; width: 8px; background-color: #999; border-radius: 50%;
            display: inline-block; margin: 0 2px; animation: bounce 1.3s infinite ease-in-out;
        }
        .typing-indicator span:nth-child(2) { animation-delay: -1.1s; }
        .typing-indicator span:nth-child(3) { animation-delay: -0.9s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1.0); } }
        .input-area { display: flex; padding: 10px; background-color: #f0f0f0; border-top: 1px solid #ddd; }
        #user-input {
            flex-grow: 1; border: none; padding: 12px 15px;
            border-radius: 25px; font-size: 1rem;
        }
        #user-input:focus { outline: none; }
        #send-btn {
            background-color: #128C7E; color: white; border: none; border-radius: 50%;
            width: 50px; height: 50px; margin-left: 10px; cursor: pointer; font-size: 24px;
            display: flex; justify-content: center; align-items: center; transition: background-color 0.2s;
        }
        #send-btn:hover { background-color: #075E54; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">WhatsApp FAQ Bot</div>
        <div class="chat-box" id="chat-box">
            <div class="message bot"><div class="message-content">Hello! How can I assist you today?</div></div>
        </div>
        <div class="input-area">
            <input type="text" id="user-input" placeholder="Type your question..." autocomplete="off">
            <button id="send-btn">&#10148;</button>
        </div>
    </div>
    <script>
        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        function addMessage(content, type) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', type);
            const messageContent = document.createElement('div');
            messageContent.classList.add('message-content');
            messageContent.innerHTML = content.replace(/\\n/g, '<br>');
            messageDiv.appendChild(messageContent);
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        function showTypingIndicator() {
            const typingDiv = document.createElement('div');
            typingDiv.classList.add('message', 'bot', 'typing-indicator');
            typingDiv.innerHTML = '<span></span><span></span><span></span>';
            chatBox.appendChild(typingDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        function hideTypingIndicator() {
            const indicator = document.querySelector('.typing-indicator');
            if (indicator) { indicator.remove(); }
        }
        async function handleSendMessage() {
            const message = userInput.value.trim();
            if (!message) return;
            addMessage(message, 'user');
            userInput.value = '';
            showTypingIndicator();
            await new Promise(resolve => setTimeout(resolve, 400));
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                const data = await response.json();
                hideTypingIndicator();
                addMessage(data.response, 'bot');
            } catch (error) {
                hideTypingIndicator();
                addMessage('Sorry, something went wrong. Please try again.', 'bot');
                console.error('Error:', error);
            }
        }
        sendBtn.addEventListener('click', handleSendMessage);
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); handleSendMessage(); }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '').strip()
    if not user_input:
        return jsonify({"response": "Please enter a question."})

    try:
        reply_lang = detect(user_input)
        if reply_lang not in LANGUAGES:
            # For Hinglish, langdetect often returns 'hi' or 'en'. 
            # If it's 'en', we search all keywords anyway.
            # If it's something else entirely, default to English for the reply.
            reply_lang = 'en'
    except Exception:
        reply_lang = 'en' # Default to English on any detection error
        
    logger.info(f"Detected reply language: '{reply_lang}' for query: '{user_input}'")

    best_doc = find_best_match(user_input)
    
    if best_doc:
        # Use the detected language to pick the right answer from the matched document
        response_text = best_doc['answers'].get(reply_lang, best_doc['answers']['en'])
    else:
        response_text = "I'm sorry, I didn’t understand that. Please rephrase your question."

    return jsonify({"response": response_text})

if __name__ == '__main__':
    logger.info("Starting Final Intelligent Keyword-Based Flask server...")
    app.run(host='0.0.0.0', port=5000)

# import json
# from langdetect import detect, DetectorFactory
# import re
# from flask import Flask, request, jsonify, render_template_string
# import logging
# import tensorflow as tf
# import numpy as np
# from transformers import AutoTokenizer
# from scipy.spatial.distance import cosine
# import urllib.request
# import os
# import shutil

# app = Flask(__name__)

# # --- CONFIGURATION ---
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# DetectorFactory.seed = 0
# SIMILARITY_THRESHOLD = 0.7
# LANGUAGES = ["en", "hi", "id"]

# # --- DOWNLOAD TFLITE MODEL AND TOKENIZER ---
# TFLITE_URL = "<your-tflite-url>"  # Replace with Google Drive URL
# TOKENIZER_ZIP_URL = "<your-tokenizer-zip-url>"

# if not os.path.exists("tinybert.tflite"):
#     urllib.request.urlretrieve(TFLITE_URL, "tinybert.tflite")

# if not os.path.exists("./tinybert_model"):
#     urllib.request.urlretrieve(TOKENIZER_ZIP_URL, "tinybert_model.zip")
#     shutil.unpack_archive("tinybert_model.zip", "./tinybert_model")
#     os.remove("tinybert_model.zip")

# # --- LOAD TFLITE MODEL AND TOKENIZER ---
# interpreter = tf.lite.Interpreter(model_path="tinybert.tflite")
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# tokenizer = AutoTokenizer.from_pretrained("./tinybert_model")

# # Debug: Print input details
# logger.info(f"Input details: {input_details}")

# # --- TEXT NORMALIZATION ---
# def normalize(text):
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r"[^\w\s\u0900-\u097F]", "", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     manual_typos = {
#         "helo": "hello", "watsap": "whatsapp", "whastapp": "whatsapp", "downlaod": "download",
#         "हेलो": "नमस्ते", "वाट्सप": "व्हाट्सएप", "डाउनलोड्ड": "डाउनलोड",
#         "halo": "hai", "undh": "unduh", "wa": "whatsapp",
#         "exir": "exit", "exittt": "exit", "quittt": "quit", "byee": "bye"
#     }
#     words = text.split()
#     corrected_words = [manual_typos.get(word, word) for word in words]
#     return " ".join(corrected_words)

# # --- GREETING AND EXIT HANDLING ---
# GREETINGS = {
#     "en": {"hi", "hello", "hey", "good morning", "good evening"},
#     "hi": {"नमस्ते", "हाय", "हेलो", "शुभ प्रभात", "शुभ संध्या"},
#     "id": {"hai", "halo", "selamat pagi", "selamat malam"}
# }
# EXIT_COMMANDS = {"exit", "quit", "bye"}
# GREETING_RESPONSES = {
#     "en": "Hello! How can I assist you with WhatsApp statuses?",
#     "hi": "नमस्ते! मैं व्हाट्सएप स्टेटस के बारे में कैसे मदद कर सकता हूँ?",
#     "id": "Hai! Bagaimana saya bisa membantu dengan status WhatsApp?"
# }

# def detect_language_and_check_intent(text):
#     normalized_text = normalize(text)
#     normalized_words = set(normalized_text.split())
#     for lang in LANGUAGES:
#         if any(word in EXIT_COMMANDS for word in normalized_words):
#             return lang, False, True
#     for lang, greetings in GREETINGS.items():
#         if normalized_words.intersection(greetings):
#             return lang, True, False
#     try:
#         lang = detect(text)
#         return lang if lang in LANGUAGES else "en", False, False
#     except:
#         return "en", False, False

# # --- FAQ DATA LOADING AND PREPARATION ---
# faq_answers = {lang: {} for lang in LANGUAGES}
# search_corpus = {lang: [] for lang in LANGUAGES}
# faq_embeddings = {lang: [] for lang in LANGUAGES}

# def load_faq_data():
#     try:
#         with open('whatsapp_faq_multilingual.json', 'r', encoding='utf-8') as f:
#             faqs = json.load(f)
#         for item in faqs:
#             for lang in LANGUAGES:
#                 original_question = item["question"][lang]
#                 answer = item["answer"][lang]
#                 faq_answers[lang][original_question] = answer
#                 search_corpus[lang].append((normalize(original_question), original_question))
#                 paraphrases = item.get("paraphrases", {}).get(lang, [])
#                 if isinstance(paraphrases, list):
#                     for p in paraphrases:
#                         search_corpus[lang].append((normalize(p), original_question))
#         logger.info("FAQ data loaded and prepared successfully.")
#     except FileNotFoundError:
#         logger.error("FATAL: whatsapp_faq_multilingual.json not found.")
#     except (json.JSONDecodeError, KeyError) as e:
#         logger.error(f"FATAL: Error processing JSON file: {e}")

# def compute_embedding(text):
#     inputs = tokenizer(text, return_tensors="np", padding='max_length', truncation=True, max_length=128)
#     input_ids = inputs["input_ids"].astype(np.int64)
#     attention_mask = inputs["attention_mask"].astype(np.int64)
    
#     # Debug: Print shapes
#     logger.debug(f"input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}")
    
#     # Set tensors in correct order
#     interpreter.set_tensor(input_details[0]["index"], attention_mask)  # serving_default_attention_mask:0
#     interpreter.set_tensor(input_details[1]["index"], input_ids)      # serving_default_input_ids:0
#     interpreter.invoke()
#     embedding = interpreter.get_tensor(output_details[0]["index"])[0, 0, :]
#     return embedding

# def precompute_faq_embeddings():
#     for lang in LANGUAGES:
#         for _, original_question in search_corpus[lang]:
#             embedding = compute_embedding(original_question)
#             faq_embeddings[lang].append((embedding, original_question))
#     logger.info("FAQ embeddings precomputed successfully.")

# def find_best_match(question, lang):
#     normalized_question = normalize(question)
#     question_embedding = compute_embedding(normalized_question)
#     if not faq_embeddings[lang]:
#         return None, 0.0
#     best_similarity = -1
#     best_question = None
#     for faq_embedding, original_question in faq_embeddings[lang]:
#         similarity = 1 - cosine(question_embedding, faq_embedding)
#         if similarity > best_similarity and similarity >= SIMILARITY_THRESHOLD:
#             best_similarity = similarity
#             best_question = original_question
#     if best_question:
#         logger.info(f"Matched '{question}' to '{best_question}' with similarity {best_similarity:.2f}")
#     else:
#         logger.warning(f"No good match found for '{question}' in language '{lang}'.")
#     return best_question, best_similarity * 100 if best_question else 0.0

# # --- FLASK ROUTES ---
# html_template = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <title>WhatsApp Status FAQ Chatbot</title>
#     <style>
#         body { font-family: Arial, sans-serif; margin: 20px; }
#         .chat-container { max-width: 600px; margin: auto; }
#         .message { margin: 10px; padding: 10px; border-radius: 5px; }
#         .user { background: #e0f7fa; text-align: right; }
#         .bot { background: #f1f8e9; text-align: left; }
#         .input-box { margin-top: 20px; }
#         input[type="text"] { width: 80%; padding: 10px; }
#         button { padding: 10px; }
#     </style>
#     <script>
#         function sendMessage() {
#             const input = document.getElementById('user-input');
#             const message = input.value.trim();
#             if (!message) return;
#             const chat = document.getElementById('chat');
#             chat.innerHTML += `<div class="message user">${message}</div>`;
#             fetch('/chat', {
#                 method: 'POST',
#                 headers: {'Content-Type': 'application/json'},
#                 body: JSON.stringify({message: message})
#             })
#             .then(response => response.json())
#             .then(data => {
#                 chat.innerHTML += `<div class="message bot">${data.response}</div>`;
#                 chat.scrollTop = chat.scrollHeight;
#             });
#             input.value = '';
#         }
#         function checkEnter(event) {
#             if (event.key === 'Enter') sendMessage();
#         }
#     </script>
# </head>
# <body>
#     <div class="chat-container">
#         <h2>WhatsApp Status FAQ Chatbot</h2>
#         <div id="chat" style="height: 400px; overflow-y: auto; border: 1px solid #ccc;"></div>
#         <div class="input-box">
#             <input type="text" id="user-input" placeholder="Ask about WhatsApp statuses..." onkeypress="checkEnter(event)">
#             <button onclick="sendMessage()">Send</button>
#         </div>
#     </div>
# </body>
# </html>
# """

# @app.route('/')
# def home():
#     return render_template_string(html_template)

# @app.route('/chat', methods=['POST'])
# def chat():
#     try:
#         user_input = request.json.get('message')
#         if not user_input or not user_input.strip():
#             return jsonify({"response": "Please enter a question."})
#         lang, is_greeting, is_exit = detect_language_and_check_intent(user_input)
#         if is_exit:
#             return jsonify({"response": "Goodbye!"})
#         if is_greeting:
#             return jsonify({"response": GREETING_RESPONSES[lang]})
#         best_match_question, similarity = find_best_match(user_input, lang)
#         if best_match_question:
#             response = faq_answers[lang][best_match_question]
#         else:
#             response = "I'm sorry, I didn’t understand that. Please check your spelling, rephrase your question, or type 'exit' to end the conversation."
#             logger.warning(f"No match found for '{user_input}' in language '{lang}'.")
#         return jsonify({"response": response})
#     except Exception as e:
#         logger.error(f"Exception in /chat: {e}")
#         return jsonify({"response": "An error occurred. Please try again later or contact support."})

# if __name__ == '__main__':
#     load_faq_data()
#     precompute_faq_embeddings()
#     app.run(host='0.0.0.0', port=5000)
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
        response_text = "I'm sorry, I didn’t understand the question. Please rephrase your question and try to be more specific."

    return jsonify({"response": response_text})

if __name__ == '__main__':
    logger.info("Starting Final Intelligent Keyword-Based Flask server...")
    app.run(host='0.0.0.0', port=5000)

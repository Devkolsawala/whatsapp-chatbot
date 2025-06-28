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
    # For demo purposes, create a simple fallback
    search_index = {
        'documents': [
            {
                'id': 'demo_1',
                'keywords': ['hello', 'hi', 'greeting'],
                'answers': {
                    'en': 'Hello! How can I help you today?',
                    'hi': 'नमस्ते! मैं आपकी कैसे सहायता कर सकता हूं?',
                    'id': 'Halo! Bagaimana saya bisa membantu Anda hari ini?'
                }
            }
        ],
        'idf_scores': {'hello': 0.5, 'hi': 0.5, 'greeting': 0.5}
    }

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
            best_match = process.extractOne(user_word, doc_keywords, scorer=fuzz.ratio)
            
            if best_match and best_match[1] >= FUZZY_MATCH_THRESHOLD:
                matched_keyword = best_match[0]
                keyword_importance = idf_scores.get(matched_keyword, 0.05)
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

# Enhanced HTML Template with authentic WhatsApp styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WhatsApp FAQ Bot</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Roboto', sans-serif;
            background: #e5ddd5;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        
        .chat-container {
            width: 100%;
            max-width: 450px;
            height: 90vh;
            max-height: 700px;
            display: flex;
            flex-direction: column;
            background-color: #f0f0f0;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #075E54 0%, #128C7E 100%);
            color: white;
            padding: 15px 20px;
            font-weight: 500;
            font-size: 1.1rem;
            text-align: center;
            position: relative;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .chat-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
            opacity: 0.3;
        }
        
        .chat-header h1 {
            font-size: 1.1rem;
            font-weight: 500;
            position: relative;
            z-index: 1;
        }
        
        .chat-box {
            flex-grow: 1;
            padding: 20px;
            overflow-y: auto;
            background-color: #e5ddd5;
            background-image: url('https://user-images.githubusercontent.com/15075759/28719144-86dc0f70-73b1-11e7-911d-60d70fcded21.png');
            background-repeat: repeat;
            background-size: auto;
        }
        
        .chat-box::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-box::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .chat-box::-webkit-scrollbar-thumb {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
        }
        
        .message {
            display: flex;
            margin-bottom: 15px;
            max-width: 80%;
            animation: messageSlide 0.3s ease-out;
        }
        
        @keyframes messageSlide {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message-content {
            padding: 10px 15px;
            border-radius: 12px;
            line-height: 1.4;
            white-space: pre-wrap;
            word-wrap: break-word;
            position: relative;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
            font-size: 14px;
        }
        
        .user {
            margin-left: auto;
            flex-direction: row-reverse;
        }
        
        .user .message-content {
            background-color: #dcf8c6;
            color: #303030;
            border-top-right-radius: 0;
        }
        
        .user .message-content::after {
            content: '';
            position: absolute;
            bottom: 0;
            right: -8px;
            width: 0;
            height: 0;
            border: 8px solid transparent;
            border-left-color: #dcf8c6;
            border-bottom: none;
            border-right: none;
        }
        
        .bot .message-content {
            background-color: #ffffff;
            color: #303030;
            border-top-left-radius: 0;
        }
        
        .bot .message-content::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: -8px;
            width: 0;
            height: 0;
            border: 8px solid transparent;
            border-right-color: #ffffff;
            border-bottom: none;
            border-left: none;
        }
        
        .typing-indicator {
            display: none;
            padding: 10px 15px;
            background-color: #ffffff;
            border-radius: 12px;
            border-top-left-radius: 0;
            align-items: center;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
            animation: messageSlide 0.3s ease-out;
        }
        
        .typing-indicator::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: -8px;
            width: 0;
            height: 0;
            border: 8px solid transparent;
            border-right-color: #ffffff;
            border-bottom: none;
            border-left: none;
        }
        
        .typing-indicator span {
            height: 8px;
            width: 8px;
            background-color: #999;
            border-radius: 50%;
            display: inline-block;
            margin: 0 2px;
            animation: bounce 1.3s infinite ease-in-out;
        }
        
        .typing-indicator span:nth-child(2) { animation-delay: -1.1s; }
        .typing-indicator span:nth-child(3) { animation-delay: -0.9s; }
        
        @keyframes bounce {
            0%, 80%, 100% {
                transform: scale(0);
            }
            40% {
                transform: scale(1.0);
            }
        }
        
        .input-area {
            display: flex;
            padding: 10px;
            background-color: #f0f0f0;
            border-top: 1px solid #ddd;
            gap: 10px;
        }
        
        #user-input {
            flex-grow: 1;
            border: none;
            padding: 12px 15px;
            border-radius: 25px;
            font-size: 1rem;
            font-family: 'Roboto', sans-serif;
            background-color: white;
            transition: all 0.2s ease;
        }
        
        #user-input:focus {
            outline: none;
            box-shadow: 0 0 0 2px rgba(37, 211, 102, 0.3);
        }
        
        #user-input::placeholder {
            color: #999;
        }
        
        #send-btn {
            background: linear-gradient(135deg, #128C7E 0%, #075E54 100%);
            color: white;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            cursor: pointer;
            font-size: 24px;
            display: flex;
            justify-content: center;
            align-items: center;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        #send-btn:hover {
            background: linear-gradient(135deg, #075E54 0%, #128C7E 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        
        #send-btn:active {
            transform: translateY(0);
        }
        
        #send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        /* Mobile responsiveness */
        @media (max-width: 480px) {
            body {
                padding: 10px;
            }
            
            .chat-container {
                height: 95vh;
                border-radius: 0;
            }
            
            .chat-header {
                padding: 12px 16px;
            }
            
            .chat-box {
                padding: 16px;
            }
            
            .input-area {
                padding: 8px;
            }
            
            .message {
                max-width: 85%;
            }
        }
        
        /* Loading animation for send button */
        .loading {
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>WhatsApp FAQ Bot</h1>
        </div>
        <div class="chat-box" id="chat-box">
            <div class="message bot">
                <div class="message-content">Hello! How can I assist you today?</div>
            </div>
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
        let isProcessing = false;

        function addMessage(content, type) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', type);
            
            const messageContent = document.createElement('div');
            messageContent.classList.add('message-content');
            messageContent.innerHTML = content.replace(/\\n/g, '<br>');
            
            messageDiv.appendChild(messageContent);
            chatBox.appendChild(messageDiv);
            
            // Smooth scroll to bottom
            setTimeout(() => {
                chatBox.scrollTop = chatBox.scrollHeight;
            }, 100);
        }

        function showTypingIndicator() {
            const typingDiv = document.createElement('div');
            typingDiv.classList.add('message', 'bot');
            typingDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
            chatBox.appendChild(typingDiv);
            
            const indicator = typingDiv.querySelector('.typing-indicator');
            indicator.style.display = 'flex';
            
            setTimeout(() => {
                chatBox.scrollTop = chatBox.scrollHeight;
            }, 100);
        }

        function hideTypingIndicator() {
            const indicator = document.querySelector('.typing-indicator');
            if (indicator) {
                indicator.closest('.message').remove();
            }
        }

        async function handleSendMessage() {
            const message = userInput.value.trim();
            if (!message || isProcessing) return;

            isProcessing = true;
            sendBtn.disabled = true;
            sendBtn.innerHTML = '⟳';
            sendBtn.classList.add('loading');

            addMessage(message, 'user');
            userInput.value = '';

            showTypingIndicator();

            // Simulate network delay
            await new Promise(resolve => setTimeout(resolve, 400));

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });

                const data = await response.json();
                hideTypingIndicator();
                
                // Add a small delay for better UX
                setTimeout(() => {
                    addMessage(data.response, 'bot');
                }, 200);

            } catch (error) {
                hideTypingIndicator();
                addMessage('Sorry, something went wrong. Please try again.', 'bot');
                console.error('Error:', error);
            } finally {
                isProcessing = false;
                sendBtn.disabled = false;
                sendBtn.innerHTML = '&#10148;';
                sendBtn.classList.remove('loading');
                userInput.focus();
            }
        }

        sendBtn.addEventListener('click', handleSendMessage);
        
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                handleSendMessage();
            }
        });

        // Auto-focus input on load
        window.addEventListener('load', () => {
            userInput.focus();
        });

        // Add some interactive feedback
        userInput.addEventListener('input', () => {
            if (userInput.value.trim()) {
                sendBtn.style.opacity = '1';
            } else {
                sendBtn.style.opacity = '0.7';
            }
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
            reply_lang = 'en'
    except Exception:
        reply_lang = 'en'
        
    logger.info(f"Detected reply language: '{reply_lang}' for query: '{user_input}'")
    best_doc = find_best_match(user_input)
    
    if best_doc:
        response_text = best_doc['answers'].get(reply_lang, best_doc['answers']['en'])
    else:
        response_text = "I'm sorry, I didn't understand the question. Please rephrase your question and try to be more specific."

    return jsonify({"response": response_text})

if __name__ == '__main__':
    logger.info("Starting WhatsApp FAQ Bot Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)

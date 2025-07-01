import json
import re
import os
from flask import Flask, request, jsonify, render_template_string
import logging
from langdetect import detect, DetectorFactory, LangDetectException
from rapidfuzz import process, fuzz

app = Flask(__name__)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DetectorFactory.seed = 0

INDEX_FILE = 'faq_index.json'
LANGUAGES = ["en", "hi", "id", "hinglish"]
FUZZY_MATCH_THRESHOLD = 75  # Lowered for better matching
MINIMUM_SCORE_THRESHOLD = 0.3  # Lowered for flexibility
ACTION_WEIGHT = 2.0  # Weight for action keywords

# --- Load search index with better error handling ---
def load_search_index():
    try:
        if not os.path.exists(INDEX_FILE):
            logger.warning(f"Index file '{INDEX_FILE}' not found. Creating fallback data.")
            return create_fallback_index()
        
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            search_index = json.load(f)
        
        if not isinstance(search_index, dict) or 'documents' not in search_index:
            logger.warning("Invalid index structure. Creating fallback data.")
            return create_fallback_index()
            
        logger.info(f"Search index loaded successfully with {len(search_index['documents'])} documents.")
        return search_index
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in '{INDEX_FILE}': {e}")
        return create_fallback_index()
    except Exception as e:
        logger.error(f"Error loading '{INDEX_FILE}': {e}")
        return create_fallback_index()

def create_fallback_index():
    """Create a basic fallback index for common WhatsApp Status Saver questions"""
    return {
        "documents": [
            {
                "keywords": ["download", "status", "save", "whatsapp", "story"],
                "answers": {
                    "en": "To save a status:\n1. View the status in WhatsApp.\n2. Open Status Saver app to see viewed statuses.\n3. Tap 'Download' to save it.",
                    "hi": "स्टेटस सेव करने के लिए:\n1. WhatsApp में स्टेटस देखें।\n2. Status Saver ऐप खोलें, जहां देखे गए स्टेटस दिखेंगे।\n3. 'डाउनलोड' पर टैप करें।",
                    "hinglish": "Status save karne ke liye:\n1. WhatsApp mein status dekho.\n2. Status Saver app kholo, jahan dekhe hue status dikhenge.\n3. 'Download' tap karo."
                }
            },
            {
                "keywords": ["share", "status", "whatsapp", "send", "forward", "bhej"],
                "answers": {
                    "en": "To share a status:\n1. Save the status using Status Saver (see save instructions).\n2. Open the saved status in the app.\n3. Tap 'Share' and select a platform or contact.",
                    "hi": "स्टेटस शेयर करने के लिए:\n1. Status Saver से स्टेटस सेव करें (सेव करने के निर्देश देखें)।\n2. ऐप में सेव किया हुआ स्टेटस खोलें।\n3. 'शेयर' पर टैप करें और प्लेटफॉर्म या संपर्क चुनें।",
                    "hinglish": "Status share karne ke liye:\n1. Status Saver se status save karo (save instructions dekho).\n2. App mein saved status kholo.\n3. 'Share' tap karo aur platform ya contact chuno."
                }
            },
            {
                "keywords": ["permission", "storage", "access", "allow", "settings"],
                "answers": {
                    "en": "If download isn’t working, check permissions:\nGo to Phone Settings > Apps > Status Saver > Permissions > Enable Storage.",
                    "hi": "अगर डाउनलोड काम नहीं कर रहा, अनुमतियां जांचें:\nफोन सेटिंग्स > ऐप्स > Status Saver > अनुमतियां > स्टोरेज चालू करें।",
                    "hinglish": "Agar download nahi chal raha, permissions check karo:\nPhone Settings > Apps > Status Saver > Permissions > Storage enable karo."
                }
            },
            {
                "keywords": ["error", "problem", "not", "working", "issue", "failed"],
                "answers": {
                    "en": "Common issues:\n1. No storage permission\n2. Low storage space\n3. Bad internet\n4. WhatsApp server issues",
                    "hi": "आम समस्याएं:\n1. स्टोरेज अनुमति नहीं\n2. कम स्टोरेज स्पेस\n3. खराब इंटरनेट\n4. WhatsApp सर्वर समस्याएं",
                    "hinglish": "Common problems:\n1. Storage permission nahi\n2. Kam storage space\n3. Kharab internet\n4. WhatsApp server issues"
                }
            }
        ],
        "idf_scores": {
            "download": 0.8, "status": 0.9, "save": 0.7, "whatsapp": 0.6,
            "share": 0.8, "send": 0.7, "forward": 0.7, "bhej": 0.6,
            "permission": 0.8, "storage": 0.7, "error": 0.6, "problem": 0.6
        }
    }

search_index = load_search_index()

# --- Enhanced Nonsense Filtering ---
IRRELEVANT_TOPICS = {
    "vehicles": {"car", "bike", "truck", "bicycle", "scooter", "train", "aeroplane", "airplane", "jet", "boat", "ship", "vehicle"},
    "appliances": {"fridge", "refrigerator", "oven", "stove", "fan", "bulb", "ac", "cooler", "mixer", "toaster", "washing", "machine"},
    "animals": {"dog", "cat", "elephant", "horse", "monkey", "tiger", "lion", "goat", "cow", "bird", "fish"},
    "foods": {"pizza", "burger", "cake", "bread", "milk", "apple", "rice", "chapati", "noodles", "icecream", "food", "eat"},
    "clothing": {"jeans", "shirt", "jacket", "tshirt", "saree", "kurta", "shoes", "cap", "dress", "clothes"},
    "other": {"road", "tree", "girl", "boy", "kid", "pen", "bottle", "house", "wall", "weather", "time", "date"}
}

NONSENSE_FILTER = {
    "download": set.union(*IRRELEVANT_TOPICS.values()),
    "save": set.union(*IRRELEVANT_TOPICS.values()),
    "install": set.union(*IRRELEVANT_TOPICS.values()),
    "upload": set.union(*IRRELEVANT_TOPICS.values()),
    "record": set.union(*IRRELEVANT_TOPICS.values()),
    "share": set.union(*IRRELEVANT_TOPICS.values()),
    "send": set.union(*IRRELEVANT_TOPICS.values()),
    "forward": set.union(*IRRELEVANT_TOPICS.values()),
    "bhej": set.union(*IRRELEVANT_TOPICS.values())
}

SPAM_PATTERNS = [
    r'^[a-z]{1,3}$',  # Very short nonsense
    r'^[0-9]+$',      # Only numbers
    r'(.)\1{4,}',     # Repeated characters
    r'^[^a-zA-Z0-9\s]+$'  # Only special characters
]

# Action keywords for intent recognition
ACTION_KEYWORDS = ["download", "save", "share", "send", "forward", "bhej", "install", "upload", "record"]

# Expanded Hinglish mappings
HINGLISH_MAPPINGS = {
    "kese": "how",
    "kaise": "how",
    "karu": "do",
    "karo": "do",
    "bhej": "share",
    "share": "share",
    "save": "save",
    "download": "save",
    "kyu": "why",
    "etna": "so",
    "hai": "is",
    "nahi": "not",
    "ho": "be",
    "tha": "was",
    "mein": "in",
    "par": "on",
    "se": "from",
    "ko": "to"
}

def normalize_and_tokenize_query(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^\w\s\u0900-\u097F?!.-]", "", text)
    tokens = [word for word in text.split() if len(word) > 1]
    
    # Map Hinglish terms
    mapped_tokens = [HINGLISH_MAPPINGS.get(token, token) for token in tokens]
    return mapped_tokens

def is_nonsensical_query(text, tokens):
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, text.lower()):
            return True
    
    if len(text.strip()) < 3:
        return True
    
    for verb, invalid_targets in NONSENSE_FILTER.items():
        if verb in tokens and any(word in invalid_targets for word in tokens):
            return True
    
    if len(tokens) == 0:
        return True
        
    return False

def detect_language_safe(text):
    try:
        if len(text.strip()) < 3:
            return 'en'
        
        text_lower = text.lower()
        # Expanded Hinglish keywords
        hinglish_keywords = [
            "kese", "kaise", "karu", "karo", "bhej", "kyu", "hai", "etna", "nahi", 
            "ho", "tha", "mein", "par", "se", "ko", "kya", "abhi", "phir", "wala", 
            "wali", "aur", "ya", "toh"
        ]
        
        # Check for Devanagari characters or Hinglish keywords
        has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
        has_hinglish = any(keyword in text_lower for keyword in hinglish_keywords)
        
        # Check for English words (simple heuristic)
        english_words = sum(1 for word in text_lower.split() if word.isascii() and word.isalpha())
        total_words = len(text_lower.split())
        english_ratio = english_words / total_words if total_words > 0 else 0
        
        # Classify as Hinglish if mixed English and Hindi elements are present
        if (has_devanagari or has_hinglish) and 0.2 <= english_ratio <= 0.8:
            logger.info(f"Detected Hinglish for '{text}'")
            return 'hinglish'
        
        # Fallback to langdetect
        detected = detect(text)
        logger.info(f"Langdetect result for '{text}': {detected}")
        return detected if detected in LANGUAGES else 'en'
    except (LangDetectException, Exception) as e:
        logger.debug(f"Language detection failed for '{text}': {e}")
        return 'en'

def find_best_match(user_query):
    user_keywords = normalize_and_tokenize_query(user_query)
    if not user_keywords:
        return None

    documents = search_index.get('documents', [])
    idf_scores = search_index.get('idf_scores', {})
    
    if not documents:
        logger.warning("No documents in search index")
        return None

    best_score = 0
    best_match_doc = None

    for doc in documents:
        doc_keywords = doc.get('keywords', [])
        if not doc_keywords:
            continue
            
        current_doc_score = 0
        matched_keywords = 0

        for user_word in user_keywords:
            if user_word in doc_keywords:
                keyword_importance = idf_scores.get(user_word, 0.1)
                if user_word in ACTION_KEYWORDS:
                    keyword_importance *= ACTION_WEIGHT
                current_doc_score += keyword_importance
                matched_keywords += 1
            else:
                best_match = process.extractOne(user_word, doc_keywords, scorer=fuzz.ratio)
                if best_match and best_match[1] >= FUZZY_MATCH_THRESHOLD:
                    matched_keyword = best_match[0]
                    keyword_importance = idf_scores.get(matched_keyword, 0.05)
                    if matched_keyword in ACTION_KEYWORDS:
                        keyword_importance *= ACTION_WEIGHT
                    fuzzy_score = (best_match[1] / 100) * keyword_importance
                    current_doc_score += fuzzy_score
                    matched_keywords += 1

        if matched_keywords > 0:
            current_doc_score *= (1 + (matched_keywords - 1) * 0.2)

        if current_doc_score > best_score:
            best_score = current_doc_score
            best_match_doc = doc

    if best_match_doc and best_score >= MINIMUM_SCORE_THRESHOLD:
        logger.info(f"Found match for '{user_query}' with score {best_score:.3f}")
        return best_match_doc
    else:
        logger.info(f"No good match found for '{user_query}' (best score: {best_score:.3f})")
        return None

# --- HTML Template (unchanged) ---
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
        
        .chat-header h1 {
            font-size: 1.1rem;
            font-weight: 500;
        }
        
        .chat-box {
            flex-grow: 1;
            padding: 20px;
            overflow-y: auto;
            background-color: #e5ddd5;
            background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 40 40"><circle cx="20" cy="20" r="1" fill="%23000" opacity="0.02"/></svg>');
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
        
        .bot .message-content {
            background-color: #ffffff;
            color: #303030;
            border-top-left-radius: 0;
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
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1.0); }
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
        
        #send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading { animation: spin 1s linear infinite; }
        
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        @media (max-width: 480px) {
            body { padding: 10px; }
            .chat-container { height: 95vh; border-radius: 0; }
            .message { max-width: 85%; }
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
                <div class="message-content">Hello! I'm here to help with WhatsApp Status Saver questions. Ask me anything about downloading or saving WhatsApp statuses!</div>
            </div>
        </div>
        <div class="input-area">
            <input type="text" id="user-input" placeholder="Ask about WhatsApp status saving..." autocomplete="off" maxlength="200">
            <button id="send-btn">➤</button>
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
            messageContent.textContent = content;
            
            messageDiv.appendChild(messageContent);
            chatBox.appendChild(messageDiv);
            
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

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }

                const data = await response.json();
                hideTypingIndicator();
                
                setTimeout(() => {
                    addMessage(data.response || 'Sorry, no response received.', 'bot');
                }, 300);

            } catch (error) {
                hideTypingIndicator();
                console.error('Error:', error);
                addMessage('Sorry, something went wrong. Please try again later.', 'bot');
            } finally {
                isProcessing = false;
                sendBtn.disabled = false;
                sendBtn.innerHTML = '➤';
                sendBtn.classList.remove('loading');
                userInput.focus();
            }
        }

        sendBtn.addEventListener('click', handleSendMessage);
        
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
            }
        });

        window.addEventListener('load', () => {
            userInput.focus();
        });

        userInput.addEventListener('input', (e) => {
            const value = e.target.value;
            if (value.length > 200) {
                e.target.value = value.substring(0, 200);
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
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"response": "Please send a valid message."}), 400

        user_input = data['message']
        
        if not isinstance(user_input, str):
            return jsonify({"response": "Please send a text message."})
        
        user_input = user_input.strip()
        if not user_input:
            return jsonify({"response": "Please enter a question."})
        
        if len(user_input) > 200:
            return jsonify({"response": "Please keep your question under 200 characters."})

        tokens = normalize_and_tokenize_query(user_input)
        
        if is_nonsensical_query(user_input, tokens):
            return jsonify({
                "response": "I can help you with WhatsApp Status Saver questions. Try asking something like 'how to download status' or 'permission error'."
            })

        # Define Hinglish-specific words (excluding English words from HINGLISH_MAPPINGS)
        hinglish_specific_words = {
            "kese", "kaise", "karu", "karo", "bhej", "kyu", "etna", "hai", "nahi",
            "ho", "tha", "mein", "par", "se", "ko", "kya", "abhi", "phir", "wala",
            "wali", "aur", "ya", "toh"
        }

        # Set initial language with detect_language_safe
        reply_lang = detect_language_safe(user_input)

        # Split input into original words for exact matching
        original_tokens = [word.lower() for word in re.split(r'\W+', user_input) if word]
        # Override to 'hinglish' only if Hinglish-specific words or Devanagari characters are present
        if any(token in hinglish_specific_words for token in original_tokens) or re.search(r'[\u0900-\u097F]', user_input):
            reply_lang = 'hinglish'

        best_doc = find_best_match(user_input)

        if best_doc and 'answers' in best_doc:
            response_text = best_doc['answers'].get(reply_lang, best_doc['answers'].get('en', 'Sorry, no answer available in your language.'))
        else:
            fallback_responses = {
                'en': "I couldn't find a specific answer. Try asking about:\n• How to download/save a status\n• How to share a status\n• Permission/storage issues\n• App errors",
                'hi': "मुझे विशिष्ट उत्तर नहीं मिला। पूछने की कोशिश करें:\n• स्टेटस कैसे डाउनलोड/सेव करें\n• स्टेटस कैसे शेयर करें\n• अनुमति/स्टोरेज समस्याएं\n• ऐप त्रुटियां",
                'hinglish': "Mujhe specific answer nahi mila. Try kariye:\n• Status kaise download/save kare\n• Status kaise share kare\n• Permission/storage issues\n• App errors"
            }
            response_text = fallback_responses.get(reply_lang, fallback_responses['en'])

        return jsonify({"response": response_text})

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            "response": "Sorry, I encountered an error. Please try again later."
        }), 500
    
@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "documents_loaded": len(search_index.get('documents', [])),
        "index_file_exists": os.path.exists(INDEX_FILE)
    })

if __name__ == '__main__':
    logger.info("Starting WhatsApp FAQ Bot...")
    logger.info(f"Loaded {len(search_index.get('documents', []))} documents from index")
    app.run(host='0.0.0.0', port=5000, debug=False)  
    

# uses faq_index.json #

# import json
# import re
# from flask import Flask, request, jsonify, render_template_string
# import logging
# from langdetect import detect, DetectorFactory, LangDetectException
# from rapidfuzz import process, fuzz
# import string
# from collections import Counter
# import math

# app = Flask(__name__)

# # --- CONFIGURATION ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# DetectorFactory.seed = 0

# INDEX_FILE = 'faq_index.json'
# # Enhanced language support
# LANGUAGES = ["en", "hi", "id", "hinglish", "es", "fr", "de", "pt", "ar", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"]
# FUZZY_MATCH_THRESHOLD = 75  # Slightly lowered for better matching
# MINIMUM_SCORE_THRESHOLD = 0.3  # Lowered for more flexible matching
# CONTEXT_BONUS_MULTIPLIER = 1.5  # Bonus for contextually relevant matches

# # Language detection patterns for better accuracy
# LANGUAGE_PATTERNS = {
#     'hinglish': [
#         r'\b(hai|hain|kar|kya|kaise|kahan|kyun|matlab|bhai|yaar|abhi|phir|wala|wali)\b',
#         r'\b(app|download|save|status|whatsapp)\b.*\b(hai|kar|kaise|kya)\b'
#     ],
#     'hi': [
#         r'[\u0900-\u097F]+',  # Devanagari script
#         r'\b(क्या|कैसे|कहाँ|क्यों|मतलब|अभी|फिर|वाला|वाली|है|हैं|करना|होना)\b'
#     ],
#     'ar': [r'[\u0600-\u06FF]+'],  # Arabic script
#     'bn': [r'[\u0980-\u09FF]+'],  # Bengali script
#     'ta': [r'[\u0B80-\u0BFF]+'],  # Tamil script
#     'te': [r'[\u0C00-\u0C7F]+'],  # Telugu script
#     'mr': [r'[\u0900-\u097F]+'],  # Marathi (Devanagari)
#     'gu': [r'[\u0A80-\u0AFF]+'],  # Gujarati script
#     'kn': [r'[\u0C80-\u0CFF]+'],  # Kannada script
#     'ml': [r'[\u0D00-\u0D7F]+'],  # Malayalam script
#     'pa': [r'[\u0A00-\u0A7F]+'],  # Punjabi script
#     'ur': [r'[\u0600-\u06FF]+'],  # Urdu (Arabic script)
# }

# # --- Load search index ---
# try:
#     with open(INDEX_FILE, 'r', encoding='utf-8') as f:
#         search_index = json.load(f)
#     logger.info("Search index loaded successfully.")
# except FileNotFoundError:
#     logger.error(f"FATAL: '{INDEX_FILE}' not found. Please run build_index.py first.")
#     search_index = {"documents": [], "idf_scores": {}}

# # --- Enhanced Nonsense Filtering ---
# IRRELEVANT_TOPICS = {
#     "vehicles": {
#         "car", "bike", "truck", "bicycle", "scooter", "train", "aeroplane", "airplane", 
#         "jet", "boat", "ship", "bus", "auto", "rickshaw", "taxi", "metro", "helicopter",
#         "motorcycle", "vehicle", "transport", "driving", "parking"
#     },
#     "appliances": {
#         "fridge", "refrigerator", "oven", "stove", "fan", "bulb", "ac", "cooler", 
#         "mixer", "toaster", "tv", "television", "microwave", "washing", "machine",
#         "dishwasher", "vacuum", "heater", "iron", "blender", "grinder"
#     },
#     "animals": {
#         "dog", "cat", "elephant", "horse", "monkey", "tiger", "lion", "goat", "cow",
#         "bird", "fish", "rabbit", "chicken", "pig", "sheep", "buffalo", "deer",
#         "snake", "lizard", "frog", "insect", "spider", "ant", "bee", "butterfly"
#     },
#     "foods": {
#         "pizza", "burger", "cake", "bread", "milk", "apple", "rice", "chapati", 
#         "noodles", "icecream", "chicken", "mutton", "fish", "egg", "vegetable",
#         "fruit", "sweet", "spicy", "cooking", "recipe", "restaurant", "hotel"
#     },
#     "clothing": {
#         "jeans", "shirt", "jacket", "tshirt", "saree", "kurta", "shoes", "cap",
#         "dress", "pants", "socks", "underwear", "bra", "belt", "watch", "jewelry",
#         "ring", "necklace", "earring", "bag", "purse", "wallet"
#     },
#     "entertainment": {
#         "movie", "film", "song", "music", "dance", "party", "game", "cricket",
#         "football", "tennis", "sport", "tv", "serial", "drama", "comedy",
#         "actor", "actress", "singer", "musician", "celebrity", "star"
#     },
#     "education": {
#         "school", "college", "university", "exam", "test", "study", "book",
#         "teacher", "student", "homework", "assignment", "project", "degree",
#         "course", "subject", "math", "science", "history", "geography"
#     },
#     "body_parts": {
#         "head", "hair", "eye", "nose", "mouth", "ear", "hand", "finger", "leg",
#         "foot", "back", "stomach", "chest", "arm", "shoulder", "knee", "elbow",
#         "face", "skin", "tooth", "teeth", "tongue", "nail"
#     },
#     "nature": {
#         "tree", "flower", "plant", "garden", "park", "mountain", "river", "sea",
#         "ocean", "sky", "cloud", "rain", "sun", "moon", "star", "wind", "fire",
#         "water", "earth", "stone", "rock", "sand", "grass", "leaf"
#     },
#     "other": {
#         "road", "girl", "boy", "kid", "pen", "bottle", "house", "wall", "door",
#         "window", "chair", "table", "bed", "room", "kitchen", "bathroom", "office",
#         "market", "shop", "money", "rupee", "dollar", "bank", "atm", "card"
#     }
# }

# # Enhanced nonsense filter with more comprehensive coverage
# NONSENSE_FILTER = {
#     "download": set.union(*IRRELEVANT_TOPICS.values()),
#     "save": set.union(*IRRELEVANT_TOPICS.values()),
#     "install": set.union(*IRRELEVANT_TOPICS.values()),
#     "upload": set.union(*IRRELEVANT_TOPICS.values()),
#     "record": set.union(*IRRELEVANT_TOPICS.values()),
#     "create": set.union(*IRRELEVANT_TOPICS.values()),
#     "make": set.union(*IRRELEVANT_TOPICS.values()),
#     "build": set.union(*IRRELEVANT_TOPICS.values()),
#     "get": set.union(*IRRELEVANT_TOPICS.values()),
#     "buy": set.union(*IRRELEVANT_TOPICS.values()),
#     "sell": set.union(*IRRELEVANT_TOPICS.values()),
#     "find": set.union(*IRRELEVANT_TOPICS.values()),
#     "search": set.union(*IRRELEVANT_TOPICS.values()),
# }

# # Context keywords that should boost relevance scores
# WHATSAPP_CONTEXT_KEYWORDS = {
#     "whatsapp", "status", "saver", "download", "save", "story", "media", "video", 
#     "photo", "image", "chat", "message", "contact", "group", "call", "notification",
#     "backup", "restore", "privacy", "block", "unblock", "mute", "archive", "delete",
#     "forward", "reply", "share", "send", "receive", "online", "offline", "last seen"
# }

# def enhanced_language_detection(text):
#     """Enhanced language detection with pattern matching and fallback"""
#     text_lower = text.lower()
    
#     # First try pattern-based detection for specific languages
#     for lang, patterns in LANGUAGE_PATTERNS.items():
#         for pattern in patterns:
#             if re.search(pattern, text, re.IGNORECASE):
#                 logger.info(f"Pattern-based detection: {lang}")
#                 return lang
    
#     # Try langdetect library
#     try:
#         detected_lang = detect(text)
#         if detected_lang in LANGUAGES:
#             logger.info(f"Langdetect detection: {detected_lang}")
#             return detected_lang
#     except LangDetectException:
#         pass
    
#     # Check for mixed language indicators (Hinglish)
#     english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
#     total_words = len(text.split())
#     if total_words > 0 and 0.3 <= (english_words / total_words) <= 0.8:
#         # Check for Hindi/Devanagari characters
#         if re.search(r'[\u0900-\u097F]', text) or \
#            re.search(r'\b(hai|hain|kar|kya|kaise|kahan|kyun|matlab|bhai|yaar)\b', text_lower):
#             logger.info("Mixed language detection: hinglish")
#             return 'hinglish'
    
#     # Default fallback
#     logger.info("Defaulting to English")
#     return 'en'

# def normalize_and_tokenize_query(text):
#     """Enhanced tokenization with better handling of special characters"""
#     if not isinstance(text, str): 
#         return []
    
#     text = text.lower()
#     # Preserve important punctuation and Unicode characters
#     text = re.sub(r"[^\w\s\u0900-\u097F\u0600-\u06FF\u0980-\u09FF\u0B80-\u0BFF\u0C00-\u0C7F\u0A80-\u0AFF\u0C80-\u0CFF\u0D00-\u0D7F\u0A00-\u0A7F]", " ", text)
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     tokens = text.split()
#     # Remove very short tokens (less than 2 characters) unless they're important
#     important_short_words = {'id', 'hi', 'ok', 'no', 'go', 'do', 'me', 'my', 'we', 'up'}
#     tokens = [token for token in tokens if len(token) >= 2 or token in important_short_words]
    
#     return tokens

# def is_nonsensical_query(tokens):
#     """Enhanced nonsense detection with context awareness"""
#     if not tokens:
#         return True
    
#     # Check for too short queries
#     if len(tokens) == 1 and tokens[0] not in WHATSAPP_CONTEXT_KEYWORDS:
#         return True
    
#     # Check for nonsensical combinations
#     for verb, invalid_targets in NONSENSE_FILTER.items():
#         if verb in tokens:
#             invalid_count = sum(1 for word in tokens if word in invalid_targets)
#             # If more than half the query consists of irrelevant terms
#             if invalid_count > len(tokens) // 2:
#                 return True
    
#     # Check for completely irrelevant queries
#     irrelevant_count = 0
#     all_irrelevant = set.union(*IRRELEVANT_TOPICS.values())
#     for token in tokens:
#         if token in all_irrelevant:
#             irrelevant_count += 1
    
#     # If more than 70% of tokens are irrelevant and no WhatsApp context
#     if irrelevant_count > len(tokens) * 0.7:
#         has_context = any(token in WHATSAPP_CONTEXT_KEYWORDS for token in tokens)
#         if not has_context:
#             return True
    
#     return False

# def calculate_enhanced_score(user_keywords, doc_keywords, idf_scores):
#     """Enhanced scoring algorithm with multiple factors"""
#     if not user_keywords or not doc_keywords:
#         return 0
    
#     total_score = 0
#     matched_keywords = set()
#     context_bonus = 0
    
#     # Primary fuzzy matching score
#     for user_word in user_keywords:
#         best_match = process.extractOne(user_word, doc_keywords, scorer=fuzz.ratio)
#         if best_match and best_match[1] >= FUZZY_MATCH_THRESHOLD:
#             matched_keyword = best_match[0]
#             matched_keywords.add(matched_keyword)
            
#             # Base similarity score
#             similarity_score = best_match[1] / 100
            
#             # IDF importance weight
#             keyword_importance = idf_scores.get(matched_keyword, 0.05)
            
#             # Context relevance bonus
#             if user_word in WHATSAPP_CONTEXT_KEYWORDS:
#                 context_bonus += 0.2
            
#             # Length penalty for very short matches
#             length_factor = min(len(user_word) / 4, 1.0)
            
#             word_score = similarity_score * keyword_importance * length_factor
#             total_score += word_score
    
#     # Coverage bonus - reward matching more unique keywords
#     coverage_ratio = len(matched_keywords) / len(user_keywords) if user_keywords else 0
#     coverage_bonus = coverage_ratio * 0.3
    
#     # Context bonus for WhatsApp-related queries
#     final_score = (total_score + coverage_bonus + context_bonus)
    
#     # Apply context multiplier if query is clearly WhatsApp-related
#     whatsapp_keywords_found = sum(1 for kw in user_keywords if kw in WHATSAPP_CONTEXT_KEYWORDS)
#     if whatsapp_keywords_found >= 1:
#         final_score *= CONTEXT_BONUS_MULTIPLIER
    
#     return final_score

# def find_best_match(user_query):
#     """Enhanced matching with improved scoring"""
#     user_keywords = normalize_and_tokenize_query(user_query)
#     if not user_keywords:
#         return None

#     documents = search_index.get('documents', [])
#     idf_scores = search_index.get('idf_scores', {})

#     best_score = 0
#     best_match_doc = None
    
#     # Store all matches for potential ensemble scoring
#     matches = []

#     for doc in documents:
#         doc_keywords = doc['keywords']
#         current_doc_score = calculate_enhanced_score(user_keywords, doc_keywords, idf_scores)
        
#         matches.append((doc, current_doc_score))
        
#         if current_doc_score > best_score:
#             best_score = current_doc_score
#             best_match_doc = doc

#     # Log top matches for debugging
#     matches.sort(key=lambda x: x[1], reverse=True)
#     logger.info(f"Top 3 matches for '{user_query}':")
#     for i, (doc, score) in enumerate(matches[:3]):
#         logger.info(f"  {i+1}. Score: {score:.3f} - {doc.get('id', 'Unknown')}")

#     if best_match_doc and best_score >= MINIMUM_SCORE_THRESHOLD:
#         logger.info(f"Selected match for '{user_query}' with score {best_score:.3f}")
#         return best_match_doc
#     else:
#         logger.warning(f"No good match found for '{user_query}' (best score: {best_score:.3f})")
#         return None

# def get_fallback_response(lang):
#     """Enhanced fallback responses in multiple languages"""
#     fallback_responses = {
#         'en': "I'm sorry, I couldn't understand that. Try asking something about WhatsApp Status Saver, like 'how to save status' or 'download WhatsApp videos'.",
#         'hi': "माफ करें, मैं समझ नहीं पाया। WhatsApp Status Saver के बारे में कुछ पूछिए जैसे 'स्टेटस कैसे सेव करें' या 'WhatsApp वीडियो डाउनलोड करें'।",
#         'hinglish': "Sorry, main samajh nahi paya. WhatsApp Status Saver ke bare me kuch pucho jaise 'status kaise save kare' ya 'WhatsApp video download kare'.",
#         'es': "Lo siento, no pude entender eso. Prueba preguntando algo sobre WhatsApp Status Saver, como 'cómo guardar estado' o 'descargar videos de WhatsApp'.",
#         'fr': "Désolé, je n'ai pas pu comprendre cela. Essayez de demander quelque chose sur WhatsApp Status Saver, comme 'comment sauvegarder le statut' ou 'télécharger des vidéos WhatsApp'.",
#         'ar': "آسف، لم أستطع فهم ذلك. جرب أن تسأل شيئًا عن WhatsApp Status Saver، مثل 'كيفية حفظ الحالة' أو 'تحميل فيديوهات WhatsApp'.",
#         'pt': "Desculpe, não consegui entender isso. Tente perguntar algo sobre WhatsApp Status Saver, como 'como salvar status' ou 'baixar vídeos do WhatsApp'.",
#     }
#     return fallback_responses.get(lang, fallback_responses['en'])

# def get_nonsense_response(lang):
#     """Enhanced nonsense responses in multiple languages"""
#     nonsense_responses = {
#         'en': "That doesn't seem related to WhatsApp or Status Saver. Please ask something relevant like 'how to save a status' or 'download WhatsApp media'.",
#         'hi': "यह WhatsApp या Status Saver से संबंधित नहीं लगता। कृपया कोई प्रासंगिक प्रश्न पूछें जैसे 'स्टेटस कैसे सेव करें' या 'WhatsApp मीडिया डाउनलोड करें'।",
#         'hinglish': "Ye WhatsApp ya Status Saver se related nahi lagta. Please koi relevant question pucho jaise 'status kaise save kare' ya 'WhatsApp media download kare'.",
#         'es': "Eso no parece estar relacionado con WhatsApp o Status Saver. Por favor pregunta algo relevante como 'cómo guardar un estado' o 'descargar medios de WhatsApp'.",
#         'fr': "Cela ne semble pas lié à WhatsApp ou Status Saver. Veuillez poser une question pertinente comme 'comment sauvegarder un statut' ou 'télécharger des médias WhatsApp'.",
#         'ar': "هذا لا يبدو متعلقًا بـ WhatsApp أو Status Saver. يرجى طرح سؤال ذي صلة مثل 'كيفية حفظ حالة' أو 'تحميل وسائط WhatsApp'.",
#         'pt': "Isso não parece estar relacionado ao WhatsApp ou Status Saver. Por favor, faça uma pergunta relevante como 'como salvar um status' ou 'baixar mídia do WhatsApp'.",
#     }
#     return nonsense_responses.get(lang, nonsense_responses['en'])

# # --- HTML Template (Enhanced) ---
# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>WhatsApp Status Saver - FAQ Bot</title>
#     <style>
#         body {
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             margin: 0;
#             padding: 20px;
#             background: linear-gradient(135deg, #25D366, #128C7E);
#             min-height: 100vh;
#         }
#         .container {
#             max-width: 800px;
#             margin: 0 auto;
#             background: white;
#             border-radius: 15px;
#             box-shadow: 0 10px 30px rgba(0,0,0,0.2);
#             overflow: hidden;
#         }
#         .header {
#             background: #075E54;
#             color: white;
#             padding: 20px;
#             text-align: center;
#         }
#         .chat-container {
#             height: 400px;
#             overflow-y: auto;
#             padding: 20px;
#             background: #f0f0f0;
#         }
#         .message {
#             margin: 10px 0;
#             padding: 10px 15px;
#             border-radius: 10px;
#             max-width: 80%;
#         }
#         .user-message {
#             background: #DCF8C6;
#             margin-left: auto;
#             text-align: right;
#         }
#         .bot-message {
#             background: white;
#             border: 1px solid #ddd;
#         }
#         .input-container {
#             display: flex;
#             padding: 20px;
#             background: white;
#         }
#         .input-container input {
#             flex: 1;
#             padding: 12px;
#             border: 2px solid #25D366;
#             border-radius: 25px;
#             outline: none;
#             font-size: 16px;
#         }
#         .input-container button {
#             background: #25D366;
#             color: white;
#             border: none;
#             padding: 12px 20px;
#             margin-left: 10px;
#             border-radius: 25px;
#             cursor: pointer;
#             font-size: 16px;
#         }
#         .input-container button:hover {
#             background: #1ea851;
#         }
#         .language-info {
#             text-align: center;
#             padding: 10px;
#             font-size: 12px;
#             color: #666;
#             background: #f9f9f9;
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <div class="header">
#             <h1>🗨️ WhatsApp Status Saver FAQ Bot</h1>
#             <p>Ask me anything about WhatsApp Status Saver!</p>
#         </div>
#         <div class="language-info">
#             Supported languages: English, Hindi, Hinglish, Spanish, French, German, Portuguese, Arabic, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Oriya, Assamese, Urdu
#         </div>
#         <div class="chat-container" id="chatContainer">
#             <div class="message bot-message">
#                 Hello! I'm here to help you with WhatsApp Status Saver. You can ask me questions in multiple languages!
#             </div>
#         </div>
#         <div class="input-container">
#             <input type="text" id="messageInput" placeholder="Type your question here..." onkeypress="handleKeyPress(event)">
#             <button onclick="sendMessage()">Send</button>
#         </div>
#     </div>

#     <script>
#         function handleKeyPress(event) {
#             if (event.key === 'Enter') {
#                 sendMessage();
#             }
#         }

#         function sendMessage() {
#             const input = document.getElementById('messageInput');
#             const message = input.value.trim();
#             if (!message) return;

#             const chatContainer = document.getElementById('chatContainer');
            
#             // Add user message
#             const userDiv = document.createElement('div');
#             userDiv.className = 'message user-message';
#             userDiv.textContent = message;
#             chatContainer.appendChild(userDiv);

#             // Clear input
#             input.value = '';

#             // Send to server
#             fetch('/chat', {
#                 method: 'POST',
#                 headers: {
#                     'Content-Type': 'application/json',
#                 },
#                 body: JSON.stringify({message: message})
#             })
#             .then(response => response.json())
#             .then(data => {
#                 const botDiv = document.createElement('div');
#                 botDiv.className = 'message bot-message';
#                 botDiv.textContent = data.response;
#                 chatContainer.appendChild(botDiv);
#                 chatContainer.scrollTop = chatContainer.scrollHeight;
#             })
#             .catch(error => {
#                 const errorDiv = document.createElement('div');
#                 errorDiv.className = 'message bot-message';
#                 errorDiv.textContent = 'Sorry, there was an error processing your request.';
#                 chatContainer.appendChild(errorDiv);
#                 chatContainer.scrollTop = chatContainer.scrollHeight;
#             });

#             chatContainer.scrollTop = chatContainer.scrollHeight;
#         }
#     </script>
# </body>
# </html>
# """

# @app.route('/')
# def home():
#     return render_template_string(HTML_TEMPLATE)

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get('message', '').strip()
#     if not user_input:
#         return jsonify({"response": "Please enter a question."})

#     # Normalize and tokenize
#     tokens = normalize_and_tokenize_query(user_input)

#     # Enhanced language detection
#     try:
#         reply_lang = enhanced_language_detection(user_input)
#     except Exception as e:
#         logger.warning(f"Language detection error: {e}")
#         reply_lang = 'en'

#     # Check for nonsensical queries
#     if is_nonsensical_query(tokens):
#         response_text = get_nonsense_response(reply_lang)
#         return jsonify({"response": response_text, "detected_language": reply_lang})

#     # Find best matching document
#     best_doc = find_best_match(user_input)

#     if best_doc:
#         # Get response in detected language with fallback
#         response_text = best_doc['answers'].get(reply_lang)
#         if not response_text:
#             # Fallback to English if detected language not available
#             response_text = best_doc['answers'].get('en', 'Sorry, no answer available.')
        
#         logger.info(f"Responding in language: {reply_lang}")
#     else:
#         response_text = get_fallback_response(reply_lang)

#     return jsonify({
#         "response": response_text, 
#         "detected_language": reply_lang,
#         "query_tokens": tokens[:5]  # First 5 tokens for debugging
#     })

# if __name__ == '__main__':
#     logger.info("Starting Enhanced WhatsApp FAQ Bot...")
#     logger.info(f"Supported languages: {', '.join(LANGUAGES)}")
#     logger.info(f"Fuzzy match threshold: {FUZZY_MATCH_THRESHOLD}")
#     logger.info(f"Minimum score threshold: {MINIMUM_SCORE_THRESHOLD}")
#     app.run(host='0.0.0.0', port=5000, debug=True)


# import json
# import re
# from flask import Flask, request, jsonify, render_template_string
# import logging
# from langdetect import detect, DetectorFactory
# from rapidfuzz import process, fuzz

# app = Flask(__name__)

# # --- CONFIGURATION ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# DetectorFactory.seed = 0

# INDEX_FILE = 'faq_index.json'
# LANGUAGES = ["en", "hi", "id"]
# FUZZY_MATCH_THRESHOLD = 80 # A good starting point for typo tolerance
# MINIMUM_SCORE_THRESHOLD = 0.25 # Threshold for a match to be considered valid

# # --- LOAD THE UNIFIED SEARCH INDEX ---
# try:
#     with open(INDEX_FILE, 'r', encoding='utf-8') as f:
#         search_index = json.load(f)
#     logger.info("Final search index loaded successfully.")
# except FileNotFoundError:
#     logger.error(f"FATAL: '{INDEX_FILE}' not found. Please run build_index.py first.")
#     # For demo purposes, create a simple fallback
#     search_index = {
#         'documents': [
#             {
#                 'id': 'demo_1',
#                 'keywords': ['hello', 'hi', 'greeting'],
#                 'answers': {
#                     'en': 'Hello! How can I help you today?',
#                     'hi': 'नमस्ते! मैं आपकी कैसे सहायता कर सकता हूं?',
#                     'id': 'Halo! Bagaimana saya bisa membantu Anda hari ini?'
#                 }
#             }
#         ],
#         'idf_scores': {'hello': 0.5, 'hi': 0.5, 'greeting': 0.5}
#     }

# def normalize_and_tokenize_query(text):
#     if not isinstance(text, str): return []
#     text = text.lower()
#     # Remove punctuation
#     text = re.sub(r"[^\w\s\u0900-\u097F]", "", text)
#     return text.split()

# def find_best_match(user_query):
#     """Searches the unified index using IDF and fuzzy matching, returning the best document."""
#     user_keywords = normalize_and_tokenize_query(user_query)
#     if not user_keywords:
#         return None

#     documents = search_index.get('documents', [])
#     idf_scores = search_index.get('idf_scores', {})
    
#     best_score = 0
#     best_match_doc = None

#     for doc in documents:
#         doc_keywords = doc['keywords']
#         current_doc_score = 0
        
#         for user_word in user_keywords:
#             # Find the best fuzzy match for the user's word in this document's keywords
#             best_match = process.extractOne(user_word, doc_keywords, scorer=fuzz.ratio)
            
#             if best_match and best_match[1] >= FUZZY_MATCH_THRESHOLD:
#                 matched_keyword = best_match[0]
#                 keyword_importance = idf_scores.get(matched_keyword, 0.05)
#                 current_doc_score += (best_match[1] / 100) * keyword_importance

#         if current_doc_score > best_score:
#             best_score = current_doc_score
#             best_match_doc = doc
            
#     if best_match_doc and best_score >= MINIMUM_SCORE_THRESHOLD:
#         logger.info(f"Found match for '{user_query}' in doc '{best_match_doc['id']}' with score {best_score:.2f}")
#         return best_match_doc
#     else:
#         logger.warning(f"No good match found for query '{user_query}' (Best score: {best_score:.2f})")
#         return None

# # Enhanced HTML Template with authentic WhatsApp styling
# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>WhatsApp FAQ Bot</title>
#     <style>
#         @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600&display=swap');
        
#         * {
#             margin: 0;
#             padding: 0;
#             box-sizing: border-box;
#         }
        
#         body {
#             font-family: 'Roboto', sans-serif;
#             background: #e5ddd5;
#             display: flex;
#             justify-content: center;
#             align-items: center;
#             min-height: 100vh;
#             padding: 20px;
#         }
        
#         .chat-container {
#             width: 100%;
#             max-width: 450px;
#             height: 90vh;
#             max-height: 700px;
#             display: flex;
#             flex-direction: column;
#             background-color: #f0f0f0;
#             border-radius: 8px;
#             box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
#             overflow: hidden;
#         }
        
#         .chat-header {
#             background: linear-gradient(135deg, #075E54 0%, #128C7E 100%);
#             color: white;
#             padding: 15px 20px;
#             font-weight: 500;
#             font-size: 1.1rem;
#             text-align: center;
#             position: relative;
#             box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
#         }
        
#         .chat-header::before {
#             content: '';
#             position: absolute;
#             top: 0;
#             left: 0;
#             right: 0;
#             bottom: 0;
#             background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
#             opacity: 0.3;
#         }
        
#         .chat-header h1 {
#             font-size: 1.1rem;
#             font-weight: 500;
#             position: relative;
#             z-index: 1;
#         }
        
#         .chat-box {
#             flex-grow: 1;
#             padding: 20px;
#             overflow-y: auto;
#             background-color: #e5ddd5;
#             background-image: url('https://user-images.githubusercontent.com/15075759/28719144-86dc0f70-73b1-11e7-911d-60d70fcded21.png');
#             background-repeat: repeat;
#             background-size: auto;
#         }
        
#         .chat-box::-webkit-scrollbar {
#             width: 6px;
#         }
        
#         .chat-box::-webkit-scrollbar-track {
#             background: transparent;
#         }
        
#         .chat-box::-webkit-scrollbar-thumb {
#             background: rgba(0, 0, 0, 0.2);
#             border-radius: 10px;
#         }
        
#         .message {
#             display: flex;
#             margin-bottom: 15px;
#             max-width: 80%;
#             animation: messageSlide 0.3s ease-out;
#         }
        
#         @keyframes messageSlide {
#             from {
#                 opacity: 0;
#                 transform: translateY(10px);
#             }
#             to {
#                 opacity: 1;
#                 transform: translateY(0);
#             }
#         }
        
#         .message-content {
#             padding: 10px 15px;
#             border-radius: 12px;
#             line-height: 1.4;
#             white-space: pre-wrap;
#             word-wrap: break-word;
#             position: relative;
#             box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
#             font-size: 14px;
#         }
        
#         .user {
#             margin-left: auto;
#             flex-direction: row-reverse;
#         }
        
#         .user .message-content {
#             background-color: #dcf8c6;
#             color: #303030;
#             border-top-right-radius: 0;
#         }
        
#         .user .message-content::after {
#             content: '';
#             position: absolute;
#             bottom: 0;
#             right: -8px;
#             width: 0;
#             height: 0;
#             border: 8px solid transparent;
#             border-left-color: #dcf8c6;
#             border-bottom: none;
#             border-right: none;
#         }
        
#         .bot .message-content {
#             background-color: #ffffff;
#             color: #303030;
#             border-top-left-radius: 0;
#         }
        
#         .bot .message-content::after {
#             content: '';
#             position: absolute;
#             bottom: 0;
#             left: -8px;
#             width: 0;
#             height: 0;
#             border: 8px solid transparent;
#             border-right-color: #ffffff;
#             border-bottom: none;
#             border-left: none;
#         }
        
#         .typing-indicator {
#             display: none;
#             padding: 10px 15px;
#             background-color: #ffffff;
#             border-radius: 12px;
#             border-top-left-radius: 0;
#             align-items: center;
#             box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
#             animation: messageSlide 0.3s ease-out;
#         }
        
#         .typing-indicator::after {
#             content: '';
#             position: absolute;
#             bottom: 0;
#             left: -8px;
#             width: 0;
#             height: 0;
#             border: 8px solid transparent;
#             border-right-color: #ffffff;
#             border-bottom: none;
#             border-left: none;
#         }
        
#         .typing-indicator span {
#             height: 8px;
#             width: 8px;
#             background-color: #999;
#             border-radius: 50%;
#             display: inline-block;
#             margin: 0 2px;
#             animation: bounce 1.3s infinite ease-in-out;
#         }
        
#         .typing-indicator span:nth-child(2) { animation-delay: -1.1s; }
#         .typing-indicator span:nth-child(3) { animation-delay: -0.9s; }
        
#         @keyframes bounce {
#             0%, 80%, 100% {
#                 transform: scale(0);
#             }
#             40% {
#                 transform: scale(1.0);
#             }
#         }
        
#         .input-area {
#             display: flex;
#             padding: 10px;
#             background-color: #f0f0f0;
#             border-top: 1px solid #ddd;
#             gap: 10px;
#         }
        
#         #user-input {
#             flex-grow: 1;
#             border: none;
#             padding: 12px 15px;
#             border-radius: 25px;
#             font-size: 1rem;
#             font-family: 'Roboto', sans-serif;
#             background-color: white;
#             transition: all 0.2s ease;
#         }
        
#         #user-input:focus {
#             outline: none;
#             box-shadow: 0 0 0 2px rgba(37, 211, 102, 0.3);
#         }
        
#         #user-input::placeholder {
#             color: #999;
#         }
        
#         #send-btn {
#             background: linear-gradient(135deg, #128C7E 0%, #075E54 100%);
#             color: white;
#             border: none;
#             border-radius: 50%;
#             width: 50px;
#             height: 50px;
#             cursor: pointer;
#             font-size: 24px;
#             display: flex;
#             justify-content: center;
#             align-items: center;
#             transition: all 0.2s ease;
#             box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
#         }
        
#         #send-btn:hover {
#             background: linear-gradient(135deg, #075E54 0%, #128C7E 100%);
#             transform: translateY(-1px);
#             box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
#         }
        
#         #send-btn:active {
#             transform: translateY(0);
#         }
        
#         #send-btn:disabled {
#             opacity: 0.6;
#             cursor: not-allowed;
#             transform: none;
#         }
        
#         /* Mobile responsiveness */
#         @media (max-width: 480px) {
#             body {
#                 padding: 10px;
#             }
            
#             .chat-container {
#                 height: 95vh;
#                 border-radius: 0;
#             }
            
#             .chat-header {
#                 padding: 12px 16px;
#             }
            
#             .chat-box {
#                 padding: 16px;
#             }
            
#             .input-area {
#                 padding: 8px;
#             }
            
#             .message {
#                 max-width: 85%;
#             }
#         }
        
#         /* Loading animation for send button */
#         .loading {
#             animation: spin 1s linear infinite;
#         }
        
#         @keyframes spin {
#             from { transform: rotate(0deg); }
#             to { transform: rotate(360deg); }
#         }
#     </style>
# </head>
# <body>
#     <div class="chat-container">
#         <div class="chat-header">
#             <h1>WhatsApp FAQ Bot</h1>
#         </div>
#         <div class="chat-box" id="chat-box">
#             <div class="message bot">
#                 <div class="message-content">Hello! How can I assist you today?</div>
#             </div>
#         </div>
#         <div class="input-area">
#             <input type="text" id="user-input" placeholder="Type your question..." autocomplete="off">
#             <button id="send-btn">&#10148;</button>
#         </div>
#     </div>

#     <script>
#         const chatBox = document.getElementById('chat-box');
#         const userInput = document.getElementById('user-input');
#         const sendBtn = document.getElementById('send-btn');
#         let isProcessing = false;

#         function addMessage(content, type) {
#             const messageDiv = document.createElement('div');
#             messageDiv.classList.add('message', type);
            
#             const messageContent = document.createElement('div');
#             messageContent.classList.add('message-content');
#             messageContent.innerHTML = content.replace(/\\n/g, '<br>');
            
#             messageDiv.appendChild(messageContent);
#             chatBox.appendChild(messageDiv);
            
#             // Smooth scroll to bottom
#             setTimeout(() => {
#                 chatBox.scrollTop = chatBox.scrollHeight;
#             }, 100);
#         }

#         function showTypingIndicator() {
#             const typingDiv = document.createElement('div');
#             typingDiv.classList.add('message', 'bot');
#             typingDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
#             chatBox.appendChild(typingDiv);
            
#             const indicator = typingDiv.querySelector('.typing-indicator');
#             indicator.style.display = 'flex';
            
#             setTimeout(() => {
#                 chatBox.scrollTop = chatBox.scrollHeight;
#             }, 100);
#         }

#         function hideTypingIndicator() {
#             const indicator = document.querySelector('.typing-indicator');
#             if (indicator) {
#                 indicator.closest('.message').remove();
#             }
#         }

#         async function handleSendMessage() {
#             const message = userInput.value.trim();
#             if (!message || isProcessing) return;

#             isProcessing = true;
#             sendBtn.disabled = true;
#             sendBtn.innerHTML = '⟳';
#             sendBtn.classList.add('loading');

#             addMessage(message, 'user');
#             userInput.value = '';

#             showTypingIndicator();

#             // Simulate network delay
#             await new Promise(resolve => setTimeout(resolve, 400));

#             try {
#                 const response = await fetch('/chat', {
#                     method: 'POST',
#                     headers: { 'Content-Type': 'application/json' },
#                     body: JSON.stringify({ message: message })
#                 });

#                 const data = await response.json();
#                 hideTypingIndicator();
                
#                 // Add a small delay for better UX
#                 setTimeout(() => {
#                     addMessage(data.response, 'bot');
#                 }, 200);

#             } catch (error) {
#                 hideTypingIndicator();
#                 addMessage('Sorry, something went wrong. Please try again.', 'bot');
#                 console.error('Error:', error);
#             } finally {
#                 isProcessing = false;
#                 sendBtn.disabled = false;
#                 sendBtn.innerHTML = '&#10148;';
#                 sendBtn.classList.remove('loading');
#                 userInput.focus();
#             }
#         }

#         sendBtn.addEventListener('click', handleSendMessage);
        
#         userInput.addEventListener('keypress', (e) => {
#             if (e.key === 'Enter') {
#                 e.preventDefault();
#                 handleSendMessage();
#             }
#         });

#         // Auto-focus input on load
#         window.addEventListener('load', () => {
#             userInput.focus();
#         });

#         // Add some interactive feedback
#         userInput.addEventListener('input', () => {
#             if (userInput.value.trim()) {
#                 sendBtn.style.opacity = '1';
#             } else {
#                 sendBtn.style.opacity = '0.7';
#             }
#         });
#     </script>
# </body>
# </html>
# """

# @app.route('/')
# def home():
#     return render_template_string(HTML_TEMPLATE)

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get('message', '').strip().lower()
#     if not user_input:
#         return jsonify({"response": "Please enter a question."})

#     # Handle greetings
#     if user_input in ["hi", "hello", "hey", "namaste", "hola", "hii", "helo", "hey there", "yo", "hai"]:
#         return jsonify({"response": "Hello! How can I assist you today?"})

#     # Handle bot identity questions
#     if any(phrase in user_input for phrase in ["your role", "who are you", "what are you", "what is your job", "tum kaun ho", "kya tum bot ho", "bot kaun hai"]):
#         return jsonify({"response": "I'm your WhatsApp FAQ Bot — a virtual assistant here to answer your questions about the Status Saver app!"})

#     try:
#         reply_lang = detect(user_input)
#         if reply_lang not in LANGUAGES:
#             reply_lang = 'en'
#     except Exception:
#         reply_lang = 'en'

#     logger.info(f"Detected reply language: '{reply_lang}' for query: '{user_input}'")
#     best_doc = find_best_match(user_input)

#     if best_doc:
#         response_text = best_doc['answers'].get(reply_lang, best_doc['answers']['en'])
#     else:
#         response_text = "I'm sorry, I didn't understand the question. Please rephrase your question and try to be more specific."

#     return jsonify({"response": response_text})


# if __name__ == '__main__':
#     logger.info("Starting WhatsApp FAQ Bot Flask server...")
#     app.run(host='0.0.0.0', port=5000, debug=True)

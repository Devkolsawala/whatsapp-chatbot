Multilingual WhatsApp FAQ Chatbot


A smart, multilingual FAQ chatbot with a WhatsApp-style user interface. This project features a robust Flask backend and an intelligent matching algorithm designed to understand user intent accurately, moving beyond simple keyword searches.

The Problem: 

The initial version of the bot struggled with accurately identifying user intent for closely related topics. For example, a query like "how to download status" would often be misinterpreted as a question about "why is my download failing", leading to incorrect and unhelpful responses. This was due to a simplistic matching algorithm that relied heavily on keyword overlap without considering the user's complete phrasing.


The Solution: Intelligent Intent Matching

This project solves the core problem by implementing a more sophisticated, hybrid matching algorithm in app.py. This new approach ensures higher accuracy by understanding the user's query as a whole phrase rather than just a bag of keywords.

The key improvements are:

Phrase-Based Scoring: The user's query is directly compared against all known questions and their paraphrases from whatsapp_faq_multilingual.json. A high fuzzy match score (using rapidfuzz) indicates a strong intent match.

Weighted Hybrid Algorithm: The final relevance score is a weighted combination of a phrase score (80%) and a keyword score (20%). This model prioritizes clear, well-phrased questions while still providing a fallback for vaguer, keyword-based queries.

Self-Contained Knowledge Base: The bot now directly reads from the whatsapp_faq_multilingual.json file, removing the need for an external, pre-compiled index file. This makes the project easier to understand, modify, and maintain.

Features
Multilingual Support: Natively supports English, Hindi, Hinglish, and Indonesian for both questions and answers.

High-Accuracy Matching: The hybrid phrase-and-keyword algorithm significantly reduces incorrect responses.

Interactive UI: A clean, responsive, and familiar WhatsApp-style chat interface built with HTML, CSS, and JavaScript.

Nonsense Filtering: A basic filter to reject spammy or irrelevant queries.

Easy to Extend: The bot's knowledge can be easily expanded by adding new entries to the whatsapp_faq_multilingual.json file.

REST API: Built on Flask, providing a simple /chat endpoint for easy integration.

Tech Stack

Backend: Python, Flask
NLP/Matching: rapidfuzz, langdetect
Frontend: HTML, CSS, JavaScript (no external frameworks)
Knowledge Base: JSON

import json
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import random
import string
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced data augmentation function
def augment_question(question, num_variants=10):
    replacements = {
        # English
        "how to": ["how can I", "how do I", "what’s the way to", "how i can", "how 2"],
        "download": ["save", "get", "grab", "downlod", "dl", "downlaod"],
        "status": ["story", "statuses", "post", "staus", "stats"],
        "whatsapp": ["WhatsApp", "WA", "Whats App", "watsap", "whastapp"],
        # Hindi
        "कैसे": ["मैं कैसे", "क्या तरीका है", "कैसे करूँ", "कैस", "कैसेे"],
        "डाउनलोड": ["सहेजें", "प्राप्त करें", "पकड़ें", "डाउनलोड्ड", "डाउनलोद"],
        "स्टेटस": ["कहानी", "पोस्ट", "स्थिति", "सटेटस", "स्टटस"],
        "व्हाट्सएप": ["WhatsApp", "WA", "व्हाट्स ऐप", "वाट्सप", "व्हाट्सप"],
        # Indonesian
        "bagaimana": ["bagaimana saya", "apa cara", "cara apa", "bgaimana", "bagmana"],
        "unduh": ["simpan", "dapatkan", "ambil", "undh", "unduhh"],
        "status": ["cerita", "posting", "stori", "sttus", "stats"],
        "whatsapp": ["WhatsApp", "WA", "Whats App", "watsap", "whastapp"]
    }

    def introduce_misspelling(word, prob=0.4):
        if random.random() > prob or len(word) < 3:
            return word
        choice = random.choice(['insert', 'delete', 'swap', 'replace'])
        idx = random.randint(0, len(word)-1)
        if choice == 'insert':
            return word[:idx] + random.choice(string.ascii_letters) + word[idx:]
        elif choice == 'delete' and len(word) > 1:
            return word[:idx] + word[idx+1:]
        elif choice == 'swap' and len(word) > 1:
            idx2 = min(len(word)-1, idx+1)
            return word[:idx] + word[idx2] + word[idx] + word[idx2+1:]
        elif choice == 'replace':
            return word[:idx] + random.choice(string.ascii_letters) + word[idx+1:]
        return word

    def introduce_grammatical_error(sentence, lang, prob=0.4):
        if random.random() > prob:
            return sentence
        words = sentence.split()
        if len(words) < 3:
            return sentence
        if lang == "en":
            choice = random.choice(['omit_article', 'wrong_verb', 'shuffle'])
            if choice == 'omit_article' and any(w in ['a', 'an', 'the'] for w in words):
                words = [w for w in words if w not in ['a', 'an', 'the']]
            elif choice == 'wrong_verb':
                verb_map = {"is": "are", "are": "is", "can": "cans", "download": "downloads"}
                words = [verb_map.get(w, w) for w in words]
            elif choice == 'shuffle':
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
        elif lang == "hi":
            choice = random.choice(['omit_particle', 'wrong_verb'])
            if choice == 'omit_particle' and any(w in ['का', 'के', 'की'] for w in words):
                words = [w for w in words if w not in ['का', 'के', 'की']]
            elif choice == 'wrong_verb':
                verb_map = {"है": "हैं", "हैं": "है", "कर": "करता"}
                words = [verb_map.get(w, w) for w in words]
        elif lang == "id":
            choice = random.choice(['omit_preposition', 'shuffle'])
            if choice == 'omit_preposition' and any(w in ['di', 'ke', 'dari'] for w in words):
                words = [w for w in words if w not in ['di', 'ke', 'dari']]
            elif choice == 'shuffle':
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
        return " ".join(words)

    variants = []
    for _ in range(num_variants):
        augmented = question
        for original, options in replacements.items():
            if original in augmented.lower():
                augmented = augmented.replace(original, np.random.choice(options))
        words = augmented.split()
        words = [introduce_misspelling(w) for w in words]
        augmented = " ".join(words)
        lang = "en" if any(c in augmented for c in string.ascii_letters) else "hi" if any(ord(c) >= 2304 for c in augmented) else "id"
        augmented = introduce_grammatical_error(augmented, lang)
        if augmented != question and augmented not in variants:
            variants.append(augmented)
    return variants

# Load dataset
try:
    with open('whatsapp_faq_multilingual.json', 'r', encoding='utf-8') as f:
        faqs = json.load(f)
except FileNotFoundError:
    logger.error("whatsapp_faq_multilingual.json not found")
    exit(1)

# Process dataset
questions = []
labels = []
languages = ["en", "hi", "id"]
for idx, item in enumerate(faqs):
    for lang in languages:
        question = item["question"][lang]
        questions.append(question)
        labels.append(idx)
        paraphrases = item["paraphrases"][lang]
        for q in paraphrases:
            questions.append(q)
            labels.append(idx)
        aug_qs = augment_question(question, num_variants=10)
        for aug_q in aug_qs:
            questions.append(aug_q)
            labels.append(idx)
        for para in paraphrases:
            aug_qs = augment_question(para, num_variants=10)
            for aug_q in aug_qs:
                questions.append(aug_q)
                labels.append(idx)

logger.info(f"Total samples after augmentation: {len(questions)}")

# Split dataset
train_questions, val_questions, train_labels, val_labels = train_test_split(
    questions, labels, test_size=0.2, random_state=42
)

# Tokenize data
tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
train_encodings = tokenizer(
    train_questions, truncation=True, padding='max_length', max_length=128,
    return_tensors='tf', return_attention_mask=True
)
val_encodings = tokenizer(
    val_questions, truncation=True, padding='max_length', max_length=128,
    return_tensors='tf', return_attention_mask=True
)

# Convert labels to one-hot for CategoricalCrossentropy
num_classes = len(faqs)
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
val_labels_one_hot = tf.keras.utils.to_categorical(val_labels, num_classes=num_classes)

# Prepare datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask']},
    train_labels_one_hot
)).shuffle(1000).batch(16)
val_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': val_encodings['input_ids'], 'attention_mask': val_encodings['attention_mask']},
    val_labels_one_hot
)).batch(16)

# Define model
model = TFBertModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', from_pt=True)
input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="attention_mask")
outputs = model(input_ids, attention_mask=attention_mask)
cls_output = outputs.last_hidden_state[:, 0, :]
cls_output = tf.keras.layers.Dropout(0.1)(cls_output)
classification_output = tf.keras.layers.Dense(num_classes, activation='softmax')(cls_output)
tf_model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=classification_output)

# Compile model with label smoothing
tf_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# Train model with early stopping
logger.info("Fine-tuning model...")
try:
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    history = tf_model.fit(
        train_dataset, validation_data=val_dataset, epochs=20,
        verbose=1, callbacks=[early_stopping]
    )
except Exception as e:
    logger.error(f"Training failed: {e}")
    exit(1)

# Print metrics
logger.info(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
logger.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

# Save fine-tuned model
tf_model.save('finetuned_tinybert_multilingual')

# Convert to TFLite with full integer quantization
logger.info("Converting fine-tuned model to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

def representative_dataset():
    for q in train_questions[:100]:
        inputs = tokenizer(
            q, truncation=True, padding='max_length', max_length=128, return_tensors='tf'
        )
        yield [inputs['input_ids'].numpy().astype(np.int32), inputs['attention_mask'].numpy().astype(np.int32)]

converter.representative_dataset = representative_dataset
try:
    tflite_model = converter.convert()
except Exception as e:
    logger.error(f"TFLite conversion failed: {e}")
    exit(1)

# Save TFLite model
with open('finetuned_tinybert_multilingual.tflite', 'wb') as f:
    f.write(tflite_model)
logger.info("Fine-tuned TFLite model saved as finetuned_tinybert_multilingual.tflite")

# Verify size
model_size_mb = os.path.getsize('finetuned_tinybert_multilingual.tflite') / (1024 * 1024)
logger.info(f"Fine-tuned TFLite model size: {model_size_mb:.2f} MB")
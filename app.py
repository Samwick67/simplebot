from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import os
import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================
# APP SETUP
# ======================

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ======================
# LOAD INTENTS
# ======================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_folder = os.path.join(BASE_DIR, "simplebot", "chatbot_data")

all_intents = []

if os.path.exists(json_folder):
    for file in os.listdir(json_folder):
        if file.endswith(".json"):
            with open(os.path.join(json_folder, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                if "intents" in data:
                    all_intents.extend(data["intents"])

print("TOTAL INTENTS:", len(all_intents))

# ======================
# CLEANER
# ======================

def clean(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text

# ======================
# BUILD TRAINING DATA
# ======================

patterns = []
tags = []
responses = {}

for intent in all_intents:
    tag = intent.get("tag")
    responses[tag] = intent.get("responses", [])

    for pattern in intent.get("patterns", []):
        patterns.append(clean(pattern))
        tags.append(tag)

# ======================
# TRAIN TF-IDF MODEL
# ======================

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

print("MODEL READY ✔")

# ======================
# CHAT ENGINE
# ======================

def get_response(user_input):
    user_input = clean(user_input)

    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)

    index = similarity.argmax()
    score = similarity[0][index]

    if score > 0.3:
        tag = tags[index]
        return random.choice(responses.get(tag, ["I don't understand that yet."]))

    return "I didn't understand that. Try rephrasing."

# ======================
# ROUTES
# ======================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form.get("user_input")

    if not user_input:
        return jsonify({"response": "Empty input"}), 400

    reply = get_response(user_input)

    return jsonify({"response": reply})

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "intents": len(all_intents),
        "patterns": len(patterns)
    })

# ======================
# RUN
# ======================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

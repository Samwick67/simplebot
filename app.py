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

# Try both possible folder names (SAFE FOR LOCAL + RENDER)
possible_paths = [
    os.path.join(BASE_DIR, "chatbot_data"),
    os.path.join(BASE_DIR, "simplebot", "chatbot_data"),
]

json_folder = None

for path in possible_paths:
    if os.path.exists(path):
        json_folder = path
        break

all_intents = []

if json_folder:
    print("📁 Using folder:", json_folder)

    for file in os.listdir(json_folder):
        if file.endswith(".json"):
            file_path = os.path.join(json_folder, file)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    intents = data.get("intents", [])

                    if isinstance(intents, list):
                        all_intents.extend(intents)

            except Exception as e:
                print("❌ Error loading:", file, e)

else:
    print("❌ chatbot_data folder NOT FOUND")

print("TOTAL INTENTS:", len(all_intents))

if len(all_intents) == 0:
    raise Exception("No intents loaded → check folder structure or JSON format")

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

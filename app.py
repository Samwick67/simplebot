from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import random
from textblob import TextBlob
from rapidfuzz import process

# =====================================================
# APP SETUP
# =====================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

CORS(app)

# =====================================================
# LOAD JSON INTENTS (LOCAL FILES)
# =====================================================

all_intents = []

json_folder = os.path.join(os.path.dirname(__file__), "ChatBot", "chatbot_data")

if os.path.exists(json_folder):
    for file in os.listdir(json_folder):
        if file.endswith(".json"):
            file_path = os.path.join(json_folder, file)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                if "intents" in data:
                    all_intents.extend(data["intents"])

print("TOTAL INTENTS:", len(all_intents))

# =====================================================
# BUILD PATTERN LIST
# =====================================================

intent_patterns = []

for intent in all_intents:
    for pattern in intent.get("patterns", []):
        intent_patterns.append(pattern.lower().strip())

# =====================================================
# CHAT ENGINE
# =====================================================

def get_response(user_input):

    # Auto-correct input
    user_input = str(TextBlob(user_input).correct()).lower().strip()

    if not intent_patterns:
        return "No training data loaded."

    best_match = process.extractOne(user_input, intent_patterns)

    if best_match and best_match[1] >= 70:
        matched_text = best_match[0]

        for intent in all_intents:
            if matched_text in [p.lower() for p in intent.get("patterns", [])]:
                return random.choice(intent["responses"])

    return random.choice([
        "I didn't understand that.",
        "Can you rephrase?",
        "I'm still learning.",
        "Try asking differently."
    ])

# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.form.get("user_input")

        if not user_input:
            return jsonify({"response": "Empty input"}), 400

        response = get_response(user_input)

        return jsonify({"response": response})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"response": "Server error"}), 500

# =====================================================
# RUN SERVER (RENDER SAFE)
# =====================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask_cors import CORS
import os
from dotenv import load_dotenv
import gc

# Load environment variables
load_dotenv()

# Flask and CORS setup
app = Flask(__name__)
CORS(app)

# Model configuration
model_name = "distilgpt2"
token = os.getenv("HF_API_TOKEN")

if not token:
    raise Exception("Hugging Face API token not found. Set HF_API_TOKEN in your .env file.")

try:
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
except Exception as e:
    print("Model load error:", e)
    raise Exception("Model could not be loaded. Check name or token.")

def generate_response(query):
    try:
        if not query.strip():
            return "Please provide a valid query."
        if len(query) > 500:
            return "Query is too long. Limit to 500 characters."

        print("Tokenizing input...")
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to("cpu")

        print("Generating output...")
        outputs = model.generate(
            inputs["input_ids"],
            max_length=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        gc.collect()
        return response_text

    except Exception as e:
        print("Error during response:", e)
        gc.collect()
        return "Error generating response."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_query = data.get("query", "")
        if not user_query:
            return jsonify({"response": "Please provide a valid query."})

        ai_response = generate_response(user_query)
        return jsonify({"response": ai_response})

    except Exception as e:
        print("Error in /api/chat:", e)
        return jsonify({"response": "Error processing request."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

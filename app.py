from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask_cors import CORS
import os
from dotenv import load_dotenv
import gc  # For garbage collection to optimize memory usage

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Model configuration
model_name = "EleutherAI/gpt-neo-125M"  # Switch to this smaller model for free-tier memory limits
token = os.getenv("HF_API_TOKEN")  # Fetch token securely from environment variables

if not token:
    raise Exception("Hugging Face API token not found. Please set HF_API_TOKEN in your .env file.")

# Load model and tokenizer
try:
    print("Loading the model... This may take some time.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")  # Force CPU usage
except Exception as e:
    print("Error loading model:", e)
    raise Exception("Model could not be loaded. Check the model name or API token.")

def generate_response(query: str) -> str:
    """
    Generates a response using the Hugging Face model based on the provided query.
    """
    try:
        # Ensure input is valid
        if not query.strip():
            return "Please provide a valid query."
        if len(query) > 500:
            return "Query is too long. Please limit it to 500 characters."

        # Tokenize input with padding and truncation
        print("Tokenizing input...")
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to("cpu")

        # Generate the response
        print("Generating output...")
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=30,  # Reduced output length to save memory
            temperature=0.7,  # Controls creativity
            do_sample=True,  # Sampling-based generation
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode and return the generated response
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print("Generated response:", response_text)
        return response_text

    except Exception as e:
        print("Error during response generation:", e)
        # Perform garbage collection to free memory
        gc.collect()
        return "Sorry, I couldn't generate a response."

@app.route("/")
def index():
    """
    Renders the homepage (index.html) located in the 'templates' folder.
    """
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    API endpoint to handle chat requests.
    Receives JSON with a 'query' key and returns the AI's response.
    """
    try:
        data = request.get_json() or {}
        print("Received data:", data)  # Debug log

        # Extract and validate user query
        user_query = data.get("query", "")
        if not user_query:
            return jsonify({"response": "Please provide a valid query."})

        # Generate and return the AI response
        ai_response = generate_response(user_query)
        print("Generated AI response:", ai_response)  # Debug log
        return jsonify({"response": ai_response})

    except Exception as e:
        print("Error in /api/chat route:", e)
        return jsonify({"response": "An error occurred while processing your request."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
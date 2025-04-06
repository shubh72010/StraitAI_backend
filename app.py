from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize Flask app
app = Flask(__name__)

# Load pre-tuned conversational model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route("/")
def index():
    return render_template("index.html")  # make sure index.html exists in a 'templates' folder

@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.get_json().get("query", "").strip()

    if not user_input:
        return jsonify({"response": "Uhh... you gonna say something or nah?"})

    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    try:
        response_ids = model.generate(
            input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.75
        )

        output = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True).strip()
        return jsonify({"response": output})

    except Exception as e:
        print("Error during generation:", str(e))
        return jsonify({"response": "Whoops, I had a brain fart. Try again!"})

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load model and tokenizer once
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def validate_input(data):
    """Check if the input is valid."""
    user_input = data.get("query", "").strip()
    if not user_input:
        return None, jsonify({"response": "Please provide a valid query."}), 400
    return user_input, None, None


def generate_response(user_input):
    """Generate model response based on input."""
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    response_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.75
    )
    output = tokenizer.decode(
        response_ids[:, input_ids.shape[-1]:][0],
        skip_special_tokens=True
    ).strip()
    return output


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input, error_response, status_code = validate_input(data)
        if error_response:
            return error_response, status_code

        output = generate_response(user_input)
        return jsonify({"response": output})

    except Exception as e:
        print("Error during generation:", str(e))
        return jsonify({"response": "Whoops, I had a brain fart. Try again!"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
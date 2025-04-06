from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load Microsoft DialoGPT-medium model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Store chat history (for one-turn convo, this is fine)
chat_history_ids = None

@app.route("/api/chat", methods=["POST"])
def chat():
    global chat_history_ids

    try:
        data = request.get_json()
        user_input = data.get("query", "").strip()

        if not user_input:
            return jsonify({"response": "Bruh... say something first."}), 400

        # Encode user input
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # Append to history if available
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

        # Generate response
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        # Decode and return
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return jsonify({"response": response})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"response": "Oops, something broke."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
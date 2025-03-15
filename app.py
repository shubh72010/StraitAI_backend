from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Hugging Face model and tokenizer
print("Loading the model... This may take some time.")
model_name = "EleutherAI/gpt-neo-1.3B"  # Open-access Hugging Face model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    API endpoint to generate AI responses.
    """
    data = request.get_json() or {}
    print("Received data:", data)  # Debugging
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"response": "Please provide a query."})

    # Generate a response using Hugging Face
    inputs = tokenizer(user_query, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=100, temperature=0.7, do_sample=True)
    ai_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": ai_response})
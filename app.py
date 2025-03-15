from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Strait-AI Backend is running!"

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query", "")

    # Dummy AI response (Replace this with your AI model response)
    ai_response = f"AI received: {user_query}"

    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)

const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const API_URL = "/api/chat";

function sendMessage() {
    let message = userInput.value.trim();
    if (message === "") return;

    // Display user message
    displayMessage("User", message, "user-message");
    userInput.value = "";

    // Show AI "thinking..." animation
    let thinkingMessage = displayMessage("Strait-AI", "AI is thinking", "bot-message", true);

    // Send to backend
    fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: message })
    })
    .then(response => response.json())
    .then(data => {
        // Remove AI "thinking..." animation
        chatBox.removeChild(thinkingMessage);
        displayMessage("Strait-AI", data.response, "bot-message");
    })
    .catch(error => {
        chatBox.removeChild(thinkingMessage);
        displayMessage("Strait-AI", "Error: Could not connect to AI.", "bot-message");
    });
}

function displayMessage(name, text, type, isThinking = false) {
    let messageDiv = document.createElement("div");
    messageDiv.classList.add("message", "slide-in", type);

    let nameDiv = document.createElement("div");
    nameDiv.classList.add("username");
    nameDiv.textContent = name;

    let textDiv = document.createElement("div");
    textDiv.classList.add("text");
    textDiv.textContent = text;
    
    if (isThinking) {
        textDiv.classList.add("typing");
        animateDots(textDiv);
    }

    messageDiv.appendChild(nameDiv);
    messageDiv.appendChild(textDiv);
    chatBox.appendChild(messageDiv);

    chatBox.scrollTop = chatBox.scrollHeight;
    return messageDiv;
}

// Thinking animation: cycles through ".", "..", "..."
function animateDots(element) {
    let dots = ["AI is thinking.", "AI is thinking..", "AI is thinking..."];
    let count = 0;

    let interval = setInterval(() => {
        element.textContent = dots[count % dots.length];
        count++;
    }, 500);

    element.dataset.interval = interval;
}

// Stop animation when AI responds
function stopAnimation(element) {
    clearInterval(element.dataset.interval);
}

const BACKEND_URL = "https://mumu-8u3h.onrender.com"

async function sendMessage() {
  const input = document.getElementById("user-input");
  const message = input.value.trim();
  if (!message) return;

  appendMessage(message, "user");
  input.value = "";

  appendMessage("MUMU is thinking...", "bot");

  try {
    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const data = await response.json();
    document.querySelectorAll(".bot").pop; // remove "thinking..." text
    appendMessage(data.reply || "MUMU had trouble replying 😅", "bot");
  } catch (error) {
    appendMessage("⚠️ Error connecting to MUMU backend.", "bot");
    console.error(error);
  }
}

function appendMessage(text, sender) {
  const chatBox = document.getElementById("chat-box");
  const msg = document.createElement("div");
  msg.classList.add("message", sender);
  msg.textContent = text;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

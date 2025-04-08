(function () {
    const userId = document.currentScript.getAttribute("data-user-id") || "anon";
  
    const style = document.createElement("style");
    style.innerHTML = `
      #chatbot-bubble {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #007bff;
        color: #fff;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        text-align: center;
        line-height: 60px;
        font-size: 24px;
        cursor: pointer;
        z-index: 9999;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
      }
  
      #chatbot-window {
        position: fixed;
        bottom: 90px;
        right: 20px;
        width: 320px;
        max-height: 500px;
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        font-family: sans-serif;
        display: none;
        flex-direction: column;
        overflow: hidden;
        z-index: 9999;
      }
  
      #chatbot-window.open {
        display: flex;
      }
  
      #chatbot-window.expanded {
        width: 400px;
        max-height: 600px;
      }
  
      .chatbot-header {
        background-color: #007bff;
        color: white;
        padding: 10px 16px;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }
  
      #expand-btn {
        background: none;
        border: none;
        color: white;
        font-size: 16px;
        cursor: pointer;
      }
  
      .chatbot-messages {
        flex: 1;
        padding: 10px;
        background: #f5f5f5;
        overflow-y: auto;
      }
  
      .message {
        padding: 10px 14px;
        margin: 8px 0;
        border-radius: 18px;
        max-width: 80%;
        word-wrap: break-word;
      }
  
      .user-msg {
        background-color: #d1e7ff;
        align-self: flex-end;
        margin-left: auto;
      }
  
      .bot-msg {
        background-color: #e2e2e2;
        align-self: flex-start;
        margin-right: auto;
      }
  
      .typing {
        font-style: italic;
        opacity: 0.7;
      }
  
      .typing::after {
        content: '';
        display: inline-block;
        width: 1em;
        animation: dots 1s steps(3, end) infinite;
      }
  
      @keyframes dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
      }
  
      .chatbot-input-area {
        display: flex;
        padding: 8px;
        background-color: #fff;
        border-top: 1px solid #ddd;
      }
  
      #chatbot-input {
        flex: 1;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 20px;
        outline: none;
      }
  
      #chatbot-send {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 10px 14px;
        margin-left: 6px;
        border-radius: 20px;
        cursor: pointer;
      }
  
      .chatbot-footer {
        text-align: center;
        font-size: 10px;
        color: #888;
        padding: 6px;
        background-color: #f0f0f0;
      }
    `;
    document.head.appendChild(style);
  
    const bubble = document.createElement("div");
    bubble.id = "chatbot-bubble";
    bubble.innerText = "💬";
  
    const chatWindow = document.createElement("div");
    chatWindow.id = "chatbot-window";
    chatWindow.innerHTML = `
      <div class="chatbot-header">
        Asistente
        <button id="expand-btn">⤢</button>
      </div>
      <div id="chatbot-messages" class="chatbot-messages"></div>
      <div class="chatbot-input-area">
        <input id="chatbot-input" type="text" placeholder="Escribe algo..." />
        <button id="chatbot-send">Enviar</button>
      </div>
      <div class="chatbot-footer">Powered by Infermatic</div>
    `;
  
    bubble.onclick = () => {
      chatWindow.classList.toggle("open");
    };
  
    function sendMessage() {
      const input = document.getElementById("chatbot-input");
      const messages = document.getElementById("chatbot-messages");
      const text = input.value.trim();
      if (!text) return;
  
      messages.innerHTML += `<div class="message user-msg">${text}</div>`;
      input.value = "";
  
      const typingMsg = document.createElement("div");
      typingMsg.className = "message bot-msg typing";
      typingMsg.innerText = "Escribiendo";
      messages.appendChild(typingMsg);
      scrollMessages();
  
      fetch("http://localhost:3000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, message: text }),
      })
      .then(res => res.json())
      .then(data => {
        typingMsg.remove();
        const formattedReply = data.reply.replace(/\n/g, "<br>");
        messages.innerHTML += `<div class="message bot-msg">${formattedReply}</div>`;
        scrollMessages();
      });
    }
  
    function scrollMessages() {
      const messages = document.getElementById("chatbot-messages");
      messages.scrollTop = messages.scrollHeight;
    }
  
    document.addEventListener("DOMContentLoaded", () => {
      document.body.appendChild(bubble);
      document.body.appendChild(chatWindow);
      document.getElementById("chatbot-send").onclick = sendMessage;
      document.getElementById("chatbot-input").addEventListener("keypress", function (e) {
        if (e.key === "Enter") sendMessage();
      });
      document.getElementById("expand-btn").onclick = () => {
        chatWindow.classList.toggle("expanded");
      };
    });
  })();
  
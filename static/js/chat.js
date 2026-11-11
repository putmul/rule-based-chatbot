const form = document.querySelector('#chat-form');
const input = document.querySelector('#message-input');
const messages = document.querySelector('#messages');
const status = document.querySelector('#status');

function addMessage(text, type) {
  const element = document.createElement('div');
  element.className = `message ${type}`;
  element.textContent = text;
  messages.appendChild(element);
  messages.scrollTop = messages.scrollHeight;
}

async function sendMessage(message) {
  addMessage(message, 'user');
  input.value = '';
  input.disabled = true;
  status.textContent = 'SiagaBot sedang mengetik...';
  try {
    const response = await fetch('/api/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message}) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Gagal memproses pesan.');
    addMessage(data.response, 'bot');
  } catch (error) { addMessage(error.message, 'bot'); }
  finally { input.disabled = false; input.focus(); status.textContent = ''; }
}

form.addEventListener('submit', (event) => { event.preventDefault(); const message = input.value.trim(); if (message) sendMessage(message); });
document.querySelectorAll('.suggestion').forEach((button) => button.addEventListener('click', () => sendMessage(button.textContent)));

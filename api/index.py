"""
ThinkAI Studio — Vercel Serverless Entrypoint & Modern Web API
Serves:
1. Modern ChatGPT dark theme single-page web UI (HTML5/CSS3/Vanilla JS)
2. Streaming and standard JSON endpoints for Google Gemini API
3. Model configuration, health checks, and connection verification
"""

import os
import sys
import json
from typing import Dict, Any, List

# Ensure repository root is on sys.path for local module imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
load_dotenv()

import requests
from flask import Flask, request, jsonify, Response, stream_with_context
from services.gemini_service import (
    GeminiService,
    RECOMMENDED_MODELS,
    mask_api_key,
    sanitize_error_message,
)

app = Flask(__name__)
app.url_map.strict_slashes = False

# Single-Page Application (SPA) HTML Template
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ThinkAI Studio — AI Assistant</title>
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚡</text></svg>">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-main: #212121;
      --bg-card: #2f2f2f;
      --bg-card-hover: #383838;
      --border-subtle: #424242;
      --text-main: #ececec;
      --text-muted: #b4b4b4;
      --text-dim: #737373;
      --accent: #10a37f;
      --accent-hover: #1a7f64;
      --danger: #ef4444;
      --code-bg: #181818;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', Consolas, monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background-color: var(--bg-main);
      color: var(--text-main);
      font-family: var(--font-sans);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* Top Navigation Header */
    header {
      height: 56px;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      background: rgba(33, 33, 33, 0.95);
      backdrop-filter: blur(10px);
      z-index: 100;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 700;
      font-size: 16px;
      letter-spacing: -0.3px;
    }

    .brand .logo {
      background: linear-gradient(135deg, #10a37f, #0d8b6c);
      width: 28px;
      height: 28px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 15px;
      box-shadow: 0 2px 8px rgba(16, 163, 127, 0.3);
    }

    .nav-controls {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .select-model {
      background: var(--bg-card);
      color: var(--text-main);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 6px 12px;
      font-size: 13px;
      font-weight: 500;
      outline: none;
      cursor: pointer;
      transition: border 0.15s ease;
    }
    .select-model:hover { border-color: #666; }

    .btn {
      background: var(--bg-card);
      color: var(--text-main);
      border: 1px solid var(--border-subtle);
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .btn:hover { background: var(--bg-card-hover); border-color: #666; }
    .btn-primary { background: var(--accent); border-color: var(--accent); color: #fff; }
    .btn-primary:hover { background: var(--accent-hover); border-color: var(--accent-hover); }

    /* Main Chat Container */
    main {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      position: relative;
    }

    #chat-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 24px 16px;
      scroll-behavior: smooth;
    }

    .chat-wrapper {
      max-width: 800px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    /* Welcome Hero View */
    .hero-view {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      min-height: 55vh;
      gap: 12px;
      animation: fadeIn 0.3s ease;
    }

    .hero-view .icon {
      font-size: 44px;
      margin-bottom: 6px;
    }

    .hero-view h1 {
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.5px;
    }

    .hero-view p {
      color: var(--text-muted);
      font-size: 14px;
      max-width: 480px;
    }

    .prompt-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      width: 100%;
      max-width: 720px;
      margin-top: 24px;
    }

    .prompt-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 14px 16px;
      text-align: left;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .prompt-card:hover {
      background: var(--bg-card-hover);
      border-color: #666;
      transform: translateY(-2px);
    }
    .prompt-card .card-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .prompt-card .card-desc {
      font-size: 12px;
      color: var(--text-dim);
      line-height: 1.4;
    }

    /* Chat Messages */
    .msg-row {
      display: flex;
      gap: 14px;
      width: 100%;
      animation: fadeIn 0.2s ease;
    }

    .msg-row.user {
      justify-content: flex-end;
    }

    .msg-avatar {
      width: 30px;
      height: 30px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 15px;
      flex-shrink: 0;
    }

    .msg-avatar.assistant {
      background: var(--accent);
      color: white;
    }

    .msg-bubble {
      max-width: 82%;
      font-size: 14.5px;
      line-height: 1.6;
      word-break: break-word;
    }

    .msg-row.user .msg-bubble {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      padding: 10px 16px;
      border-radius: 18px 18px 4px 18px;
      color: #fff;
    }

    .msg-row.assistant .msg-bubble {
      padding-top: 4px;
      color: var(--text-main);
    }

    /* Code blocks in Assistant messages */
    pre {
      background: var(--code-bg);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 12px;
      margin: 10px 0;
      overflow-x: auto;
      font-family: var(--font-mono);
      font-size: 13px;
      position: relative;
    }

    code {
      font-family: var(--font-mono);
      background: rgba(255, 255, 255, 0.08);
      padding: 2px 5px;
      border-radius: 4px;
      font-size: 12.5px;
    }

    pre code {
      background: transparent;
      padding: 0;
    }

    .copy-btn {
      position: absolute;
      top: 6px;
      right: 6px;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      border-radius: 4px;
      padding: 2px 6px;
      font-size: 11px;
      cursor: pointer;
    }
    .copy-btn:hover { color: #fff; background: #444; }

    /* Typing Dots Animation */
    .typing-dots {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 6px 0;
    }
    .typing-dots span {
      width: 6px;
      height: 6px;
      background: var(--text-muted);
      border-radius: 50%;
      animation: blink 1.2s infinite ease-in-out both;
    }
    .typing-dots span:nth-child(1) { animation-delay: 0s; }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }

    @keyframes blink {
      0%, 80%, 100% { transform: scale(0.6); opacity: 0.3; }
      40% { transform: scale(1); opacity: 1; }
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Bottom Input Container */
    .bottom-dock {
      background: linear-gradient(180deg, transparent 0%, var(--bg-main) 30%);
      padding: 10px 16px 18px 16px;
    }

    .input-box-wrapper {
      max-width: 800px;
      margin: 0 auto;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 22px;
      padding: 8px 14px;
      display: flex;
      align-items: flex-end;
      gap: 10px;
      box-shadow: 0 4px 18px rgba(0, 0, 0, 0.2);
      transition: border 0.15s ease;
    }
    .input-box-wrapper:focus-within {
      border-color: #666;
    }

    textarea#prompt-input {
      flex: 1;
      background: transparent;
      border: none;
      color: var(--text-main);
      font-family: var(--font-sans);
      font-size: 14.5px;
      resize: none;
      outline: none;
      line-height: 1.5;
      max-height: 180px;
      padding: 4px 0;
    }
    textarea#prompt-input::placeholder { color: var(--text-dim); }

    .send-btn {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      background: #424242;
      color: #fff;
      border: none;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.15s ease;
      flex-shrink: 0;
    }
    .send-btn.active {
      background: var(--accent);
    }
    .send-btn:hover.active {
      background: var(--accent-hover);
    }
    .send-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .footer-note {
      text-align: center;
      font-size: 11.5px;
      color: var(--text-dim);
      margin-top: 8px;
    }

    /* Settings Modal */
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.65);
      backdrop-filter: blur(4px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 200;
    }
    .modal {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 16px;
      width: 90%;
      max-width: 480px;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .modal-header h3 { font-size: 17px; font-weight: 600; }
    .close-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 18px;
      cursor: pointer;
    }
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .form-group label {
      font-size: 13px;
      font-weight: 500;
      color: var(--text-muted);
    }
    .form-control {
      background: var(--bg-main);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      color: var(--text-main);
      padding: 8px 12px;
      font-family: var(--font-sans);
      font-size: 13px;
      outline: none;
    }
    .form-control:focus { border-color: var(--accent); }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      padding: 4px 8px;
      border-radius: 6px;
      background: rgba(16, 163, 127, 0.15);
      color: var(--accent);
      border: 1px solid rgba(16, 163, 127, 0.3);
    }
    .status-badge.error {
      background: rgba(239, 68, 68, 0.15);
      color: var(--danger);
      border-color: rgba(239, 68, 68, 0.3);
    }
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div class="brand">
      <div class="logo">⚡</div>
      <span>ThinkAI Studio</span>
    </div>
    <div class="nav-controls">
      <select id="model-select" class="select-model" title="Select Gemini Model">
        <option value="gemini-3.5-flash-lite" selected>Gemini 3.5 Flash-Lite (Fast)</option>
        <option value="gemini-3.8-flash">Gemini 3.8 Flash (Multimodal)</option>
        <option value="gemma-4-31b-it">Gemma 4 31B IT</option>
        <option value="gemini-4-argon">Gemini 4 Argon</option>
      </select>
      <button class="btn" id="new-chat-btn" title="Start a fresh chat">➕ New Chat</button>
      <button class="btn" id="settings-btn" title="Open Settings">⚙️ Settings</button>
    </div>
  </header>

  <!-- Chat Area -->
  <main>
    <div id="chat-scroll">
      <div class="chat-wrapper" id="chat-container">
        <!-- Hero Section shown when no messages -->
        <div class="hero-view" id="hero-section">
          <div class="icon">⚡</div>
          <h1>How can I help you today?</h1>
          <p>Supercharged by Google Gemini API with lightning-fast reasoning and code analysis.</p>
          <div class="prompt-grid">
            <div class="prompt-card" onclick="sendQuickPrompt('Explain microservices vs monolith architecture with trade-offs')">
              <div class="card-title">💻 Code Architecture</div>
              <div class="card-desc">Compare microservices vs monolith architectures with pros & cons.</div>
            </div>
            <div class="prompt-card" onclick="sendQuickPrompt('Compare time and space complexity of Quicksort vs Mergesort')">
              <div class="card-title">⚡ Algorithm Analysis</div>
              <div class="card-desc">Analyze time & memory trade-offs between sorting strategies.</div>
            </div>
            <div class="prompt-card" onclick="sendQuickPrompt('Design a high-throughput event streaming pipeline with Kafka and Spark')">
              <div class="card-title">📊 Big Data Pipelines</div>
              <div class="card-desc">Architect scalable real-time streaming ETL workflows.</div>
            </div>
            <div class="prompt-card" onclick="sendQuickPrompt('Explain the Attention mechanism and Transformer architecture simply')">
              <div class="card-title">🧠 Deep Learning</div>
              <div class="card-desc">How Self-Attention and Transformers drive modern LLMs.</div>
            </div>
          </div>
        </div>
        <!-- Messages will be injected here -->
      </div>
    </div>

    <!-- Bottom Input -->
    <div class="bottom-dock">
      <div class="input-box-wrapper">
        <textarea id="prompt-input" rows="1" placeholder="Message ThinkAI Studio... (Enter to send, Shift+Enter for newline)"></textarea>
        <button id="send-button" class="send-btn" disabled title="Send message">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="12" y1="19" x2="12" y2="5"></line>
            <polyline points="5 12 12 5 19 12"></polyline>
          </svg>
        </button>
      </div>
      <div class="footer-note">ThinkAI Studio · Google Gen AI SDK · Vercel Serverless Ready</div>
    </div>
  </main>

  <!-- Settings Modal -->
  <div class="modal-backdrop" id="settings-modal">
    <div class="modal">
      <div class="modal-header">
        <h3>⚙️ Studio Settings</h3>
        <button class="close-btn" id="close-modal-btn">&times;</button>
      </div>

      <div class="form-group">
        <label>Gemini API Credentials</label>
        <div id="credential-badge" class="status-badge">Checking credentials...</div>
      </div>

      <div class="form-group">
        <label for="temp-slider">Temperature: <span id="temp-val">0.7</span></label>
        <input type="range" id="temp-slider" min="0" max="1" step="0.1" value="0.7">
      </div>

      <div class="form-group">
        <label for="system-prompt">System Instruction</label>
        <textarea id="system-prompt" class="form-control" rows="3" placeholder="You are an expert AI research assistant...">You are a senior AI research specialist and senior technical mentor. Provide concise, accurate, well-formatted answers with practical code examples.</textarea>
      </div>

      <div style="display: flex; justify-content: flex-end; gap: 8px; margin-top: 6px;">
        <button class="btn" id="test-conn-btn">Test Connection</button>
        <button class="btn btn-primary" id="save-settings-btn">Done</button>
      </div>
    </div>
  </div>

  <script>
    // State
    const state = {
      messages: [],
      model: "gemini-3.5-flash-lite",
      temperature: 0.7,
      systemInstruction: document.getElementById('system-prompt').value,
      isGenerating: false
    };

    // DOM Elements
    const chatContainer = document.getElementById('chat-container');
    const heroSection = document.getElementById('hero-section');
    const chatScroll = document.getElementById('chat-scroll');
    const promptInput = document.getElementById('prompt-input');
    const sendButton = document.getElementById('send-button');
    const modelSelect = document.getElementById('model-select');
    const newChatBtn = document.getElementById('new-chat-btn');
    const settingsBtn = document.getElementById('settings-btn');
    const settingsModal = document.getElementById('settings-modal');
    const closeModalBtn = document.getElementById('close-modal-btn');
    const saveSettingsBtn = document.getElementById('save-settings-btn');
    const tempSlider = document.getElementById('temp-slider');
    const tempVal = document.getElementById('temp-val');
    const systemPromptInput = document.getElementById('system-prompt');
    const credentialBadge = document.getElementById('credential-badge');
    const testConnBtn = document.getElementById('test-conn-btn');

    // Auto-resize input
    promptInput.addEventListener('input', () => {
      promptInput.style.height = 'auto';
      promptInput.style.height = Math.min(promptInput.scrollHeight, 180) + 'px';
      const hasText = promptInput.value.trim().length > 0;
      sendButton.disabled = !hasText || state.isGenerating;
      sendButton.classList.toggle('active', hasText && !state.isGenerating);
    });

    // Enter to submit
    promptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendButton.disabled) handleSend();
      }
    });

    sendButton.addEventListener('click', handleSend);

    function sendQuickPrompt(text) {
      promptInput.value = text;
      promptInput.dispatchEvent(new Event('input'));
      handleSend();
    }

    // Modal Events
    settingsBtn.addEventListener('click', () => {
      settingsModal.style.display = 'flex';
      checkHealth();
    });
    closeModalBtn.addEventListener('click', () => settingsModal.style.display = 'none');
    saveSettingsBtn.addEventListener('click', () => {
      state.temperature = parseFloat(tempSlider.value);
      state.systemInstruction = systemPromptInput.value;
      settingsModal.style.display = 'none';
    });
    tempSlider.addEventListener('input', () => tempVal.innerText = tempSlider.value);

    newChatBtn.addEventListener('click', () => {
      state.messages = [];
      chatContainer.innerHTML = '';
      chatContainer.appendChild(heroSection);
      heroSection.style.display = 'flex';
      promptInput.value = '';
      promptInput.dispatchEvent(new Event('input'));
      promptInput.focus();
    });

    modelSelect.addEventListener('change', (e) => {
      state.model = e.target.value;
    });

    // API Health Check
    async function checkHealth() {
      try {
        const res = await fetch('/api/health');
        const data = await res.json();
        if (data.has_api_key) {
          credentialBadge.className = 'status-badge';
          credentialBadge.innerHTML = '● Key Configured: ' + data.key_masked;
        } else {
          credentialBadge.className = 'status-badge error';
          credentialBadge.innerHTML = '⚠️ No API Key found in env';
        }
      } catch (e) {
        credentialBadge.className = 'status-badge error';
        credentialBadge.innerHTML = '⚠️ Error checking backend';
      }
    }

    testConnBtn.addEventListener('click', async () => {
      testConnBtn.innerText = 'Testing...';
      try {
        const res = await fetch('/api/verify', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({model: state.model})
        });
        const data = await res.json();
        if (data.success) {
          alert('✅ ' + data.message);
        } else {
          alert('❌ Connection check failed: ' + data.message);
        }
      } catch (e) {
        alert('❌ Error: ' + e.message);
      } finally {
        testConnBtn.innerText = 'Test Connection';
      }
    });

    // Send Message
    async function handleSend() {
      const text = promptInput.value.trim();
      if (!text || state.isGenerating) return;

      // Hide hero
      heroSection.style.display = 'none';

      // Add user message
      appendMessage('user', text);
      state.messages.push({ role: 'user', content: text });

      promptInput.value = '';
      promptInput.style.height = 'auto';
      sendButton.disabled = true;
      sendButton.classList.remove('active');
      state.isGenerating = true;

      // Add assistant placeholder with typing dots
      const assistantRow = appendMessage('assistant', '', true);
      const bubble = assistantRow.querySelector('.msg-bubble');

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text,
            model: state.model,
            temperature: state.temperature,
            system_instruction: state.systemInstruction,
            history: state.messages.slice(0, -1)
          })
        });

        const data = await response.json();
        if (!response.ok || data.error) {
          bubble.innerHTML = '<span style="color:var(--danger)">⚠️ ' + (data.error || 'Server error') + '</span>';
        } else {
          const replyText = data.reply || '';
          bubble.innerHTML = renderMarkdown(replyText);
          state.messages.push({ role: 'assistant', content: replyText });
          attachCopyButtons(bubble);
        }
      } catch (err) {
        bubble.innerHTML = '<span style="color:var(--danger)">⚠️ Network Error: Unable to contact server.</span>';
      } finally {
        state.isGenerating = false;
        promptInput.dispatchEvent(new Event('input'));
        chatScroll.scrollTop = chatScroll.scrollHeight;
      }
    }

    function appendMessage(role, content, isTyping = false) {
      const row = document.createElement('div');
      row.className = `msg-row ${role}`;

      if (role === 'assistant') {
        const avatar = document.createElement('div');
        avatar.className = 'msg-avatar assistant';
        avatar.innerText = '⚡';
        row.appendChild(avatar);
      }

      const bubble = document.createElement('div');
      bubble.className = 'msg-bubble';

      if (isTyping) {
        bubble.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
      } else if (role === 'user') {
        bubble.innerText = content;
      } else {
        bubble.innerHTML = renderMarkdown(content);
        attachCopyButtons(bubble);
      }

      row.appendChild(bubble);
      chatContainer.appendChild(row);
      chatScroll.scrollTop = chatScroll.scrollHeight;
      return row;
    }

    function renderMarkdown(text) {
      if (!text) return '';
      // Escape HTML
      let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

      // Code blocks ```code```
      html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><button class="copy-btn">Copy</button><code>${code.trim()}</code></pre>`;
      });

      // Inline code
      html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

      // Bold & Italic
      html = html.replace(/\\*\\*([^\\*]+)\\*\\*/g, '<strong>$1</strong>');
      html = html.replace(/\\*([^\\*]+)\\*/g, '<em>$1</em>');

      // Line breaks
      html = html.replace(/\\n/g, '<br>');
      return html;
    }

    function attachCopyButtons(container) {
      container.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const code = btn.nextElementSibling.innerText;
          navigator.clipboard.writeText(code).then(() => {
            btn.innerText = 'Copied!';
            setTimeout(() => btn.innerText = 'Copy', 2000);
          });
        });
      });
    }

    // Initial check
    checkHealth();
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
@app.route("/index", methods=["GET"])
@app.route("/index.html", methods=["GET"])
@app.route("/index.py", methods=["GET"])
@app.route("/app", methods=["GET"])
@app.route("/app.py", methods=["GET"])
@app.route("/api/index", methods=["GET"])
@app.route("/api/index.py", methods=["GET"])
def index():
    """Serves the ChatGPT-style single-page interface."""
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    """Returns application health status and credential readiness."""
    studio_url = os.getenv("THINKAI_STUDIO_URL")
    if studio_url:
        try:
            target = f"{studio_url.rstrip('/')}/api/health"
            resp = requests.get(target, timeout=5)
            data = resp.json()
            data["bound_service"] = "thinkai-studio"
            return jsonify(data), resp.status_code
        except Exception:
            pass

    api_key = os.getenv("GEMINI_API_KEY")
    return jsonify({
        "status": "healthy",
        "service": "ThinkAI Studio",
        "platform": "Vercel / Cloud Serverless",
        "has_api_key": bool(api_key),
        "key_masked": mask_api_key(api_key),
    })


@app.route("/api/models", methods=["GET"])
@app.route("/models", methods=["GET"])
def models():
    """Returns list of curated and recommended models."""
    studio_url = os.getenv("THINKAI_STUDIO_URL")
    if studio_url:
        try:
            target = f"{studio_url.rstrip('/')}/api/models"
            resp = requests.get(target, timeout=5)
            return jsonify(resp.json()), resp.status_code
        except Exception:
            pass
    return jsonify({"models": RECOMMENDED_MODELS})


@app.route("/api/verify", methods=["POST", "GET"])
@app.route("/verify", methods=["POST", "GET"])
def verify():
    """Tests live connection to Gemini API."""
    data = request.get_json(silent=True) or {}
    studio_url = os.getenv("THINKAI_STUDIO_URL")
    if studio_url:
        try:
            target = f"{studio_url.rstrip('/')}/api/verify"
            resp = requests.post(target, json=data, timeout=15)
            return jsonify(resp.json()), resp.status_code
        except Exception:
            pass
    model = data.get("model", "gemini-3.5-flash-lite")
    service = GeminiService()
    connected, message = service.verify_connection(model=model)
    return jsonify({"success": connected, "message": message})


@app.route("/api/chat", methods=["POST", "GET"])
@app.route("/chat", methods=["POST", "GET"])
def chat():
    """
    Executes a multi-turn chat request using Google Gemini API.
    Guarantees no raw secrets are leaked in error responses.
    """
    if request.method == "GET":
        return jsonify({
            "status": "active",
            "endpoint": "/api/chat",
            "method": "POST",
            "info": "Send a POST request with JSON payload {'message': '...'} to generate an AI response."
        })

    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Message content cannot be empty."}), 400

    studio_url = os.getenv("THINKAI_STUDIO_URL")
    if studio_url:
        try:
            target = f"{studio_url.rstrip('/')}/api/chat"
            resp = requests.post(target, json=data, timeout=30)
            return jsonify(resp.json()), resp.status_code
        except Exception as forward_err:
            sanitized = sanitize_error_message(forward_err)
            return jsonify({"error": f"Internal service communication error: {sanitized}"}), 502

    try:
        model = data.get("model", "gemini-3.5-flash-lite")
        temperature = float(data.get("temperature", 0.7))
        system_instruction = data.get("system_instruction")
        history = data.get("history", [])

        # Build conversation message list for GeminiService
        conversation_messages: List[Dict[str, str]] = []
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ["user", "assistant", "model"] and content:
                conversation_messages.append({"role": "user" if role == "user" else "assistant", "content": content})

        # Append current user prompt
        conversation_messages.append({"role": "user", "content": user_message})

        service = GeminiService()

        # Collect response chunks
        chunks = []
        for chunk in service.stream_conversation(
            model=model,
            messages=conversation_messages,
            system_instruction=system_instruction,
            temperature=temperature,
        ):
            chunks.append(chunk)

        full_reply = "".join(chunks)
        return jsonify({
            "reply": full_reply,
            "model": model,
        })

    except Exception as e:
        sanitized = sanitize_error_message(e)
        return jsonify({"error": sanitized}), 500


@app.errorhandler(404)
def handle_404(e):
    """
    Fallback 404 handler.
    If the requested URL is an unrecognized /api/ route, return a clean JSON 404.
    For all other browser paths or Vercel routing rewrites, serve the ChatGPT SPA interface.
    """
    if request.path.startswith("/api/") and not request.path.startswith("/api/index"):
        return jsonify({"error": f"API endpoint '{request.path}' was not found on this server."}), 404
    return Response(INDEX_HTML, mimetype="text/html")


# Export WSGI application for Vercel
handler = app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)



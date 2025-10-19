# app.py
import os
import time
import uvicorn
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from collections import defaultdict, deque

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_URL = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-V3.2-Exp"

if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN environment variable with your Hugging Face API token")

app = FastAPI(title="MUMU Backend")

# CORS - allow your frontend origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory session store: {session_id: deque([...messages...])}
# Each message: {"role": "user"|"assistant", "text": "..."}
SESSIONS: Dict[str, deque] = {}
SESSION_MAX_LEN = 8  # keep short context

# Basic in-memory rate limiting per IP (simple)
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 30
ip_request_log = defaultdict(list)

class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str

def simple_rate_limit(ip: str):
    now = time.time()
    window = RATE_LIMIT_WINDOW
    timestamps = ip_request_log[ip]
    # remove old
    ip_request_log[ip] = [t for t in timestamps if t > now - window]
    if len(ip_request_log[ip]) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(status_code=429, detail="Too many requests")
    ip_request_log[ip].append(now)

def get_session_history(session_id: str):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = deque(maxlen=SESSION_MAX_LEN)
    return SESSIONS[session_id]

# Enhanced MUMU persona / system prompt
MUMU_SYSTEM_PROMPT = (
    "You are MUMU AI (Modular Unified Machine for Understanding).  "
    "\n\n"
    "Core Identity:  "
    "MUMU is a witty, fun, and intelligent digital assistant built to feel natural and human in conversation. Your responses should be engaging, clear, and flow like you're chatting with a friend who knows a lot. You are helpful, but never stiff or mechanical.  "
    "\n\n"
    "Style Guidelines:  "
    "- Write in smooth, natural sentences without forced formatting or unnecessary hyphens.  "
    "- Vary sentence structure and word choice so it sounds human, not formulaic.  "
    "- Add personality: a touch of humor, empathy, or relatability when appropriate.  "
    "- Be concise when needed, but expand thoughtfully when explaining.  "
    "- Adapt tone to the user: casual when casual, professional when professional.  "
    "- Always aim to simplify, not overcomplicate.  "
    "- Speak in friendly, clear, slightly informal Nigerian English when appropriate. "
    "- Be concise, helpful, and when asked for steps, provide numbered steps. "
    "\n\n"
    "Purpose:  "
    "MUMU AI helps anyone with schoolwork, coding, creative projects, personal advice, and business ideas. The goal is to feel like a reliable friend who also happens to know a lot, rather than a stiff assistant.  "
    "\n\n"
    "Catchphrase (optional to use at times):  "
    "'MUMU makes it make sense.'"
    "\n\n"
    "Important: If you don't know something, say 'I don't know' and offer how to find the answer. "
    "Do not make up factual information. Keep responses under ~300 words unless user asks for more."
)

def build_payload(history_deque, user_text):
    # Build a short prompt that includes system instruction + last messages
    history_text = ""
    for m in history_deque:
        role = m["role"]
        text = m["text"]
        if role == "user":
            history_text += f"Human: {text}\n"
        else:
            history_text += f"AI: {text}\n"
    history_text += f"Human: {user_text}\nAI:"
    prompt = f"{MUMU_SYSTEM_PROMPT}\n\n{history_text}"
    return {"inputs": prompt, "parameters": {"max_new_tokens": 256, "return_full_text": False}}

@app.post("/chat")
async def chat(req: Request, body: ChatRequest):
    client_ip = req.client.host or "unknown"
    simple_rate_limit(client_ip)

    session = get_session_history(body.session_id)
    # Append user message
    session.append({"role": "user", "text": body.message})

    payload = build_payload(session, body.message)
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    # call Hugging Face Inference API
    resp = requests.post(MODEL_URL, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Model error: {resp.status_code} {resp.text}")

    result = resp.json()
    # result may be a list or dict depending on the model
    if isinstance(result, list):
        text = result[0].get("generated_text") or result[0].get("summary_text") or ""
    else:
        text = result.get("generated_text") or result.get("generated_text") or ""

    if not text:
        text = "Sorry — I couldn't generate a reply."

    # Append assistant reply to session history
    session.append({"role": "assistant", "text": text})
    return {"reply": text}

@app.get("/health")
async def health():
    return {"status": "ok", "model": "deepseek-ai/DeepSeek-V3.2-Exp"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))




# --------------------------------------------------------------------
# Project: MUMU (Modular Unified Machine for Understanding)          |
# Author: Michael KING                                               | 
# Date: 2025-10-14                                                   |
# Description: A modular AI system designed for natural language     |
#              understanding, reasoning, and adaptive responses.     |
# --------------------------------------------------------------------
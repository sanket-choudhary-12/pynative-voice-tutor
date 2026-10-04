"""
config.py — every tunable setting lives here, nowhere else.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# OPTIONAL: only used when running locally with a .env file. On the deployed
# app each visitor types their own key into the page instead (see app.py).
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- Speech-to-text ---
# "base" is a good speed/accuracy balance on a normal laptop CPU.
# Try "small" if you want noticeably better accuracy and don't mind an
# extra second or so of latency per answer.
WHISPER_MODEL_SIZE = "base"
WHISPER_LANGUAGE = "en"

# --- Text-to-speech ---
TTS_VOICE = "en-US-AriaNeural"
TTS_RATE = "+5%"

# --- LLM ---
# Used only if we can't reach Groq's model-listing endpoint at all (see
# llm_chains.py's resolve_model_name) - normally the app figures out which
# model your specific account can use automatically.
LLM_MODEL_FALLBACK = "llama-3.3-70b-versatile"

# --- Assistant identity ---
ASSISTANT_NAME = "PyNative"

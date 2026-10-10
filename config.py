"""
config.py  -  every tunable setting lives here, nowhere else.

Sections:
    1. API key (optional, for running locally)
    2. Speech-to-text (Whisper): model choices + Python vocabulary hint
    3. Text-to-speech (edge-tts): voice choices + speaking speed
    4. LLM fallback model
    5. Assistant name
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 1. API key
# ---------------------------------------------------------------------------
# OPTIONAL: only used when running locally with a .env file. On the deployed
# app each visitor types their own key into the page instead (see app.py).
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ---------------------------------------------------------------------------
# 2. Speech-to-text (Whisper, runs locally)
# ---------------------------------------------------------------------------
# DEFAULT model size when the app starts. The student can switch it in the
# sidebar ("Speech recognition model"):
#   tiny  = fastest, least accurate
#   base  = good balance on a normal laptop CPU (default)
#   small = noticeably more accurate, a bit slower, uses more memory
WHISPER_MODEL_SIZE = "base"

# The sizes offered in the sidebar dropdown, and the text shown for each.
# (Bigger models such as "medium" are left out on purpose: too slow on a CPU.)
WHISPER_MODEL_OPTIONS = ["tiny", "base", "small"]
WHISPER_MODEL_LABELS = {
    "tiny": "tiny - fastest, least accurate",
    "base": "base - balanced (default)",
    "small": "small - most accurate, slower",
}

WHISPER_LANGUAGE = "en"

# Vocabulary hint for Whisper ("initial_prompt"). Whisper treats this text as
# if it had been said just before the recording, which nudges it towards
# these spellings. Without it, Python terms are often misheard (e.g. "tuple"
# -> "toople"/"two pull", "elif" -> "else if"/"Eliff", "dict" -> "dick").
# Written as a natural sentence on purpose: Whisper copes better with that
# than with a bare word list. Keep it short - Whisper only uses the last
# ~224 tokens, and a very long prompt can make it "echo" the prompt back on
# near-silent recordings (asr_utils.py has a guard against that).
# USE_ASR_PROMPT is only the STARTING value of the sidebar switch
# "Python vocabulary hint" - handy to compare results with and without it.
USE_ASR_PROMPT = True
ASR_PROMPT = (
    "A student answers a Python quiz question about variables, integers, floats, "
    "strings, booleans, lists, tuples, dictionaries, dicts, sets, None, if, elif, "
    "else, for loops, while loops, range, break, continue, functions, def, return, "
    "arguments, parameters, *args, **kwargs, lambda, list comprehensions, classes, "
    "objects, self, __init__, inheritance, methods, modules, import, try, except, "
    "exceptions, f-strings, slicing, indexing, len, append, print and input."
)

# ---------------------------------------------------------------------------
# 3. Text-to-speech (edge-tts)
# ---------------------------------------------------------------------------
# Voices the student can pick in the sidebar: {text shown : edge-tts voice id}.
# All of these are standard Microsoft neural voices supported by edge-tts.
# To add one, run  python -m edge_tts --list-voices  and copy a "Name".
VOICE_OPTIONS = {
    "Aria (US, female)": "en-US-AriaNeural",
    "Jenny (US, female)": "en-US-JennyNeural",
    "Guy (US, male)": "en-US-GuyNeural",
    "Sonia (UK, female)": "en-GB-SoniaNeural",
    "Ryan (UK, male)": "en-GB-RyanNeural",
    "Natasha (Australia, female)": "en-AU-NatashaNeural",
    "William (Australia, male)": "en-AU-WilliamNeural",
    "Neerja (India, female)": "en-IN-NeerjaNeural",
    "Prabhat (India, male)": "en-IN-PrabhatNeural",
}
DEFAULT_VOICE_LABEL = "Aria (US, female)"   # must be one of the keys above

# This is also the SAFETY-NET voice: if the chosen voice fails, tts_utils.py
# retries once with this one.
TTS_VOICE = VOICE_OPTIONS[DEFAULT_VOICE_LABEL]

# Speaking speed in percent: 0 = normal, +20 = 20% faster, -20 = 20% slower.
# The sidebar slider moves between MIN and MAX in steps of STEP.
TTS_RATE_PERCENT = 5        # starting value of the slider
TTS_RATE_MIN = -30
TTS_RATE_MAX = 40
TTS_RATE_STEP = 5
# The same default in the text format edge-tts expects, e.g. "+5%" or "-10%".
TTS_RATE = f"{TTS_RATE_PERCENT:+d}%"

# ---------------------------------------------------------------------------
# 4. LLM
# ---------------------------------------------------------------------------
# Used only if we can't reach Groq's model-listing endpoint at all (see
# llm_chains.py's resolve_model_name) - normally the app figures out which
# model your specific account can use automatically.
LLM_MODEL_FALLBACK = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# 5. Assistant identity
# ---------------------------------------------------------------------------
def tutor_name_from_label(voice_label: str) -> str:
    """The tutor's name is simply the voice's name: the text before the first
    bracket in the label. "Ryan (UK, male)" -> "Ryan", "Aria (US, female)" -> "Aria".
    If the label is empty or has no usable name, returns an empty string."""
    return (voice_label or "").split(" (")[0].strip()


# Name used when no voice has been chosen yet = the name of the default voice
# ("Aria"). While the app runs, the name always follows the selected voice.
ASSISTANT_NAME = tutor_name_from_label(DEFAULT_VOICE_LABEL) or "Tutor"

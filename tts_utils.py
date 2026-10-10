"""
tts_utils.py  -  the "mouth" of the app (text to speech).

WHAT THIS FILE DOES (simple words):
    You give it a sentence (text), optionally a voice and a speaking speed. It
    gives you back the spoken sentence as mp3 audio bytes, using edge-tts (free
    Microsoft neural voices). Nothing is saved to disk - the audio stays in memory.

WHY IT LOOKS A BIT UNUSUAL:
    edge-tts is "async" code, but the rest of our app is normal code. Streamlit
    itself also uses asyncio, so calling asyncio.run() every time can crash with
    "event loop already running". So we start ONE background thread that keeps
    one event loop alive forever, and we hand every speech job to that loop.

SAFETY:
    * A time limit (20 seconds) on every speech request, so a stuck network can
      never freeze the app forever.
    * Two attempts. The FIRST uses the voice and speed the student chose. If it
      fails, the SECOND uses the default voice/speed from config.py (in case the
      chosen voice is the problem).
    * An invalid speed value is replaced by the default instead of crashing.
    * If speech still fails we return empty bytes -> the app simply shows the
      text without voice. It never crashes.
"""

import asyncio
import re
import threading

import edge_tts

import config

# --- Start the background event loop ONCE, when this file is first imported ---
# (daemon=True means this thread will not stop Python from closing normally.)
_loop = asyncio.new_event_loop()
_thread = threading.Thread(target=_loop.run_forever, daemon=True)
_thread.start()

# How long (seconds) we are willing to wait for one speech request.
_TIMEOUT_SECONDS = 20.0

# A valid speed looks like "+5%", "-20%", "+0%" (this is what edge-tts accepts).
_RATE_PATTERN = re.compile(r"[+-]\d+%")


async def _generate_mp3_bytes(text: str, voice: str, rate: str) -> bytes:
    """Asks edge-tts to speak `text` and collects the mp3 pieces it streams back.
    Returns all the pieces joined together as one bytes object."""
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    chunks = bytearray()
    async for chunk in communicate.stream():
        # edge-tts also sends "word timing" pieces; we only want the audio ones.
        if chunk["type"] == "audio":
            chunks.extend(chunk["data"])
    return bytes(chunks)


def text_to_mp3_bytes(text: str, voice: str = None, rate: str = None) -> bytes:
    """MAIN FUNCTION: text in -> mp3 bytes out.

    voice : an edge-tts voice id such as "en-GB-RyanNeural". None = config default.
    rate  : speed text such as "+10%" or "-20%". None/invalid = config default.

    Returns b"" (empty bytes) if the text is empty or if speech failed twice.
    The app treats empty bytes as "no voice this time" and carries on."""
    if not text or not text.strip():
        return b""

    voice = voice or config.TTS_VOICE
    if not (isinstance(rate, str) and _RATE_PATTERN.fullmatch(rate)):
        rate = config.TTS_RATE

    for attempt in (1, 2):  # try at most twice
        # Attempt 1: the student's choice. Attempt 2: the safe defaults.
        use_voice, use_rate = (voice, rate) if attempt == 1 else (config.TTS_VOICE, config.TTS_RATE)
        future = None
        try:
            future = asyncio.run_coroutine_threadsafe(
                _generate_mp3_bytes(text, use_voice, use_rate), _loop
            )
            audio = future.result(timeout=_TIMEOUT_SECONDS)  # wait, but not forever
            if audio:
                return audio
            print(f"[TTS warning] attempt {attempt} ({use_voice}, {use_rate}): got no audio back.")
        except Exception as e:  # includes the timeout error
            print(f"[TTS error] attempt {attempt} ({use_voice}, {use_rate}): {e}")
            if future is not None:
                future.cancel()  # stop the stuck job so it does not pile up
    return b""

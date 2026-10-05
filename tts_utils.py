"""
tts_utils.py  -  the "mouth" of the app.

WHAT THIS FILE DOES (simple words):
    You give it a sentence (text). It gives you back the spoken version of that
    sentence as mp3 audio bytes, using edge-tts (free Microsoft neural voices).
    Nothing is saved to disk - the audio stays in memory.

WHY IT LOOKS A BIT UNUSUAL:
    edge-tts is "async" code, but the rest of our app is normal code. Streamlit
    itself also uses asyncio, so calling asyncio.run() every time can crash with
    "event loop already running". So we start ONE background thread that keeps
    one event loop alive forever, and we hand every speech job to that loop.

WHAT IS NEW IN THIS VERSION (safety):
    * A time limit (20 seconds) on every speech request, so a stuck network can
      never freeze the app forever.
    * One automatic retry if the first attempt fails.
    * If speech still fails we return empty bytes -> the app simply shows the
      text without voice. It never crashes.
"""

import asyncio
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


def text_to_mp3_bytes(text: str) -> bytes:
    """MAIN FUNCTION: text in -> mp3 bytes out.

    Returns b"" (empty bytes) if the text is empty or if speech failed twice.
    The app treats empty bytes as "no voice this time" and carries on."""
    if not text or not text.strip():
        return b""

    for attempt in (1, 2):  # try at most twice
        future = None
        try:
            future = asyncio.run_coroutine_threadsafe(
                _generate_mp3_bytes(text, config.TTS_VOICE, config.TTS_RATE), _loop
            )
            audio = future.result(timeout=_TIMEOUT_SECONDS)  # wait, but not forever
            if audio:
                return audio
            print(f"[TTS warning] attempt {attempt}: got no audio back.")
        except Exception as e:  # includes the timeout error
            print(f"[TTS error] attempt {attempt}: {e}")
            if future is not None:
                future.cancel()  # stop the stuck job so it does not pile up
    return b""

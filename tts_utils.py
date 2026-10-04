"""
tts_utils.py
------------
Turns text into spoken audio (mp3 bytes in memory - no temp files needed)
using edge-tts (free, natural Microsoft Edge neural voices).

edge-tts's API is async; the rest of this app calls it synchronously. We
run one dedicated background event loop just for this, rather than calling
asyncio.run() directly each time - asyncio.run() refuses to start if a
loop is already running in the current thread, and Streamlit's own
internals use asyncio too, so this keeps TTS calls safe no matter what
else is happening.
"""

import asyncio
import threading

import edge_tts

import config

_loop = asyncio.new_event_loop()
_thread = threading.Thread(target=_loop.run_forever, daemon=True)
_thread.start()


async def _generate_mp3_bytes(text: str, voice: str, rate: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    chunks = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.extend(chunk["data"])
    return bytes(chunks)


def text_to_mp3_bytes(text: str) -> bytes:
    """Returns the spoken audio for `text` as raw mp3 bytes, ready to hand
    to st.audio(..., autoplay=True). Returns empty bytes (and prints the
    error) if TTS fails for any reason, so a network hiccup never crashes
    the whole app - the text is still visible on screen."""
    if not text or not text.strip():
        return b""
    try:
        future = asyncio.run_coroutine_threadsafe(
            _generate_mp3_bytes(text, config.TTS_VOICE, config.TTS_RATE), _loop
        )
        return future.result()
    except Exception as e:
        print(f"[TTS error]: {e}")
        return b""

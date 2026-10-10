"""
asr_utils.py  -  the "ears" of the app (speech to text).

WHAT THIS FILE DOES (simple words):
    You give it the bytes of a recording (from Streamlit's st.audio_input
    microphone widget). It gives you back what was said as a normal string,
    using faster-whisper. Everything runs on your own computer: free, no API key.

NEW IN THIS VERSION:
    * The Whisper model size can be chosen at run time (tiny / base / small).
      Each size is loaded once and kept in memory (at most 2 at a time, so
      memory never grows without limit). If a chosen size cannot be loaded
      (for example no internet for the first download), we quietly fall back
      to the default size from config.py instead of failing.
    * The Python vocabulary hint ("initial_prompt") can be switched on/off,
      so you can show the difference in your presentation.
    * An "echo guard": on near-silent recordings Whisper can repeat the hint
      text back. If the result is just a piece of the hint, we treat it as
      "nothing was said".

Important note on audio format: we do NOT assume the recorder sends plain
WAV, and we do NOT use faster-whisper's own decode_audio() function, because
it calls av.open(..., metadata_errors="ignore") and newer PyAV versions
removed that argument (TypeError every time). We decode with `av` ourselves,
without that argument, which avoids depending on a fragile version pairing.
"""

import io
import traceback

import av
import numpy as np
import streamlit as st
from faster_whisper import WhisperModel

import config


@st.cache_resource(max_entries=2, show_spinner=False)
def get_whisper_model(model_size: str) -> WhisperModel:
    """Loads the Whisper model of the given size ("tiny", "base", "small"...).
    st.cache_resource keeps the loaded model, so each size is loaded only ONCE
    (loading takes seconds; the very first time it also downloads the model).
    max_entries=2 keeps at most two sizes in memory.
    compute_type="int8" is faster on a normal CPU with a tiny accuracy loss."""
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def preload_model(model_size: str) -> bool:
    """Loads a model ahead of time, so the first answer is not delayed.
    Returns True on success, False on failure. NEVER raises an error."""
    try:
        get_whisper_model(model_size)
        return True
    except Exception:
        print(f"[ASR ERROR] Could not load the Whisper model '{model_size}':")
        traceback.print_exc()
        return False


def _get_model_with_fallback(model_size: str) -> WhisperModel:
    """Returns the model of the requested size. If it cannot be loaded, falls
    back to the default size from config.py. Raises only if the default fails too."""
    try:
        return get_whisper_model(model_size)
    except Exception:
        if model_size == config.WHISPER_MODEL_SIZE:
            raise  # already the default - nothing left to fall back to
        print(f"[ASR WARNING] Model '{model_size}' failed to load; using the default "
              f"'{config.WHISPER_MODEL_SIZE}' instead. Details:")
        traceback.print_exc()
        return get_whisper_model(config.WHISPER_MODEL_SIZE)


def _decode_to_numpy_16k(audio_bytes: bytes) -> np.ndarray:
    """Decodes audio bytes (WAV, WebM/Opus, whatever the browser sent - av
    detects the format via its bundled ffmpeg) into a 16 kHz mono float32
    array with values between -1 and +1, the format faster-whisper expects."""
    container = av.open(io.BytesIO(audio_bytes))
    audio_stream = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)

    chunks = []
    for packet in container.demux(audio_stream):
        for frame in packet.decode():
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().flatten())
    for resampled in resampler.resample(None):  # flush any buffered samples
        chunks.append(resampled.to_ndarray().flatten())
    container.close()

    if not chunks:
        return np.array([], dtype=np.float32)

    samples_int16 = np.concatenate(chunks)
    return samples_int16.astype(np.float32) / 32768.0


def transcribe_wav_bytes(audio_bytes: bytes, model_size: str = None, use_prompt: bool = None) -> str:
    """MAIN FUNCTION: recording bytes in -> text out.

    model_size : "tiny" / "base" / "small". None = the default from config.py.
    use_prompt : True = give Whisper the Python vocabulary hint, False = don't.
                 None = the default from config.py.

    Returns "" on ANY failure (empty audio, broken audio, model problem) instead
    of raising, so a bad recording never crashes the app. The full error is
    printed to the terminal so the real cause is always visible there."""
    if not audio_bytes:
        return ""

    size = model_size or config.WHISPER_MODEL_SIZE
    if use_prompt is None:
        use_prompt = config.USE_ASR_PROMPT

    try:
        audio_array = _decode_to_numpy_16k(audio_bytes)
        if audio_array.size == 0:
            return ""

        model = _get_model_with_fallback(size)
        prompt = config.ASR_PROMPT if use_prompt else None
        segments, _info = model.transcribe(
            audio_array,
            language=config.WHISPER_LANGUAGE,
            beam_size=1,            # greedy decoding = fastest
            initial_prompt=prompt,  # None = no hint
        )
        text = " ".join(segment.text for segment in segments).strip()

        # Echo guard: a long result that is only a piece of the hint text means
        # Whisper "heard" nothing and repeated the hint back -> treat as silence.
        if prompt and len(text) > 25 and text.lower() in prompt.lower():
            print("[ASR WARNING] Result looked like an echo of the vocabulary hint; ignoring it.")
            return ""
        return text
    except Exception:
        print("[ASR ERROR] Full traceback below - copy this if you need help diagnosing it:")
        traceback.print_exc()
        return ""

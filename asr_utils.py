"""
asr_utils.py
------------
Turns recorded audio (from the browser mic recorder) into text, using
faster-whisper - runs 100% locally, free, no API key.

Important note on audio format: we do NOT assume the recorder sends plain
WAV. In testing it sent a different container format, which ruled out a
simple Python `wave`-module decode (WAV only).

We also do NOT use faster-whisper's own built-in decode_audio() function,
even though that's normally the "just let the library handle it" choice.
Reason: that function calls `av.open(..., metadata_errors="ignore")`
internally, and PyAV removed that argument in newer major versions (we
hit this directly: av 19.0.0 installed, faster-whisper 1.2.1 still calling
the old signature -> TypeError every time). Rather than chase an exact
compatible version pairing between two fast-moving libraries - which
depends on prebuilt-wheel availability per platform and breaks again on
the next update - we decode the audio ourselves with `av` directly,
deliberately WITHOUT the removed argument (we don't actually need it; it
only suppressed noisy metadata-parsing warnings). This sidesteps the
incompatibility entirely rather than depending on a fragile version pin.
"""

import io
import traceback

import av
import numpy as np
import streamlit as st
from faster_whisper import WhisperModel

import config


@st.cache_resource(show_spinner="Loading local speech-to-text model...")
def get_whisper_model() -> WhisperModel:
    # compute_type="int8" is noticeably faster on a normal CPU with a very
    # small accuracy trade-off - good for a responsive assistant.
    return WhisperModel(config.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


def _decode_to_numpy_16k(audio_bytes: bytes) -> np.ndarray:
    """Decodes arbitrary audio bytes (WAV, WebM/Opus, whatever the browser
    sent - av auto-detects the format via bundled ffmpeg) into a 16kHz
    mono float32 array, the exact format faster-whisper expects when given
    a numpy array directly (which skips its own internal decoder)."""
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


def transcribe_wav_bytes(audio_bytes: bytes) -> str:
    """Takes the raw audio bytes returned by streamlit-mic-recorder and
    returns the transcribed text. Returns "" on any failure instead of
    raising, so a bad recording never crashes the app - but prints the
    FULL error traceback to the terminal, so the real cause is always
    visible there instead of being hidden behind a generic message."""
    if not audio_bytes:
        return ""
    try:
        audio_array = _decode_to_numpy_16k(audio_bytes)
        if audio_array.size == 0:
            return ""
        model = get_whisper_model()
        segments, _info = model.transcribe(audio_array, language=config.WHISPER_LANGUAGE, beam_size=1)
        return " ".join(segment.text for segment in segments).strip()
    except Exception:
        print("[ASR ERROR] Full traceback below - copy this if you need help diagnosing it:")
        traceback.print_exc()
        return ""
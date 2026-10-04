"""
app.py — run with: python -m streamlit run app.py

Flow: click Start -> say what Python topic to practice -> answer spoken
questions with real, specific feedback each time -> end anytime for a
final score + summary, or switch topics mid-session.

Streamlit re-runs this whole script top-to-bottom on every interaction, so
anything that must survive between reruns is stored in st.session_state.
Order in this file matters: each stage's input is handled first (which may
update state and call st.rerun()), then we render the orb + current turn's
text, then - at the very end - speak anything new.
"""

import streamlit as st
from streamlit_mic_recorder import mic_recorder

import asr_utils
import config
import llm_chains
import tts_utils

st.set_page_config(page_title=f"{config.ASSISTANT_NAME} - Python Voice Tutor", page_icon="🐍", layout="centered")

# ---------------------------------------------------------------------------
# API-key gate. Every visitor uses THEIR OWN free Groq key, so the app owner's
# key is never stored in the code or on the server. The key lives only in this
# visitor's st.session_state (server memory for their session) - it is never
# written to disk, never logged, never shared with other visitors.
# If a local .env file has GROQ_API_KEY, it is used automatically (handy when
# running on your own laptop) and the gate is skipped.
# ---------------------------------------------------------------------------
if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = config.GROQ_API_KEY or ""

if not st.session_state["groq_api_key"]:
    st.markdown(f'<p style="text-align:center; font-size:2.6rem; font-weight:800; margin-bottom:0;">🐍 {config.ASSISTANT_NAME}</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#9aa5b1;">Your voice-based Python quiz buddy</p>', unsafe_allow_html=True)
    st.info(
        "This app uses Groq's free LLM API. Paste your own free key to begin - "
        "get one in under a minute at https://console.groq.com/keys . "
        "Your key is kept only in your browser session's memory and is never saved."
    )
    entered = st.text_input("Groq API key", type="password", placeholder="gsk_...")
    if st.button("Continue", type="primary"):
        ok, err = llm_chains.validate_api_key(entered)
        if ok:
            st.session_state["groq_api_key"] = entered.strip()
            st.rerun()
        else:
            st.error(err)
    st.stop()

# ---------------------------------------------------------------------------
# Light custom styling - keeps default Streamlit widgets, just tidies colors
# and adds card-style boxes for the current turn's messages.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .pn-title { text-align: center; font-size: 2.6rem; font-weight: 800; margin-bottom: 0; }
    .pn-subtitle { text-align: center; color: #9aa5b1; margin-top: 0; margin-bottom: 1.2rem; }
    .pn-bubble {
        border-radius: 14px; padding: 16px 20px; margin: 10px 0;
        font-size: 1.05rem; line-height: 1.5;
    }
    .pn-assistant { background: rgba(34, 197, 94, 0.08); border-left: 4px solid #22c55e; }
    .pn-user { background: rgba(59, 130, 246, 0.08); border-left: 4px solid #3b82f6; }
    .pn-label { font-weight: 700; font-size: 0.8rem; letter-spacing: 0.04em;
        text-transform: uppercase; opacity: 0.65; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)


def render_orb(state: str = "idle"):
    """Pure-CSS animated circle. Honest note: it switches color/speed
    instantly based on `state`, it does NOT analyze live microphone
    frequencies (the mic recorder only hands over audio after you click
    Stop, so there's nothing to analyze in real time)."""
    st.components.v1.html(f"""
    <div style="display:flex; justify-content:center; padding:10px 0;">
      <div style="display:flex; flex-direction:column; align-items:center;">
        <div class="orb {state}"></div>
        <div class="bars {state}"><span></span><span></span><span></span><span></span><span></span></div>
      </div>
    </div>
    <style>
      .orb {{
        width: 120px; height: 120px; border-radius: 50%;
        background: radial-gradient(circle at 32% 30%, #a5c9ff, #3b82f6);
        box-shadow: 0 0 40px rgba(59,130,246,0.5);
        animation: breathe 3s ease-in-out infinite;
      }}
      .orb.speaking {{
        background: radial-gradient(circle at 32% 30%, #b9fbc0, #22c55e);
        box-shadow: 0 0 50px rgba(34,197,94,0.6);
        animation: pulse 0.55s ease-in-out infinite;
      }}
      @keyframes breathe {{ 0%,100%{{transform:scale(1);}} 50%{{transform:scale(1.05);}} }}
      @keyframes pulse {{ 0%,100%{{transform:scale(1);}} 50%{{transform:scale(1.16);}} }}
      .bars {{ display:flex; gap:6px; height:24px; margin-top:12px; opacity:0; transition:opacity .2s; }}
      .bars.speaking {{ opacity:1; }}
      .bars span {{ width:6px; border-radius:3px; background:#22c55e; animation: eq 0.9s ease-in-out infinite; }}
      .bars span:nth-child(1){{animation-delay:0s;}} .bars span:nth-child(2){{animation-delay:.15s;}}
      .bars span:nth-child(3){{animation-delay:.3s;}} .bars span:nth-child(4){{animation-delay:.1s;}}
      .bars span:nth-child(5){{animation-delay:.25s;}}
      @keyframes eq {{ 0%,100%{{height:6px;}} 50%{{height:22px;}} }}
    </style>
    """, height=190)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "stage": "start",          # start -> topic -> quiz -> result
    "topic": "",
    "history": [],              # full hidden transcript, used as LLM context only
    "turn_scores": [],
    "last_assistant_msg": "",
    "last_user_msg": "",
    "pending_speech": None,
    "last_audio_id": None,
    "final_summary": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def reset_all():
    for _k, _v in _DEFAULTS.items():
        st.session_state[_k] = _v


st.markdown(f'<p class="pn-title">🐍 {config.ASSISTANT_NAME}</p>', unsafe_allow_html=True)
st.markdown('<p class="pn-subtitle">Your voice-based Python quiz buddy</p>', unsafe_allow_html=True)

# Errors are stored in session_state (not shown with st.error right before
# st.rerun(), which would make them vanish instantly) and displayed here on
# the next run, so the real message is always readable.
_err = st.session_state.pop("last_error", None)
if _err:
    st.warning(_err)

with st.sidebar:
    avg = sum(st.session_state.turn_scores) / len(st.session_state.turn_scores) if st.session_state.turn_scores else None
    st.metric("Session average", f"{avg:.1f} / 10" if avg is not None else "—")
    st.caption(f"Questions answered: {len(st.session_state.turn_scores)}")
    st.divider()
    st.caption(f"ASR: faster-whisper ({config.WHISPER_MODEL_SIZE})")
    st.caption(f"LLM: Groq {llm_chains.resolve_model_name(llm_chains.get_api_key())}")
    st.caption("TTS: edge-tts")
    st.divider()
    if st.button("🔄 Restart entire session", use_container_width=True):
        reset_all()
        st.rerun()
    if st.button("🔑 Use a different API key", use_container_width=True):
        reset_all()
        st.session_state["groq_api_key"] = ""
        st.rerun()

# ===========================================================================
# STAGE: start — a single button, no wake word
# ===========================================================================
if st.session_state.stage == "start":
    render_orb("idle")
    st.write("")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("▶️ Start Quiz", use_container_width=True, type="primary"):
            greeting = (
                f"Hey there! I'm {config.ASSISTANT_NAME}, your personal Python quiz buddy. "
                "Which Python topic would you like to practice today? You can also just say "
                "'mixed' for a general quiz."
            )
            st.session_state.last_assistant_msg = greeting
            st.session_state.pending_speech = greeting
            st.session_state.stage = "topic"
            st.rerun()

# ===========================================================================
# STAGE: topic — asking what to study
# ===========================================================================
elif st.session_state.stage == "topic":
    render_orb("speaking" if st.session_state.pending_speech else "idle")

    st.markdown('<div class="pn-label">PyNative</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pn-bubble pn-assistant">{st.session_state.last_assistant_msg}</div>', unsafe_allow_html=True)

    audio = mic_recorder(start_prompt="🎤 Tell me the topic", stop_prompt="⏹ Stop && submit", just_once=True, key="topic_rec")
    if audio and audio.get("id") != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio["id"]
        with st.spinner("Listening..."):
            heard = asr_utils.transcribe_wav_bytes(audio["bytes"])

        if not heard.strip():
            st.warning("I didn't catch that - make sure your browser allowed microphone access, and try again.")
        else:
            try:
                check = llm_chains.check_topic(heard)
            except Exception as e:
                st.session_state["last_error"] = f"Couldn't reach the LLM: {e}"
                check = None

            if check and check.is_valid:
                st.session_state.topic = heard.strip()
                st.session_state.history = [{"role": "user", "content": f"(chosen topic: {heard})"}]
                try:
                    question = llm_chains.generate_first_question(st.session_state.topic)
                except Exception as e:
                    st.session_state["last_error"] = f"Couldn't reach the LLM: {e}"
                    question = "Let's start simple - what is a Python list?"
                combined = f"{check.message} {question}"
                st.session_state.history.append({"role": "assistant", "content": combined})
                st.session_state.last_assistant_msg = combined
                st.session_state.pending_speech = combined
                st.session_state.stage = "quiz"
            elif check:
                st.session_state.last_assistant_msg = check.message
                st.session_state.pending_speech = check.message
        st.rerun()

# ===========================================================================
# STAGE: quiz — the main oral Q&A loop
# ===========================================================================
elif st.session_state.stage == "quiz":
    render_orb("speaking" if st.session_state.pending_speech else "idle")

    st.markdown('<div class="pn-label">PyNative</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pn-bubble pn-assistant">{st.session_state.last_assistant_msg}</div>', unsafe_allow_html=True)

    if st.session_state.last_user_msg:
        st.markdown('<div class="pn-label">You said</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pn-bubble pn-user">{st.session_state.last_user_msg}</div>', unsafe_allow_html=True)

    audio = mic_recorder(start_prompt="🎤 Answer", stop_prompt="⏹ Stop && submit", just_once=True, key="quiz_rec")
    if audio and audio.get("id") != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio["id"]
        with st.spinner("Listening..."):
            heard = asr_utils.transcribe_wav_bytes(audio["bytes"])

        if not heard.strip():
            st.warning("I didn't catch that - please try recording again, a little closer to the mic.")
        else:
            st.session_state.last_user_msg = heard
            st.session_state.history.append({"role": "user", "content": heard})
            try:
                result = llm_chains.evaluate_answer(st.session_state.topic, st.session_state.history, heard)
                st.session_state.turn_scores.append(result.score)
                combined = f"{result.feedback} {result.next_question}"
                st.session_state.history.append({"role": "assistant", "content": combined})
                st.session_state.last_assistant_msg = combined
                st.session_state.pending_speech = combined
            except Exception as e:
                st.session_state["last_error"] = f"Couldn't reach the LLM: {e}"
        st.rerun()

    st.divider()
    col_end, col_switch = st.columns(2)
    with col_end:
        if st.button("🏁 End & get results", use_container_width=True, type="primary"):
            try:
                summary = llm_chains.generate_final_summary(st.session_state.topic, st.session_state.history, st.session_state.turn_scores)
                st.session_state.final_summary = summary
                st.session_state.pending_speech = summary.summary
            except Exception as e:
                st.session_state["last_error"] = f"Couldn't reach the LLM: {e}"
                st.session_state.final_summary = None
            st.session_state.stage = "result"
            st.rerun()
    with col_switch:
        if st.button("🔁 Change topic", use_container_width=True):
            msg = "Sure! What Python topic would you like to practice next?"
            st.session_state.last_assistant_msg = msg
            st.session_state.last_user_msg = ""
            st.session_state.pending_speech = msg
            st.session_state.stage = "topic"
            st.rerun()

# ===========================================================================
# STAGE: result — final score + summary
# ===========================================================================
elif st.session_state.stage == "result":
    render_orb("speaking" if st.session_state.pending_speech else "idle")

    if st.session_state.final_summary:
        st.markdown(
            f'<p style="text-align:center; font-size:2.2rem; font-weight:800;">'
            f'{st.session_state.final_summary.overall_score:.1f} / 10</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="pn-label">PyNative</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pn-bubble pn-assistant">{st.session_state.final_summary.summary}</div>', unsafe_allow_html=True)
    else:
        st.warning("Couldn't generate a summary - see the error above.")

    if st.button("🔄 Start a new session", use_container_width=True, type="primary"):
        reset_all()
        st.rerun()

# ---------------------------------------------------------------------------
# Speak anything new, exactly once, then clear it.
# ---------------------------------------------------------------------------
if st.session_state.pending_speech:
    text_to_speak = st.session_state.pending_speech
    st.session_state.pending_speech = None
    mp3_bytes = tts_utils.text_to_mp3_bytes(text_to_speak)
    if mp3_bytes:
        st.audio(mp3_bytes, format="audio/mp3", autoplay=True)

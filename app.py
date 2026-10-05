"""
app.py  -  the web page and the quiz flow.   Run with:  python -m streamlit run app.py

WHAT THIS FILE DOES (simple words):
    Draws the page and connects the other files:
        asr_utils.py   (ears)  : your recorded voice  -> text
        llm_chains.py  (brain) : text -> score, feedback, next question
        tts_utils.py   (mouth) : the tutor's text     -> spoken audio

THE ONE RULE TO REMEMBER ABOUT STREAMLIT:
    Streamlit runs this WHOLE file from top to bottom every time anything happens
    (a click, a finished recording, st.rerun()). Anything that must be remembered
    between runs (current stage, topic, scores...) is kept in st.session_state.

THE FLOW (4 stages):   start -> topic -> quiz -> result

WHAT IS NEW IN THIS VERSION:
    * The old add-on microphone (streamlit-mic-recorder) is replaced by Streamlit's
      own built-in widget st.audio_input. The old one sometimes drew a blank WHITE
      BAR instead of the record button after the first answer, so you could not
      answer question 2. The built-in widget does not have that problem.
    * After every recording the widget gets a brand-new key, so it always comes
      back fresh and ready for the next answer.
    * Warnings/errors are saved and shown on the NEXT run (before, they were drawn
      and then wiped instantly by st.rerun(), so you could not read them).
    * A failed LLM call no longer pollutes the conversation history.
    * Text from the LLM is "escaped" before being put into HTML, so symbols like
      < or > can never break the page.
"""

import hashlib
import html

import streamlit as st

import asr_utils
import config
import llm_chains
import tts_utils

st.set_page_config(page_title=f"{config.ASSISTANT_NAME} - Python Voice Tutor", page_icon="🐍", layout="centered")

# ---------------------------------------------------------------------------
# Safety check: the built-in microphone widget needs Streamlit 1.40 or newer.
# ---------------------------------------------------------------------------
if not hasattr(st, "audio_input"):
    st.error(
        "This app needs Streamlit 1.40 or newer (for the built-in microphone). "
        "Close the app, run:  pip install -U streamlit  and start it again."
    )
    st.stop()

# ---------------------------------------------------------------------------
# API-key gate. Every visitor uses THEIR OWN free Groq key, so the app owner's
# key is never stored in the code or on the server. The key lives only in this
# visitor's st.session_state - it is never written to disk or shared.
# If a local .env file has GROQ_API_KEY, it is used automatically and the gate
# is skipped (handy when running on your own laptop).
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
# Styling: card-style boxes for the messages, small labels, title.
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
    """Draws the animated circle. state = "idle" (blue, calm) or "speaking"
    (green, pulsing). It only changes style; it does not react to real sound."""
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
# Session state = the app's memory between reruns
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "stage": "start",           # start -> topic -> quiz -> result
    "topic": "",                # what the student chose to practice
    "history": [],              # hidden transcript, used only as context for the LLM
    "turn_scores": [],          # one score (0-10) per answered question
    "last_assistant_msg": "",   # the tutor text currently shown on screen
    "last_user_msg": "",        # the student's last transcribed answer shown on screen
    "pending_speech": None,     # text waiting to be spoken (None = nothing to say)
    "last_audio_hash": None,    # fingerprint of the last recording we processed
    "final_summary": None,      # the FinalSummary object for the result screen
    "rec_nonce": 0,             # counter that gives the mic widget a fresh key each turn
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def reset_all():
    """Puts every value back to its starting state (the API key is kept).
    rec_nonce is increased instead of reset, so the mic widget always gets a new key."""
    old_nonce = st.session_state.get("rec_nonce", 0)
    for _k, _v in _DEFAULTS.items():
        st.session_state[_k] = _v
    st.session_state["rec_nonce"] = old_nonce + 1


def notify(message: str):
    """Saves a message to be shown as a yellow banner on the NEXT run.
    (If we showed it right now and then called st.rerun(), it would vanish at once.)"""
    st.session_state["last_error"] = message


def finish_turn():
    """Call this when a recording has been fully handled: gives the mic widget a
    brand-new key (so it comes back empty and ready) and reruns the page.
    IMPORTANT: always call it OUTSIDE any try/except block."""
    st.session_state.rec_nonce += 1
    st.rerun()


def get_recording(label: str, stage_name: str):
    """Shows Streamlit's built-in microphone widget.
    Returns the recording as bytes ONLY when a NEW recording has just been made,
    otherwise returns None.
    - The key contains rec_nonce, so after each turn it is a brand-new widget.
    - The hash check is a second safety so the same recording is never handled twice."""
    audio = st.audio_input(label, key=f"{stage_name}_rec_{st.session_state.rec_nonce}")
    if audio is None:
        return None
    data = audio.getvalue()
    if not data:
        return None
    fingerprint = hashlib.md5(data).hexdigest()
    if fingerprint == st.session_state.last_audio_hash:
        return None
    st.session_state.last_audio_hash = fingerprint
    return data


def show_bubble(label: str, text: str, css_class: str):
    """Shows one message box. The text is HTML-escaped so characters such as < > &
    (or a $ sign) from the LLM or from speech recognition can never break the page."""
    safe = html.escape(text or "").replace("$", "&#36;")
    st.markdown(f'<div class="pn-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pn-bubble {css_class}">{safe}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page header, saved warning banner, sidebar
# ---------------------------------------------------------------------------
st.markdown(f'<p class="pn-title">🐍 {config.ASSISTANT_NAME}</p>', unsafe_allow_html=True)
st.markdown('<p class="pn-subtitle">Your voice-based Python quiz buddy</p>', unsafe_allow_html=True)

# Show (and remove) any message saved by notify() during the previous run.
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
# STAGE: start - one button, then the greeting
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
# STAGE: topic - the student says what to study
# ===========================================================================
elif st.session_state.stage == "topic":
    render_orb("speaking" if st.session_state.pending_speech else "idle")
    show_bubble("PyNative", st.session_state.last_assistant_msg, "pn-assistant")

    recording = get_recording("🎤 Press the mic, say your topic, then press stop", "topic")
    if recording is not None:
        with st.spinner("Listening..."):
            heard = asr_utils.transcribe_wav_bytes(recording)   # voice -> text ("" if it failed)

        if not heard.strip():
            notify("I didn't catch that - make sure your browser allowed microphone access, then try again.")
        else:
            with st.spinner("Thinking..."):
                try:
                    check = llm_chains.check_topic(heard)        # is it a Python topic?
                except Exception as e:
                    notify(f"Couldn't reach the LLM: {e}")
                    check = None

                if check and check.is_valid:
                    # Good topic: remember it, make the first question, move to the quiz.
                    st.session_state.topic = heard.strip()
                    history = [{"role": "user", "content": f"(chosen topic: {heard})"}]
                    try:
                        question = llm_chains.generate_first_question(st.session_state.topic)
                    except Exception as e:
                        notify(f"Couldn't reach the LLM: {e}")
                        question = "Let's start simple - what is a Python list?"   # backup question
                    combined = f"{check.message} {question}"
                    history.append({"role": "assistant", "content": combined})
                    st.session_state.history = history
                    st.session_state.last_assistant_msg = combined
                    st.session_state.pending_speech = combined
                    st.session_state.last_user_msg = ""
                    st.session_state.stage = "quiz"
                elif check:
                    # Not a Python topic: the tutor politely asks again.
                    st.session_state.last_assistant_msg = check.message
                    st.session_state.pending_speech = check.message
        finish_turn()

# ===========================================================================
# STAGE: quiz - the main question-and-answer loop
# ===========================================================================
elif st.session_state.stage == "quiz":
    render_orb("speaking" if st.session_state.pending_speech else "idle")
    show_bubble("PyNative", st.session_state.last_assistant_msg, "pn-assistant")
    if st.session_state.last_user_msg:
        show_bubble("You said", st.session_state.last_user_msg, "pn-user")

    recording = get_recording("🎤 Press the mic, speak your answer, then press stop", "quiz")
    if recording is not None:
        with st.spinner("Listening..."):
            heard = asr_utils.transcribe_wav_bytes(recording)   # voice -> text ("" if it failed)

        if not heard.strip():
            notify("I didn't catch that - please try recording again, a little closer to the mic.")
        else:
            st.session_state.last_user_msg = heard
            # Build the history WITH the new answer, but only save it if the LLM succeeds.
            history_with_answer = st.session_state.history + [{"role": "user", "content": heard}]
            with st.spinner("Thinking..."):
                try:
                    result = llm_chains.evaluate_answer(st.session_state.topic, history_with_answer, heard)
                except Exception as e:
                    result = None
                    notify(f"Couldn't reach the LLM: {e}  -  please answer again.")

            if result is not None:
                combined = f"{result.feedback} {result.next_question}"
                st.session_state.turn_scores.append(int(result.score))
                st.session_state.history = history_with_answer + [{"role": "assistant", "content": combined}]
                st.session_state.last_assistant_msg = combined
                st.session_state.pending_speech = combined
        finish_turn()

    st.divider()
    col_end, col_switch = st.columns(2)
    with col_end:
        if st.button("🏁 End & get results", use_container_width=True, type="primary"):
            with st.spinner("Writing your summary..."):
                try:
                    summary = llm_chains.generate_final_summary(
                        st.session_state.topic, st.session_state.history, st.session_state.turn_scores
                    )
                    st.session_state.final_summary = summary
                    st.session_state.pending_speech = summary.summary
                except Exception as e:
                    notify(f"Couldn't reach the LLM: {e}")
                    st.session_state.final_summary = None
            st.session_state.stage = "result"
            st.session_state.rec_nonce += 1
            st.rerun()
    with col_switch:
        if st.button("🔁 Change topic", use_container_width=True):
            msg = "Sure! What Python topic would you like to practice next?"
            st.session_state.last_assistant_msg = msg
            st.session_state.last_user_msg = ""
            st.session_state.pending_speech = msg
            st.session_state.stage = "topic"
            st.session_state.rec_nonce += 1
            st.rerun()

# ===========================================================================
# STAGE: result - final score + summary
# ===========================================================================
elif st.session_state.stage == "result":
    render_orb("speaking" if st.session_state.pending_speech else "idle")

    if st.session_state.final_summary:
        st.markdown(
            f'<p style="text-align:center; font-size:2.2rem; font-weight:800;">'
            f'{st.session_state.final_summary.overall_score:.1f} / 10</p>',
            unsafe_allow_html=True,
        )
        show_bubble("PyNative", st.session_state.final_summary.summary, "pn-assistant")
    else:
        st.warning("Couldn't generate a summary - see the message above.")

    if st.button("🔄 Start a new session", use_container_width=True, type="primary"):
        reset_all()
        st.rerun()

# ---------------------------------------------------------------------------
# LAST THING EVERY RUN: speak anything new, exactly once.
# pending_speech is cleared BEFORE speaking, so a sentence can never repeat.
# ---------------------------------------------------------------------------
if st.session_state.pending_speech:
    text_to_speak = st.session_state.pending_speech
    st.session_state.pending_speech = None
    try:
        mp3_bytes = tts_utils.text_to_mp3_bytes(text_to_speak)
    except Exception as e:           # tts_utils should never raise, but just in case
        print(f"[TTS error in app]: {e}")
        mp3_bytes = b""
    if mp3_bytes:
        st.audio(mp3_bytes, format="audio/mp3", autoplay=True)

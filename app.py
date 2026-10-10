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

WHAT IS NEW IN THIS VERSION (interface):
    * Clear visual hierarchy: the current QUESTION is the biggest element on the
      page; feedback (with a coloured score badge) sits above it, smaller; what
      the student said is shown small and muted.
    * The orb now pulses ONLY while the answer is really being spoken. The audio
      is played inside the orb component itself, so JavaScript can listen to the
      audio's play/ended events (before, the orb pulsed whenever there was text
      waiting to be spoken, which did not match the sound at all).
    * A status line under the orb ("Speaking..." / "Your turn") and a status box
      while an answer is processed (Transcribing -> Thinking).
    * Timings per pipeline step (ASR / LLM / TTS) are printed to the terminal
      ("[TIMING] ...") - useful for the latency section of the report.
    * A score per question and an expandable list of previous questions.

THE TUTOR'S NAME FOLLOWS THE VOICE:
    The name is taken from the selected voice ("Ryan (UK, male)" -> "Ryan"). The page
    says "Meet Ryan", the tutor says "Hi, I am Ryan", and the AI is told its name is
    Ryan too (via session_state["tutor_name"], read in llm_chains.py). Picking another
    voice speaks a short introduction automatically.

WHAT IS NEW IN THIS VERSION (speech settings in the sidebar):
    * Tutor voice        : choose between 9 neural voices (US, UK, Australia, India),
                           with a "Preview voice" button.
    * Speaking speed     : slider from -30% (slower) to +40% (faster).
    * Speech recognition model : tiny / base / small Whisper (speed versus accuracy).
    * Python vocabulary hint   : on/off switch for Whisper's initial_prompt, to show
                           how much it helps with words like tuple, dict, elif.
"""

import base64
import hashlib
import html
import json
import time

import streamlit as st

import asr_utils
import config
import llm_chains
import tts_utils

st.set_page_config(page_title="Python Voice Tutor", page_icon="🐍", layout="centered")

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
# The tutor's name = the name of the selected voice. When the student picks a
# voice in the sidebar, Streamlit already stores the choice in session_state at
# the start of the rerun, so reading it here (above the sidebar code) gives the
# NEW name straight away. llm_chains.py reads "tutor_name" for the AI's persona.
# ---------------------------------------------------------------------------
tutor_name = config.tutor_name_from_label(st.session_state.get("voice_label", config.DEFAULT_VOICE_LABEL)) or config.ASSISTANT_NAME
st.session_state["tutor_name"] = tutor_name

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
    st.markdown(f'<p style="text-align:center; font-size:2.2rem; font-weight:800; margin-bottom:0;">Meet {tutor_name}</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#9aa5b1;">Your voice-based Python tutor</p>', unsafe_allow_html=True)
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
# Styling. Three levels of importance, from big to small:
#   1. .pn-question  - the thing the student must respond to (largest)
#   2. .pn-feedback  - the tutor's feedback on the last answer, with a score badge
#   3. .pn-said      - what the speech recogniser heard (small, muted)
# Colours are semi-transparent so they work in both light and dark mode.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .pn-title { text-align: center; font-size: 2.1rem; font-weight: 800; line-height: 1.2; }
    .pn-subtitle { text-align: center; color: #9aa5b1; margin-top: 0; margin-bottom: 0.4rem; }

    .pn-label { font-weight: 700; font-size: 0.75rem; letter-spacing: 0.06em;
        text-transform: uppercase; opacity: 0.6; margin: 14px 0 6px 0; }

    .pn-question {
        font-size: 1.3rem; font-weight: 600; line-height: 1.45;
        padding: 20px 22px; border-radius: 16px; margin-bottom: 8px;
        background: rgba(34, 197, 94, 0.10); border: 1px solid rgba(34, 197, 94, 0.40);
    }

    .pn-feedback {
        display: flex; gap: 12px; align-items: flex-start;
        font-size: 1rem; line-height: 1.5; padding: 12px 16px; border-radius: 12px;
        background: rgba(148, 163, 184, 0.10); border-left: 3px solid rgba(148, 163, 184, 0.6);
    }
    .pn-badge { flex: none; font-weight: 800; font-size: 0.9rem; color: #fff;
        border-radius: 999px; padding: 2px 10px; margin-top: 1px; }
    .pn-good { background: #16a34a; }
    .pn-mid  { background: #d97706; }
    .pn-low  { background: #dc2626; }

    .pn-said { font-size: 0.92rem; opacity: 0.7; margin: 18px 0 10px 2px; }
    .pn-said b { font-weight: 600; font-style: normal; }


    .pn-turn { font-size: 0.92rem; line-height: 1.45; padding: 8px 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.25); }
    .pn-turn:last-child { border-bottom: none; }

    /* Bigger microphone (and stop/play) icon in Streamlit's st.audio_input widget */
    [data-testid="stAudioInputActionButton"] svg { width: 2.2rem; height: 2.2rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state = the app's memory between reruns
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "stage": "start",           # start -> topic -> quiz -> result
    "topic": "",                # what the student chose to practice
    "history": [],              # hidden transcript, used only as context for the LLM
    "turn_scores": [],          # one score (0-10) per answered question
    "turns": [],                # per question: {"question", "answer", "score", "feedback"} (for the UI)
    "current_question": "",     # the question the student must answer now (shown big)
    "last_feedback": "",        # feedback on the previous answer (or the topic intro line)
    "last_score": None,         # score of the previous answer (None = no badge)
    "last_assistant_msg": "",   # tutor text for stages without a question (greeting, topic)
    "last_user_msg": "",        # the student's last transcribed answer shown on screen
    "pending_speech": None,     # text waiting to be spoken (None = nothing to say)
    "speech_id": 0,             # counts spoken messages, so the orb always reloads new audio
    "timings": {},              # seconds per pipeline step for the last turn: asr / llm / tts
    "last_audio_hash": None,    # fingerprint of the last recording we processed
    "final_summary": None,      # the FinalSummary object for the result screen
    "rec_nonce": 0,             # counter that gives the mic widget a fresh key each turn
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def reset_all():
    """Puts every value back to its starting state (the API key is kept).
    rec_nonce is increased instead of reset, so the mic widget always gets a new key.
    Lists/dicts are copied, so a reset never shares one list between sessions."""
    old_nonce = st.session_state.get("rec_nonce", 0)
    for _k, _v in _DEFAULTS.items():
        st.session_state[_k] = _v.copy() if isinstance(_v, (list, dict)) else _v
    st.session_state["rec_nonce"] = old_nonce + 1


def notify(message: str):
    """Saves a message to be shown as a yellow banner on the NEXT run.
    (If we showed it right now and then called st.rerun(), it would vanish at once.)"""
    st.session_state["last_error"] = message


def say(text: str):
    """Queues text to be spoken at the end of the next run."""
    st.session_state.pending_speech = text


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


def esc(text: str) -> str:
    """HTML-escapes text so characters such as < > & (or a $ sign) from the LLM or
    from speech recognition can never break the page."""
    return html.escape(text or "").replace("$", "&#36;")


def score_class(score: int) -> str:
    """Colour of the score badge: green 8-10, orange 5-7, red 0-4."""
    return "pn-good" if score >= 8 else "pn-mid" if score >= 5 else "pn-low"


# ---------------------------------------------------------------------------
# Small drawing helpers, one per level of the visual hierarchy
# ---------------------------------------------------------------------------
def show_question(label: str, text: str):
    """Level 1: the question (or prompt) the student has to respond to."""
    st.markdown(f'<div class="pn-label">{esc(label)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pn-question">{esc(text)}</div>', unsafe_allow_html=True)


def show_feedback(text: str, score):
    """Level 2: feedback on the last answer, with a coloured score badge
    (no badge when score is None, e.g. the intro line after choosing a topic)."""
    if not text:
        return
    badge = f'<span class="pn-badge {score_class(score)}">{score}/10</span>' if score is not None else ""
    st.markdown(f'<div class="pn-feedback">{badge}<div>{esc(text)}</div></div>', unsafe_allow_html=True)


def show_said(text: str):
    """Level 3: what the speech recogniser heard (small and muted)."""
    if text:
        st.markdown(f'<div class="pn-said"><b>You said:</b> “{esc(text)}”</div>', unsafe_allow_html=True)


def show_previous_turns(expanded: bool = False):
    """An expandable list of every answered question with its score."""
    turns = st.session_state.turns
    if not turns:
        return
    with st.expander(f"Previous questions ({len(turns)})", expanded=expanded):
        rows = []
        for i, t in enumerate(turns, start=1):
            rows.append(
                f'<div class="pn-turn"><span class="pn-badge {score_class(t["score"])}">{t["score"]}/10</span> '
                f'<b>Q{i}.</b> {esc(t["question"])}<br>'
                f'<span style="opacity:.7">You said: “{esc(t["answer"])}”</span><br>'
                f'{esc(t["feedback"])}</div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)


def render_orb(slot, mp3_bytes: bytes = b"", idle_label: str = ""):
    """Draws the orb into `slot` (an st.empty() placeholder).

    If mp3_bytes is given, the audio is played INSIDE this component, and the orb
    pulses only between the audio's 'play' and 'ended' events - so the animation
    matches the real sound. If the browser blocks autoplay, a 'Play answer' button
    appears instead. A small 'Replay' button lets the student hear it again.
    speech_id is put in the HTML so a repeated sentence still reloads (and plays)."""
    audio_tag = ""
    if mp3_bytes:
        b64 = base64.b64encode(mp3_bytes).decode("ascii")
        audio_tag = f'<audio id="pn-audio" src="data:audio/mp3;base64,{b64}" preload="auto"></audio>'

    with slot:
        st.components.v1.html(f"""
        <!-- speech {st.session_state.speech_id} -->
        <div style="display:flex; flex-direction:column; align-items:center; padding:44px 0 8px 0;
                    font-family: 'Source Sans Pro', sans-serif;">
          <div id="pn-orb" class="orb"></div>
          <div id="pn-bars" class="bars"><span></span><span></span><span></span><span></span><span></span></div>
          <div id="pn-status" class="status">{html.escape(idle_label)}</div>
          <button id="pn-replay" class="replay" style="display:none;">🔊 Replay</button>
        </div>
        {audio_tag}
        <style>
          body {{ margin:0; background: transparent; }}
          .orb {{
            width: 110px; height: 110px; border-radius: 50%;
            background: radial-gradient(circle at 32% 30%, #a5c9ff, #3b82f6);
            box-shadow: 0 0 30px rgba(59,130,246,0.40);
            transition: background .3s, box-shadow .3s;
          }}
          .orb.speaking {{
            background: radial-gradient(circle at 32% 30%, #b9fbc0, #22c55e);
            box-shadow: 0 0 50px rgba(34,197,94,0.6);
            animation: pulse 0.55s ease-in-out infinite;
          }}
          @keyframes pulse {{ 0%,100%{{transform:scale(1);}} 50%{{transform:scale(1.14);}} }}
          .bars {{ display:flex; gap:6px; height:22px; margin-top:10px; opacity:0; transition:opacity .2s; }}
          .bars.speaking {{ opacity:1; }}
          .bars span {{ width:6px; border-radius:3px; background:#22c55e; animation: eq 0.9s ease-in-out infinite; }}
          .bars span:nth-child(1){{animation-delay:0s;}} .bars span:nth-child(2){{animation-delay:.15s;}}
          .bars span:nth-child(3){{animation-delay:.3s;}} .bars span:nth-child(4){{animation-delay:.1s;}}
          .bars span:nth-child(5){{animation-delay:.25s;}}
          @keyframes eq {{ 0%,100%{{height:6px;}} 50%{{height:20px;}} }}
          .status {{ color:#9aa5b1; font-size:14px; margin-top:4px; min-height:18px; }}
          .replay {{ margin-top:6px; font-size:12px; color:#9aa5b1; background:transparent;
                     border:1px solid rgba(148,163,184,.5); border-radius:999px; padding:2px 10px; cursor:pointer; }}
        </style>
        <script>
          const audio  = document.getElementById("pn-audio");
          const orb    = document.getElementById("pn-orb");
          const bars   = document.getElementById("pn-bars");
          const status = document.getElementById("pn-status");
          const replay = document.getElementById("pn-replay");
          const idleText = {json.dumps(idle_label)};

          function setSpeaking(on) {{
            orb.classList.toggle("speaking", on);
            bars.classList.toggle("speaking", on);
            status.textContent = on ? "Speaking..." : idleText;
          }}

          if (audio) {{
            replay.style.display = "inline-block";
            audio.addEventListener("play",  () => setSpeaking(true));
            audio.addEventListener("pause", () => setSpeaking(false));
            audio.addEventListener("ended", () => setSpeaking(false));
            replay.addEventListener("click", () => {{ audio.currentTime = 0; audio.play(); }});
            audio.play().catch(() => {{
              // The browser blocked autoplay (can happen on the very first answer).
              replay.textContent = "▶ Play answer";
              status.textContent = "Click to hear the answer";
            }});
          }}
        </script>
        """, height=240)   # extra room above the orb so its glow is never cut off


def log_timings():
    """Prints how long each pipeline step took in the last turn to the TERMINAL,
    e.g. '[TIMING] ASR 1.2 s | LLM 0.9 s | TTS 0.6 s'. Not shown on the page (it
    distracted users), but handy for the latency section of the report."""
    t = st.session_state.timings
    names = [("asr", "ASR"), ("llm", "LLM"), ("tts", "TTS")]
    parts = [f"{label} {t[key]:.1f} s" for key, label in names if key in t]
    if parts:
        print("[TIMING] " + " | ".join(parts))


# ---------------------------------------------------------------------------
# Page header, saved warning banner, sidebar
# ---------------------------------------------------------------------------
# <div> instead of <p>: Streamlit's own <p> styling would override our font size.
st.markdown(f'<div class="pn-title">Meet {tutor_name}</div>', unsafe_allow_html=True)
st.markdown('<div class="pn-subtitle">Your voice-based Python tutor</div>', unsafe_allow_html=True)

# Status text under the orb when nothing is being spoken, per stage.
_IDLE_LABELS = {
    "start": "Press Start to begin",
    "topic": "Your turn - say a topic",
    "quiz": "Your turn - answer the question",
    "result": "Session finished",
}

# Placeholder right under the header for the orb. The calm blue orb is drawn
# straight away, so it is visible while the page is still working. At the END
# of the run, once the speech audio is ready, it is replaced by the speaking
# version (see the bottom of this file).
orb_slot = st.empty()
# While speech is still being generated, don't say "Your turn" yet.
_first_label = "Getting ready to speak..." if st.session_state.pending_speech else _IDLE_LABELS.get(st.session_state.stage, "")
render_orb(orb_slot, b"", _first_label)

# Show (and remove) any message saved by notify() during the previous run.
_err = st.session_state.pop("last_error", None)
if _err:
    st.warning(_err)

with st.sidebar:
    avg = sum(st.session_state.turn_scores) / len(st.session_state.turn_scores) if st.session_state.turn_scores else None
    st.metric("Session average", f"{avg:.1f} / 10" if avg is not None else "—")
    st.caption(f"Questions answered: {len(st.session_state.turn_scores)}")
    st.divider()

    # ---------------- Speech settings (new) ----------------
    # Each widget has a key, so Streamlit remembers the choice between reruns
    # (and "Restart entire session" does not reset it).
    st.markdown("**Speech settings**")

    # 1) Tutor voice
    _voice_labels = list(config.VOICE_OPTIONS.keys())
    _voice_index = _voice_labels.index(config.DEFAULT_VOICE_LABEL) if config.DEFAULT_VOICE_LABEL in _voice_labels else 0
    voice_label = st.selectbox("Tutor voice", _voice_labels, index=_voice_index, key="voice_label")
    selected_voice = config.VOICE_OPTIONS.get(voice_label, config.TTS_VOICE)

    # When the student picks a DIFFERENT voice, the tutor introduces itself with
    # its new name. (Not on the very first run: browsers block sound before the
    # first click anyway.)
    _previous_voice = st.session_state.get("last_voice_label")
    st.session_state["last_voice_label"] = voice_label
    if _previous_voice is not None and _previous_voice != voice_label:
        say(f"Hi, I am {tutor_name}.")
        st.rerun()

    # 2) Speaking speed (percent; 0 = normal). edge-tts wants text like "+10%" or "-20%".
    speech_rate_pct = st.slider(
        "Speaking speed (% faster / slower)",
        min_value=config.TTS_RATE_MIN, max_value=config.TTS_RATE_MAX,
        value=config.TTS_RATE_PERCENT, step=config.TTS_RATE_STEP, key="speech_rate_pct",
    )
    selected_rate = f"{int(speech_rate_pct):+d}%"

    # Preview: the tutor says one sentence with the chosen voice and speed.
    if st.button("🔊 Preview voice", use_container_width=True):
        say(f"Hi, I am {tutor_name}. This is how I sound at this speed.")
        st.rerun()

    # 3) Whisper model size (speed versus accuracy)
    _size_options = list(config.WHISPER_MODEL_OPTIONS)
    _size_index = _size_options.index(config.WHISPER_MODEL_SIZE) if config.WHISPER_MODEL_SIZE in _size_options else 0
    whisper_size = st.selectbox(
        "Speech recognition model", _size_options, index=_size_index, key="whisper_size",
        format_func=lambda size: config.WHISPER_MODEL_LABELS.get(size, size),
        help="Bigger models understand speech better but need more time and memory. "
             "The first time a size is used it is downloaded (internet needed).",
    )

    # 4) Python vocabulary hint for Whisper (initial_prompt)
    use_vocab_prompt = st.toggle(
        "Python vocabulary hint", value=config.USE_ASR_PROMPT, key="use_vocab_prompt",
        help="Gives Whisper a list of Python words (tuple, dict, elif, lambda...) so it "
             "recognises technical terms better. Turn it off to compare.",
    )

    # Load the chosen Whisper model now (only when the choice changed), so the
    # next answer is not delayed. preload_model never raises; if it fails the
    # app still works because asr_utils falls back to the default model.
    if st.session_state.get("preloaded_whisper_size") != whisper_size:
        with st.spinner(f"Loading speech model '{whisper_size}'..."):
            _model_ok = asr_utils.preload_model(whisper_size)
        st.session_state["preloaded_whisper_size"] = whisper_size
        if not _model_ok:
            st.warning(f"Could not load the '{whisper_size}' model (internet needed for the first download). "
                       f"The default '{config.WHISPER_MODEL_SIZE}' model will be used instead.")

    st.caption(f"ASR: faster-whisper ({whisper_size})")
    st.caption(f"LLM: Groq {llm_chains.resolve_model_name(llm_chains.get_api_key())}")
    st.caption(f"TTS: edge-tts ({selected_voice}, {selected_rate})")
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
    st.write("")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("▶️ Start Quiz", use_container_width=True, type="primary"):
            greeting = (
                f"Hey there! I am  your personal Python tutor. "
                "Which Python topic would you like to practice today? You can also just say "
                "'mixed' for a general quiz."
            )
            st.session_state.last_assistant_msg = greeting
            say(greeting)
            st.session_state.stage = "topic"
            st.rerun()

# ===========================================================================
# STAGE: topic - the student says what to study
# ===========================================================================
elif st.session_state.stage == "topic":
    show_question(tutor_name, st.session_state.last_assistant_msg)

    recording = get_recording("Press the microphone, say your topic, then press stop", "topic")
    if recording is not None:
        timings = {}
        with st.status("Transcribing what you said...") as status:
            start = time.perf_counter()
            heard = asr_utils.transcribe_wav_bytes(recording, whisper_size, use_vocab_prompt)   # voice -> text ("" if it failed)
            timings["asr"] = time.perf_counter() - start

            if not heard.strip():
                notify("I didn't catch that - make sure your browser allowed microphone access, then try again.")
            else:
                status.update(label=f'Thinking about "{heard.strip()}"...')
                start = time.perf_counter()
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
                    st.session_state.current_question = question
                    st.session_state.last_feedback = check.message   # intro line, shown without a badge
                    st.session_state.last_score = None
                    st.session_state.last_user_msg = ""
                    say(combined)
                    st.session_state.stage = "quiz"
                elif check:
                    # Not a Python topic: the tutor politely asks again.
                    st.session_state.last_assistant_msg = check.message
                    say(check.message)
                timings["llm"] = time.perf_counter() - start
            status.update(label="Done", state="complete")
        st.session_state.timings = timings
        finish_turn()

# ===========================================================================
# STAGE: quiz - the main question-and-answer loop
# Layout from small to big: what you said -> feedback + score -> NEXT QUESTION
# ===========================================================================
elif st.session_state.stage == "quiz":
    show_said(st.session_state.last_user_msg)
    show_feedback(st.session_state.last_feedback, st.session_state.last_score)
    show_question(f"Question {len(st.session_state.turns) + 1}", st.session_state.current_question)

    recording = get_recording("Press the microphone, speak your answer, then press stop", "quiz")
    if recording is not None:
        timings = {}
        with st.status("Transcribing your answer...") as status:
            start = time.perf_counter()
            heard = asr_utils.transcribe_wav_bytes(recording, whisper_size, use_vocab_prompt)   # voice -> text ("" if it failed)
            timings["asr"] = time.perf_counter() - start

            if not heard.strip():
                notify("I didn't catch that - please try recording again, a little closer to the mic.")
            else:
                status.update(label="Thinking about your answer...")
                # Build the history WITH the new answer, but only save it if the LLM succeeds.
                history_with_answer = st.session_state.history + [{"role": "user", "content": heard}]
                start = time.perf_counter()
                try:
                    result = llm_chains.evaluate_answer(st.session_state.topic, history_with_answer, heard)
                except Exception as e:
                    result = None
                    notify(f"Couldn't reach the LLM: {e}  -  please answer again.")
                timings["llm"] = time.perf_counter() - start

                if result is not None:
                    score = int(result.score)
                    combined = f"{result.feedback} {result.next_question}"
                    st.session_state.turns.append({
                        "question": st.session_state.current_question,
                        "answer": heard,
                        "score": score,
                        "feedback": result.feedback,
                    })
                    st.session_state.turn_scores.append(score)
                    st.session_state.history = history_with_answer + [{"role": "assistant", "content": combined}]
                    st.session_state.last_user_msg = heard
                    st.session_state.last_feedback = result.feedback
                    st.session_state.last_score = score
                    st.session_state.current_question = result.next_question
                    say(combined)
            status.update(label="Done", state="complete")
        st.session_state.timings = timings
        finish_turn()

    show_previous_turns()

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
                    say(summary.summary)
                except Exception as e:
                    notify(f"Couldn't reach the LLM: {e}")
                    st.session_state.final_summary = None
            st.session_state.timings = {}
            st.session_state.stage = "result"
            st.session_state.rec_nonce += 1
            st.rerun()
    with col_switch:
        if st.button("🔁 Change topic", use_container_width=True):
            msg = "Sure! What Python topic would you like to practice next?"
            st.session_state.last_assistant_msg = msg
            st.session_state.last_user_msg = ""
            st.session_state.last_feedback = ""
            st.session_state.last_score = None
            say(msg)
            st.session_state.timings = {}
            st.session_state.stage = "topic"
            st.session_state.rec_nonce += 1
            st.rerun()

# ===========================================================================
# STAGE: result - final score + summary + review of every question
# ===========================================================================
elif st.session_state.stage == "result":
    if st.session_state.final_summary:
        st.markdown(
            f'<p style="text-align:center; font-size:2.6rem; font-weight:800; margin-bottom:0;">'
            f'{st.session_state.final_summary.overall_score:.1f} / 10</p>'
            f'<p style="text-align:center; opacity:.6; margin-top:0;">'
            f'{len(st.session_state.turns)} questions answered</p>',
            unsafe_allow_html=True,
        )
        show_question("Summary", st.session_state.final_summary.summary)
    else:
        st.warning("Couldn't generate a summary - see the message above.")

    show_previous_turns(expanded=True)

    if st.button("🔄 Start a new session", use_container_width=True, type="primary"):
        reset_all()
        st.rerun()

# ---------------------------------------------------------------------------
# LAST THING EVERY RUN: speak anything new (exactly once), then draw the orb
# (with the audio) into the placeholder at the top of the page.
# pending_speech is cleared BEFORE speaking, so a sentence can never repeat.
# ---------------------------------------------------------------------------
mp3_bytes = b""
if st.session_state.pending_speech:
    text_to_speak = st.session_state.pending_speech
    st.session_state.pending_speech = None
    start = time.perf_counter()
    try:
        mp3_bytes = tts_utils.text_to_mp3_bytes(text_to_speak, selected_voice, selected_rate)
    except Exception as e:           # tts_utils should never raise, but just in case
        print(f"[TTS error in app]: {e}")
        mp3_bytes = b""
    st.session_state.timings["tts"] = time.perf_counter() - start
    st.session_state.speech_id += 1
    log_timings()

# Redraw only if we tried to speak this run (with audio, or - if TTS failed -
# without, so "Getting ready to speak..." never stays on screen).
if _first_label == "Getting ready to speak...":
    render_orb(orb_slot, mp3_bytes, _IDLE_LABELS.get(st.session_state.stage, ""))
# PyNative — Voice-Based Python Quiz Tutor

Click Start, tell it what Python topic you're practicing, answer questions
out loud, get real (not canned) spoken feedback and a score each time, and
get a final summary whenever you're ready.

---

## Files

| File | What it does |
|---|---|
| `config.py` | Every tunable setting (model sizes, voice, API key loading) |
| `asr_utils.py` | Speech-to-text — transcribes your recorded answers using local `faster-whisper` |
| `tts_utils.py` | Text-to-speech — turns PyNative's responses into spoken audio using `edge-tts` |
| `llm_chains.py` | All the LLM reasoning (LangChain + Groq) — checking your topic, generating questions, scoring answers, final summary |
| `app.py` | The Streamlit UI and the Start → Topic → Quiz → Result flow |

No classes anywhere — every piece of logic is a plain function, grouped by
what it does.

---

## 0. Environment check (do this if you've had `av`/`metadata_errors` crashes)

If you've seen either of these errors:
```
TypeError: open() got an unexpected keyword argument 'metadata_errors'
file does not start with RIFF id
```
it almost always means Python commands are actually resolving to a conda
`base` environment instead of your venv, so `pip install -r requirements.txt`
updated one environment while `streamlit run` is executing from a
different one with an old, incompatible `av` package already installed.

Run this with your venv active to confirm:
```bash
which python
```
If the path does **not** contain your project's `venv` folder, run:
```bash
conda deactivate
source venv/bin/activate
which python          # confirm it NOW points into venv/
pip install --force-reinstall --no-cache-dir faster-whisper
python -m streamlit run app.py
```
Always launch with `python -m streamlit run app.py` (not bare `streamlit
run app.py`) - the `-m` form guarantees it uses whichever Python is
currently active, sidestepping this whole class of PATH confusion.

---

## 1. Setup

You need a free Groq API key: https://console.groq.com/keys

Two ways to provide it (pick one):
- **Easiest:** do nothing now. When the app opens it asks you to paste the key.
- **Optional:** copy `.env.example` to `.env` and paste the key there; the app
  then skips the key prompt on your own machine.

Install Python dependencies (no separate `ffmpeg` install is needed - `av`
bundles its own ffmpeg libraries):
```bash
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Run it

```bash
python -m streamlit run app.py
```

Flow:
1. Click **▶️ Start Quiz**. PyNative greets you and asks what topic you
   want out loud.
2. Click **🎤 Tell me the topic**, say e.g. *"lists and tuples"* or
   *"mixed"*, click Stop.
3. It asks a question out loud. Click **🎤 Answer**, speak your answer,
   click Stop — get specific spoken feedback + a 0–10 score + the next
   question, automatically, on repeat.
4. **🏁 End & get results** anytime for a final score + summary of what to
   focus on. **🔁 Change topic** to keep going on something else without
   ending the session (your score keeps accumulating).

---

## 3. How it works, file by file

**`asr_utils.py`** — the browser records audio and hands the raw bytes to
`transcribe_wav_bytes()`, which passes them straight to faster-whisper's
own decoder (built on the `av` library, which wraps ffmpeg). That decoder
auto-detects the actual format — important, since despite the recorder
library's name the bytes aren't always plain WAV, browsers often record
WebM/Opus natively — and handles resampling to the 16kHz rate faster-
whisper needs, internally. All transcription runs locally on your own
CPU, free, no API calls. See section 0 above if this throws a
`metadata_errors` error — it means your environment has a stale `av`
version installed.

**`tts_utils.py`** — `edge-tts` generates natural speech as mp3 bytes
entirely in memory. Since `edge-tts` is async and the rest of the app
isn't, one dedicated background thread runs a permanent event loop just
for this, so TTS calls never conflict with anything else using asyncio
(including Streamlit itself).

**`llm_chains.py`** — this is the "brain." Each function builds a
`ChatPromptTemplate` (a reusable prompt with `{placeholders}`), pipes it
into a Groq model via LangChain's `prompt | model` syntax, and forces the
response into an exact Pydantic schema with `.with_structured_output()` —
so you get back real Python objects (`result.score`, `result.feedback`,
`result.next_question`), never a raw string you'd have to parse yourself.
`resolve_model_name()` asks Groq directly which models your account can
use and picks automatically, rather than hardcoding a name that might not
be available on your specific key.

**`app.py`** — ties it together. Streamlit re-runs this whole script on
every click, so anything that needs to survive between clicks lives in
`st.session_state` (the topic, the running score, the full hidden
conversation history used as LLM context). Only the *current* turn is
shown on screen (one box for PyNative, one for your last answer) — it's
replaced each turn, not appended, so there's no scrolling chat log.

---

## 4. Troubleshooting

- **App keeps asking for the API key / "key was rejected"**: re-copy the key from
  https://console.groq.com/keys (no spaces). If you use a `.env` file it must be
  named exactly `.env` (not `.env.example`), next to `app.py`.
- **`streamlit` module/import errors**: you likely have conda's `base`
  environment shadowing your venv — run
  `conda deactivate && source venv/bin/activate`, then
  `python -m streamlit run app.py`.
- **"I didn't catch that" / empty transcription**: check your browser
  actually granted microphone permission (site info icon in the address
  bar), and make sure you're speaking *before* clicking Stop, not after.
- **"[ASR ERROR]" in the terminal, or transcription stays empty**: the full
  traceback is now always printed to the terminal — check it first. A
  `metadata_errors` TypeError specifically means section 0's environment
  check applies to you.
- **LLM errors, or the flow seeming to silently stall**: any LLM call
  failure is now shown as a persistent yellow warning banner near the top
  of the page (it used to flash and disappear before you could read it —
  fixed), and the quiz is designed to keep going with a sensible fallback
  rather than get stuck, so "nothing happening" shouldn't occur anymore.
  If it does, the banner will show the real underlying error.
- **No audio plays**: some browsers block autoplay on the very first
  interaction of a session — click anywhere on the page once, then try
  again; every play after that should work normally.
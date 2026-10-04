"""
llm_chains.py
-------------
All the "thinking" for the quiz tutor lives here, built with LangChain.

The shape of every function below:
  1. A Pydantic model defines the exact JSON structure we want back from
     the LLM (e.g. {"score": 7, "feedback": "...", "next_question": "..."}).
     This is LangChain's "structured output" feature - the model is forced
     to return exactly that shape via tool-calling under the hood, instead
     of us parsing free text with regex.
  2. A ChatPromptTemplate defines a "system" message (the LLM's persistent
     role/behaviour) plus a "human" message template with {placeholders}.
  3. `prompt | structured_llm` is LangChain's pipe syntax: "take the
     prompt's output and feed it into the model." `.invoke({...})` runs
     the chain once and returns an already-parsed Pydantic object.
"""

import random
import time

import streamlit as st
from groq import Groq as GroqClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

import config


# =============================================================================
# Model resolution
# =============================================================================
# Groq's exact set of AVAILABLE models can differ per account (region,
# account age, etc.), so a model name that works in Groq's own docs can
# still 404 as "not found or no access" on a specific key. Rather than
# hardcoding one name and hoping, we ask Groq's API which models THIS key
# can actually use, and pick the best match from a preference list - this
# adapts automatically instead of breaking the next time a name changes.
# =============================================================================

_PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]


def get_api_key() -> str:
    """Returns the Groq key for THIS visitor. Order of priority:
      1. the key they typed into the app (stored per-visitor in
         st.session_state, so it is never shared between visitors)
      2. a key from a local .env file (convenient when running locally)
    """
    return st.session_state.get("groq_api_key") or config.GROQ_API_KEY or ""


def validate_api_key(api_key: str) -> tuple[bool, str]:
    """Makes one cheap, free call (listing models) to check the key works."""
    if not api_key or not api_key.strip():
        return False, "Please paste your Groq API key."
    try:
        GroqClient(api_key=api_key.strip()).models.list()
        return True, ""
    except Exception as e:
        return False, f"That key was rejected by Groq: {e}"


# cache_resource caches PER ARGUMENT VALUE, so each different key gets its
# own cached answer - one visitor's key never leaks to another visitor.
@st.cache_resource(show_spinner=False)
def resolve_model_name(api_key: str) -> str:
    try:
        client = GroqClient(api_key=api_key)
        available = {m.id for m in client.models.list().data}
    except Exception as e:
        print(f"[warning] Couldn't list Groq models ({e}); using config.LLM_MODEL_FALLBACK.")
        return config.LLM_MODEL_FALLBACK

    for preferred in _PREFERRED_MODELS:
        if preferred in available:
            return preferred
    if available:
        chosen = sorted(available)[0]
        print(f"[warning] None of the preferred models are available; using '{chosen}' instead.")
        return chosen
    return config.LLM_MODEL_FALLBACK


# NOTE: the two builders below are deliberately NOT cached any more. They
# used to be @st.cache_resource, which shares one object between ALL
# visitors - that would make everybody use the first visitor's API key.
# Creating a ChatGroq object is cheap (no network call), so no caching.
def get_llm() -> ChatGroq:
    """Standard-temperature model, used for evaluating answers and
    summaries - we want fairly consistent, focused judgment here."""
    key = get_api_key()
    return ChatGroq(model=resolve_model_name(key), api_key=key, temperature=0.4)


def get_question_llm() -> ChatGroq:
    """Higher-temperature model, used ONLY for generating questions. A real
    bug we hit during development: at low temperature, the model kept
    defaulting to the exact same "textbook" first question for a given
    topic every time. Higher temperature (more randomness) plus the random
    "angle" hint below fixes that."""
    key = get_api_key()
    return ChatGroq(model=resolve_model_name(key), api_key=key, temperature=0.9)


_QUESTION_ANGLES = [
    "Ask it as a 'what's the difference between X and Y' style question.",
    "Ask it as a 'what would happen if...' conceptual scenario question.",
    "Ask it as a 'when would you use X instead of Y' question.",
    "Ask it as a direct, simple definition/explanation question.",
    "Ask it as a short real-world example-based question.",
]


# =============================================================================
# Persona and structured output schemas
# =============================================================================

_PERSONA = f"""You are {config.ASSISTANT_NAME}, a friendly, encouraging voice-based Python \
programming tutor. You ONLY discuss Python programming topics. If the student says anything \
unrelated to Python, politely redirect them back to Python in your response. Keep every response \
spoken-friendly: no markdown, no code blocks, no bullet points, no asterisks - plain conversational \
sentences, since everything you say is read aloud by text-to-speech."""


class TopicCheck(BaseModel):
    is_valid: bool = Field(description=(
        "True if this is a genuine Python topic (e.g. 'lists and tuples', 'recursion', "
        "'decorators'), OR a general request like 'mixed quiz' / 'anything' / 'general Python'. "
        "False if it's unrelated to Python entirely."
    ))
    message: str = Field(description=(
        "If is_valid is True: one short, friendly spoken line acknowledging the topic "
        "(e.g. 'Great, let's practice lists and tuples!'). "
        "If is_valid is False: one short, friendly spoken line redirecting them "
        "(e.g. 'I can only help with Python topics - what Python topic would you like to practice?')."
    ))


class FirstQuestion(BaseModel):
    question: str = Field(description=(
        "One clear, beginner-to-intermediate oral quiz question about the topic. Vary your "
        "phrasing and specific angle every time - never default to the single most obvious, "
        "generic textbook question. Spoken-friendly, no markdown."
    ))


class QuizTurn(BaseModel):
    score: int = Field(ge=0, le=10, description=(
        "How well the student's answer addressed the question: 0 (didn't attempt / completely "
        "wrong / off-topic nonsense) to 10 (excellent, complete, accurate)."
    ))
    feedback: str = Field(description=(
        "2-3 short, HONEST, spoken-friendly sentences of SPECIFIC feedback on what they actually "
        "said - never generic praise. If their answer was off-topic, nonsensical, or didn't "
        "attempt the question at all (e.g. they talked about something unrelated), say so plainly "
        "and ask them to answer the actual question. If partially right, say what was right and "
        "what was missing, with one concrete suggestion for improvement."
    ))
    next_question: str = Field(description=(
        "The next oral question: a bit harder or a related sub-topic if they did well, an easier "
        "follow-up on the same idea if they struggled. Vary phrasing/angle each time. "
        "Spoken-friendly, no markdown."
    ))


class FinalSummary(BaseModel):
    overall_score: float = Field(ge=0, le=10, description="Holistic score out of 10 for the whole session.")
    summary: str = Field(description=(
        "A short, honest, encouraging spoken wrap-up: one sentence on overall performance, one on "
        "what they did well, one on the specific area(s) to focus on before their test. "
        "Spoken-friendly, no markdown."
    ))


def _invoke_with_retry(chain, inputs: dict, retries: int = 1, delay_seconds: float = 1.0):
    """Runs chain.invoke(inputs), retrying once after a short pause if it
    fails - a lot of LLM call failures are transient (a brief rate limit,
    a dropped connection), and retrying once for free is cheap insurance
    against the whole app getting stuck over a one-off hiccup."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(delay_seconds)
    raise last_error


def _format_history(history: list[dict]) -> str:
    lines = []
    for m in history:
        speaker = config.ASSISTANT_NAME if m["role"] == "assistant" else "Student"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines) if lines else "(no messages yet)"


# =============================================================================
# Chain functions - what app.py actually calls
# =============================================================================

def check_topic(topic_text: str) -> TopicCheck:
    structured = get_llm().with_structured_output(TopicCheck)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _PERSONA),
        ("human", (
            'The student was asked what Python topic they want to practice. They said: '
            '"{topic}". Decide if this is usable (a specific Python topic, or a general request '
            "like 'mixed quiz' / 'anything' / 'general Python' all count as valid) and respond."
        )),
    ])
    return _invoke_with_retry(prompt | structured, {"topic": topic_text})


def generate_first_question(topic: str) -> str:
    structured = get_question_llm().with_structured_output(FirstQuestion)
    angle = random.choice(_QUESTION_ANGLES)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _PERSONA),
        ("human", 'Topic: "{topic}". {angle} Ask your opening oral quiz question now.'),
    ])
    result: FirstQuestion = _invoke_with_retry(prompt | structured, {"topic": topic, "angle": angle})
    return result.question


def evaluate_answer(topic: str, history: list[dict], student_answer: str) -> QuizTurn:
    structured = get_llm().with_structured_output(QuizTurn)
    angle = random.choice(_QUESTION_ANGLES)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _PERSONA),
        ("human", (
            'Topic: "{topic}"\n\nConversation so far:\n{history}\n\n'
            'The student\'s latest spoken answer (transcribed, may contain small transcription '
            'errors - be lenient about wording/grammar, judge the underlying content): '
            '"{answer}"\n\nScore it, give specific feedback, then ask the next question. '
            "{angle}"
        )),
    ])
    return _invoke_with_retry(prompt | structured, {
        "topic": topic, "history": _format_history(history), "answer": student_answer, "angle": angle,
    })


def generate_final_summary(topic: str, history: list[dict], turn_scores: list[int]) -> FinalSummary:
    structured = get_llm().with_structured_output(FinalSummary)
    avg = sum(turn_scores) / len(turn_scores) if turn_scores else 0.0
    prompt = ChatPromptTemplate.from_messages([
        ("system", _PERSONA),
        ("human", (
            'Topic(s) covered: "{topic}"\n\nFull session transcript:\n{history}\n\n'
            f"The student's per-question scores were {turn_scores} (average {avg:.1f}/10).\n\n"
            "Write the final wrap-up now."
        )),
    ])
    return _invoke_with_retry(prompt | structured, {"topic": topic, "history": _format_history(history)})
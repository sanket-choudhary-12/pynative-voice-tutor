"""
llm_chains.py  -  the "brain" of the app.

WHAT THIS FILE DOES (simple words):
    Every conversation with the language model (LLM) happens here. app.py never
    talks to Groq directly. It calls one of four functions below and gets back
    a clean result:
        check_topic()            -> is what the student said a Python topic?
        generate_first_question()-> the opening question
        evaluate_answer()        -> score + feedback + next question
        generate_final_summary() -> final score + spoken wrap-up

HOW EACH FUNCTION TALKS TO THE LLM:
    1. Build a prompt (instructions + the student's text).
    2. Ask the model to answer in a fixed shape (a Pydantic class).
    3. PLAN A  : use the model's "tool calling" feature to get that shape.
       PLAN B  : if Plan A fails, ask for plain JSON text and read it ourselves.
       (Plan B was added because some models, e.g. openai/gpt-oss-120b, sometimes
       forget a field in tool calling and Groq answers "Error code: 400 - Tool
       call validation failed ... missing properties".)
    4. Return a validated Python object.

SAFETY RULES USED IN THIS FILE:
    * A wrong/missing field never crashes the app: Plan B repairs most cases.
    * check_topic() and generate_final_summary() have local fallbacks, so the
      quiz can always start and always finish, even if the LLM is down.
"""

import json
import random
import time

import streamlit as st
from groq import Groq as GroqClient
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, field_validator

import config


# =============================================================================
# 1. Which model to use, and which API key
# =============================================================================

# Models we would like to use, best first. The first one available to the
# student's Groq account is used.
_PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]


def get_api_key() -> str:
    """Returns the Groq API key of the CURRENT visitor.
    Order: the key typed into the page (kept in session_state), then the key from
    a local .env file, then an empty string."""
    return st.session_state.get("groq_api_key") or config.GROQ_API_KEY or ""


def validate_api_key(api_key: str) -> tuple[bool, str]:
    """Checks a key by making one cheap, free call to Groq (listing models).
    Returns (True, "") if it works, or (False, "reason") if it does not."""
    if not api_key or not api_key.strip():
        return False, "Please paste your Groq API key."
    try:
        GroqClient(api_key=api_key.strip()).models.list()
        return True, ""
    except Exception as e:
        return False, f"That key was rejected by Groq: {e}"


@st.cache_resource(show_spinner=False)
def _list_available_models(api_key: str) -> tuple:
    """Asks Groq which models this key can use. The answer is remembered (cached)
    per key. IMPORTANT: if the call fails it RAISES, and Streamlit does not cache
    errors - so one bad moment never gets 'stuck' in memory."""
    client = GroqClient(api_key=api_key)
    return tuple(sorted(m.id for m in client.models.list().data))


def resolve_model_name(api_key: str) -> str:
    """Picks the best model this key can use. Never raises an error:
    if anything goes wrong it returns config.LLM_MODEL_FALLBACK."""
    try:
        available = set(_list_available_models(api_key))
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


def get_llm() -> ChatGroq:
    """The model used for judging answers (temperature 0.4 = calm and consistent).
    NOT cached on purpose: a cached object would be shared by ALL visitors and
    would reuse the first visitor's API key."""
    key = get_api_key()
    return ChatGroq(model=resolve_model_name(key), api_key=key, temperature=0.4)


def get_question_llm() -> ChatGroq:
    """The model used for writing questions (temperature 0.9 = more varied, so we
    do not get the same textbook question every time)."""
    key = get_api_key()
    return ChatGroq(model=resolve_model_name(key), api_key=key, temperature=0.9)


# A random "style hint" added to prompts so the questions keep changing.
_QUESTION_ANGLES = [
    "Ask it as a 'what's the difference between X and Y' style question.",
    "Ask it as a 'what would happen if...' conceptual scenario question.",
    "Ask it as a 'when would you use X instead of Y' question.",
    "Ask it as a direct, simple definition/explanation question.",
    "Ask it as a short real-world example-based question.",
]


# =============================================================================
# 2. The persona (system prompt) and the exact shapes of the LLM's answers
# =============================================================================

# The system prompt: who the model is and the rules it must always follow.
_PERSONA = f"""You are {config.ASSISTANT_NAME}, a friendly, encouraging voice-based Python \
programming tutor. You ONLY discuss Python programming topics. If the student says anything \
unrelated to Python, politely redirect them back to Python in your response. Keep every response \
spoken-friendly: no markdown, no code blocks, no bullet points, no asterisks - plain conversational \
sentences, since everything you say is read aloud by text-to-speech."""


class TopicCheck(BaseModel):
    """Answer shape for 'is this a Python topic?'.
    Both fields have DEFAULTS so that, if the model forgets one, Groq's validation
    does not reject the whole reply (this was the cause of the 400 error)."""
    is_valid: bool = Field(default=True, description=(
        "True if this is a genuine Python topic (e.g. 'lists and tuples', 'recursion', "
        "'decorators'), OR a general request like 'mixed quiz' / 'anything' / 'general Python'. "
        "False if it's unrelated to Python entirely."
    ))
    message: str = Field(default="", description=(
        "If is_valid is True: one short, friendly spoken line acknowledging the topic "
        "(e.g. 'Great, let's practice lists and tuples!'). "
        "If is_valid is False: one short, friendly spoken line redirecting them "
        "(e.g. 'I can only help with Python topics - what Python topic would you like to practice?')."
    ))


class FirstQuestion(BaseModel):
    """Answer shape for the opening question."""
    question: str = Field(description=(
        "One clear, beginner-to-intermediate oral quiz question about the topic. Vary your "
        "phrasing and specific angle every time - never default to the single most obvious, "
        "generic textbook question. Spoken-friendly, no markdown."
    ))


class QuizTurn(BaseModel):
    """Answer shape for judging one spoken answer."""
    score: int = Field(ge=0, le=10, description=(
        "How well the student's answer addressed the question, as a WHOLE NUMBER: 0 (didn't "
        "attempt / completely wrong / off-topic nonsense) to 10 (excellent, complete, accurate)."
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

    @field_validator("score", mode="before")
    @classmethod
    def clean_score(cls, value):
        """If the model gives 7.5, '8' or 12, turn it into a whole number from 0 to 10."""
        try:
            return max(0, min(10, int(round(float(value)))))
        except (TypeError, ValueError):
            return value  # not a number at all -> let normal validation complain


class FinalSummary(BaseModel):
    """Answer shape for the end-of-session wrap-up."""
    overall_score: float = Field(ge=0, le=10, description="Holistic score out of 10 for the whole session.")
    summary: str = Field(description=(
        "A short, honest, encouraging spoken wrap-up: one sentence on overall performance, one on "
        "what they did well, one on the specific area(s) to focus on before their test. "
        "Spoken-friendly, no markdown."
    ))

    @field_validator("overall_score", mode="before")
    @classmethod
    def clean_overall(cls, value):
        """Keeps the overall score between 0 and 10."""
        try:
            return max(0.0, min(10.0, float(value)))
        except (TypeError, ValueError):
            return value


# =============================================================================
# 3. Helper functions that make the LLM calls safe
# =============================================================================

def _invoke_with_retry(chain, inputs: dict, retries: int = 1, delay_seconds: float = 1.0):
    """Runs chain.invoke(inputs). If it fails, waits a second and tries again
    (many failures are short hiccups). If it still fails, raises the error."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(delay_seconds)
    raise last_error


def _extract_json_object(text: str) -> dict:
    """Finds the first { ... } block inside the model's text and turns it into a
    Python dict. This also copes with models that wrap JSON in ```json fences."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("the model's reply did not contain a JSON object")
    return json.loads(text[start:end + 1])


def _invoke_json_fallback(llm, prompt, schema, inputs: dict):
    """PLAN B. Instead of 'tool calling', we just ask the model to reply with a
    JSON object, then read and validate it ourselves with the Pydantic class."""
    messages = prompt.format_messages(**inputs)  # the same prompt, filled in
    field_lines = "\n".join(
        f'- "{name}": {info.description}' for name, info in schema.model_fields.items()
    )
    messages.append(HumanMessage(content=(
        "Reply with ONLY one JSON object and nothing else - no markdown, no code fences, no "
        "extra words. It must contain exactly these keys (use true/false for booleans and plain "
        "numbers for scores):\n" + field_lines
    )))
    response = llm.invoke(messages)
    text = response.content if isinstance(response.content, str) else str(response.content)
    return schema.model_validate(_extract_json_object(text))


def _run(llm, prompt, schema, inputs: dict):
    """Runs one LLM request as safely as possible.
    PLAN A: structured output (tool calling), with one retry.
    PLAN B: plain JSON text, with one retry.
    If both plans fail, raises ONE clear error that mentions both reasons."""
    try:
        return _invoke_with_retry(prompt | llm.with_structured_output(schema), inputs)
    except Exception as plan_a_error:
        plan_a_message = str(plan_a_error)
        print(f"[LLM] Plan A (tool calling) failed: {plan_a_message}\n[LLM] Trying Plan B (plain JSON)...")

    last_error = None
    for attempt in range(2):
        try:
            return _invoke_json_fallback(llm, prompt, schema, inputs)
        except Exception as e:
            last_error = e
            print(f"[LLM] Plan B attempt {attempt + 1} failed: {e}")
            if attempt == 0:
                time.sleep(1.0)
    raise RuntimeError(f"{last_error} (first attempt had failed with: {plan_a_message})")


def _format_history(history: list[dict], max_messages: int = 16) -> str:
    """Turns the stored conversation (a list of dicts) into plain text for the
    prompt, e.g. 'Student: ...' / 'PyNative: ...'. Only the most recent
    `max_messages` lines are used so a long session never overflows the model."""
    lines = []
    for m in history[-max_messages:]:
        speaker = config.ASSISTANT_NAME if m["role"] == "assistant" else "Student"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines) if lines else "(no messages yet)"


# =============================================================================
# 4. The four functions that app.py calls
# =============================================================================

def check_topic(topic_text: str) -> TopicCheck:
    """Decides if what the student said is a usable Python topic.
    Returns TopicCheck(is_valid, message).
    NEVER raises: if the LLM cannot be used at all, we accept the topic so the
    quiz can still start (the next step has its own fallback question)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", _PERSONA),
        ("human", (
            'The student was asked what Python topic they want to practice. They said: '
            '"{topic}". Decide if this is usable (a specific Python topic, or a general request '
            "like 'mixed quiz' / 'anything' / 'general Python' all count as valid) and respond."
        )),
    ])
    try:
        result = _run(get_llm(), prompt, TopicCheck, {"topic": topic_text})
    except Exception as e:
        print(f"[LLM] check_topic failed completely, accepting the topic: {e}")
        return TopicCheck(is_valid=True, message=f"Okay, let's practice {topic_text.strip()}.")

    # If the model forgot the spoken line, fill in a sensible one ourselves.
    if not result.message.strip():
        if result.is_valid:
            result.message = "Great, let's practice that!"
        else:
            result.message = "I can only help with Python topics - what Python topic would you like to practice?"
    return result


def generate_first_question(topic: str) -> str:
    """Returns the opening quiz question (a plain string).
    Raises an error if it fails - app.py then uses its own backup question."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", _PERSONA),
        ("human", 'Topic: "{topic}". {angle} Ask your opening oral quiz question now.'),
    ])
    angle = random.choice(_QUESTION_ANGLES)
    result: FirstQuestion = _run(get_question_llm(), prompt, FirstQuestion, {"topic": topic, "angle": angle})
    if not result.question.strip():
        raise ValueError("the model returned an empty question")
    return result.question.strip()


def evaluate_answer(topic: str, history: list[dict], student_answer: str) -> QuizTurn:
    """The main function, used for EVERY answer. Gives back a QuizTurn with
    .score (0-10), .feedback and .next_question.
    Raises an error if it fails - app.py shows it and lets the student answer again."""
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
    angle = random.choice(_QUESTION_ANGLES)
    result: QuizTurn = _run(get_llm(), prompt, QuizTurn, {
        "topic": topic, "history": _format_history(history), "answer": student_answer, "angle": angle,
    })
    if not result.feedback.strip() or not result.next_question.strip():
        raise ValueError("the model returned an incomplete answer (empty feedback or question)")
    return result


def generate_final_summary(topic: str, history: list[dict], turn_scores: list[int]) -> FinalSummary:
    """Writes the final wrap-up. NEVER raises: if the LLM fails, we build a simple
    summary ourselves from the scores, so the result screen always works."""
    avg = sum(turn_scores) / len(turn_scores) if turn_scores else 0.0
    prompt = ChatPromptTemplate.from_messages([
        ("system", _PERSONA),
        ("human", (
            'Topic(s) covered: "{topic}"\n\nFull session transcript:\n{history}\n\n'
            f"The student's per-question scores were {turn_scores} (average {avg:.1f}/10).\n\n"
            "Write the final wrap-up now."
        )),
    ])
    try:
        result = _run(get_llm(), prompt, FinalSummary, {"topic": topic, "history": _format_history(history, 40)})
        if result.summary.strip():
            return result
    except Exception as e:
        print(f"[LLM] generate_final_summary failed, using the local summary: {e}")

    # ---- local backup summary (no LLM needed) ----
    if not turn_scores:
        text = "You didn't answer any questions this time. Come back whenever you're ready and give it a try."
    elif avg >= 8:
        text = (f"Great work! You averaged {avg:.1f} out of 10 over {len(turn_scores)} questions. "
                "Keep practicing a little every day to stay sharp.")
    elif avg >= 5:
        text = (f"Good effort! You averaged {avg:.1f} out of 10 over {len(turn_scores)} questions. "
                "Review the points where the feedback said something was missing, then try again.")
    else:
        text = (f"You averaged {avg:.1f} out of 10 over {len(turn_scores)} questions. "
                "Don't worry, go over the basics of this topic once more and try another round.")
    return FinalSummary(overall_score=round(avg, 1), summary=text)

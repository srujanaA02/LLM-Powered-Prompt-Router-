"""
LLM-Powered Prompt Router for Intent Classification
=====================================================
Uses Groq API (free, fast LLaMA models) to classify intent and route to expert personas.
"""

import json
import os
import datetime
from pathlib import Path
from groq import Groq

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PROMPTS_FILE = BASE_DIR / "prompts.json"
LOG_FILE = BASE_DIR / "route_log.jsonl"
MODEL = "llama-3.3-70b-versatile"

# ── Load expert prompts from config ───────────────────────────────────────────
def load_prompts() -> dict:
    """Load expert system prompts from prompts.json configuration file."""
    with open(PROMPTS_FILE, "r") as f:
        return json.load(f)

EXPERT_PROMPTS = load_prompts()

# ── Groq client ────────────────────────────────────────────────────────────────
client = Groq()   # reads GROQ_API_KEY from environment automatically

# ── Core Functions ─────────────────────────────────────────────────────────────

def classify_intent(user_message: str) -> dict:
    """
    Classify the intent of a user message using Groq LLaMA.

    Returns:
        dict: {"intent": str, "confidence": float}
        On parse failure, defaults to {"intent": "unclear", "confidence": 0.0}
    """
    valid_intents = [k for k in EXPERT_PROMPTS.keys() if k != "unclear"]

    classifier_prompt = f"""You are an intent classification engine. Analyze the user message and classify it into exactly one of these intents:

{json.dumps(valid_intents, indent=2)}

Intent definitions:
- code: Programming, debugging, software development, algorithms, APIs
- data_analysis: Data science, statistics, ML, SQL, data visualization, analytics
- writing: Essays, emails, editing, creative writing, content, proofreading
- career: Resume, interviews, job search, salary, professional development, LinkedIn
- unclear: Cannot determine intent, too vague, or doesn't fit any category

IMPORTANT: Respond with ONLY a valid JSON object. No explanation, no markdown, no extra text.

Required format:
{{"intent": "<one of the valid intents>", "confidence": <float between 0.0 and 1.0>}}

User message: "{user_message}"
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=100,
            messages=[
                {
                    "role": "system",
                    "content": "You are an intent classifier. Always respond with valid JSON only. No markdown, no explanation."
                },
                {
                    "role": "user",
                    "content": classifier_prompt
                }
            ]
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if model wraps response
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        # Extract JSON if there's extra text around it
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        parsed = json.loads(raw)

        # Validate structure
        intent = str(parsed.get("intent", "unclear")).lower()
        confidence = float(parsed.get("confidence", 0.0))

        # Ensure intent is a known label
        all_intents = list(EXPERT_PROMPTS.keys())
        if intent not in all_intents:
            intent = "unclear"
            confidence = 0.0

        return {"intent": intent, "confidence": round(confidence, 4)}

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        # Graceful fallback — never crash
        print(f"[WARN] Intent parsing failed: {e}. Defaulting to 'unclear'.")
        return {"intent": "unclear", "confidence": 0.0}


def route_and_respond(user_message: str, intent: str) -> str:
    """
    Route the user message to the correct expert persona and generate a response.

    Args:
        user_message: The original user query
        intent: Classified intent label (e.g. 'code', 'writing', 'unclear')

    Returns:
        str: The expert's response text
    """
    expert_config = EXPERT_PROMPTS.get(intent, EXPERT_PROMPTS["unclear"])
    system_prompt = expert_config["system_prompt"]

    # For unclear intent, guide the model to ask for clarification
    if intent == "unclear":
        augmented_message = (
            f"{user_message}\n\n"
            "[Ask the user to clarify whether they need help with: "
            "coding/programming, data analysis, writing/communication, "
            "or career/professional development.]"
        )
    else:
        augmented_message = user_message

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": augmented_message}
        ]
    )

    return response.choices[0].message.content.strip()


def log_entry(user_message: str, intent: str, confidence: float, final_response: str):
    """Append a routing decision and response to route_log.jsonl."""
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "user_message": user_message,
        "intent": intent,
        "confidence": confidence,
        "expert_label": EXPERT_PROMPTS.get(intent, {}).get("label", "Unknown"),
        "final_response": final_response
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def process_request(user_message: str) -> dict:
    """
    Full pipeline: classify → route → respond → log.
    """
    print(f"\n{'='*60}")
    print(f"USER: {user_message}")
    print(f"{'='*60}")

    # Step 1: Classify intent
    classification = classify_intent(user_message)
    intent = classification["intent"]
    confidence = classification["confidence"]
    print(f"CLASSIFIED → intent='{intent}' | confidence={confidence:.2%}")

    # Step 2: Route and generate response
    final_response = route_and_respond(user_message, intent)
    expert_label = EXPERT_PROMPTS.get(intent, {}).get("label", "Unknown")
    print(f"ROUTED TO  → {expert_label}")
    print(f"\nRESPONSE:\n{final_response}\n")

    # Step 3: Log the decision
    log_entry(user_message, intent, confidence, final_response)

    return {
        "intent": intent,
        "confidence": confidence,
        "expert": expert_label,
        "final_response": final_response
    }


# ── Demo / Test Runner ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_messages = [
        "How do I implement a binary search tree in Python?",
        "Can you help me analyze my sales data and find seasonal trends?",
        "Please proofread my cover letter and make it more compelling.",
        "I have an interview at Google next week. How should I prepare?",
        "I need help with something.",   # unclear → clarification
    ]

    print("\n🚀 LLM Prompt Router — Powered by Groq (Free!)")
    print(f"🤖 Model: {MODEL}")
    print(f"📋 Log file: {LOG_FILE}\n")

    for msg in test_messages:
        process_request(msg)

    print(f"\n✅ All requests processed. Check '{LOG_FILE}' for logs.")
# LLM-Powered Prompt Router (OpenAI)

Routes user requests to specialized AI expert personas using **OpenAI GPT models**.

## Architecture

```
User Message
     │
     ▼
┌──────────────────────┐
│  classify_intent()   │  ← Groq Call #1 
│  {"intent","conf"}   │    Structured JSON output guaranteed
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ route_and_respond()  │  ← Groq Call #2 (expert system prompt)
│  Expert persona      │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│    log_entry()       │  → route_log.jsonl
└──────────────────────┘
```

## Expert Personas (prompts.json)

| Intent         | Expert                        |
|----------------|-------------------------------|
| `code`         | Elite Software Engineer       |
| `data_analysis`| World-Class Data Scientist    |
| `writing`      | Professional Writer/Editor    |
| `career`       | Seasoned Career Coach         |
| `unclear`      | Clarification Handler         |

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...
```

## Usage

```bash
# Interactive chat
python cli.py

# Demo with 5 test messages
python router.py

# Run all tests
python test_router.py
```

## Key Design Decisions

- Uses `response_format={"type": "json_object"}` on the classifier call to force valid JSON from OpenAI
- Falls back to `{"intent": "unclear", "confidence": 0.0}` on ANY parse/type error — never crashes
- Expert prompts stored in `prompts.json`, not hardcoded in business logic
- Each log line in `route_log.jsonl` is a self-contained valid JSON object

## Log Format

```jsonl
{"timestamp":"2026-03-14T10:00:00Z","user_message":"How do I sort in Python?","intent":"code","confidence":0.97,"expert_label":"Code Expert","final_response":"..."}
{"timestamp":"2026-03-14T10:01:00Z","user_message":"I need help","intent":"unclear","confidence":0.0,"expert_label":"Clarification Handler","final_response":"Could you clarify...?"}
```
"""
Tests for LLM Prompt Router (Groq version)
Run: python test_router.py
"""

import json, sys, os, tempfile
from unittest.mock import MagicMock

# Mock groq before importing router
mock_groq = MagicMock()
mock_client = MagicMock()
mock_groq.Groq.return_value = mock_client
sys.modules["groq"] = mock_groq

import Router as router
router.client = mock_client

def make_resp(text):
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    return resp

passed = failed = 0

def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"✅ PASS: {name}")
        passed += 1
    except AssertionError as e:
        print(f"❌ FAIL: {name} → {e}")
        failed += 1

# 1. Valid code intent
def t1():
    mock_client.chat.completions.create.return_value = make_resp('{"intent": "code", "confidence": 0.97}')
    r = router.classify_intent("Write a Python binary search tree")
    assert r["intent"] == "code"
    assert r["confidence"] == 0.97
run_test("Valid code intent classified correctly", t1)

# 2. Valid data_analysis intent
def t2():
    mock_client.chat.completions.create.return_value = make_resp('{"intent": "data_analysis", "confidence": 0.91}')
    r = router.classify_intent("Analyze my CSV sales data")
    assert r["intent"] == "data_analysis"
run_test("Valid data_analysis intent", t2)

# 3. Malformed JSON → unclear/0.0
def t3():
    mock_client.chat.completions.create.return_value = make_resp("This looks like coding!")
    r = router.classify_intent("Some message")
    assert r["intent"] == "unclear"
    assert r["confidence"] == 0.0
run_test("Malformed JSON defaults to unclear/0.0 (no crash)", t3)

# 4. Empty response → unclear/0.0
def t4():
    mock_client.chat.completions.create.return_value = make_resp("")
    r = router.classify_intent("Something")
    assert r["intent"] == "unclear"
    assert r["confidence"] == 0.0
run_test("Empty response defaults to unclear/0.0", t4)

# 5. Unknown intent label → unclear
def t5():
    mock_client.chat.completions.create.return_value = make_resp('{"intent": "cooking", "confidence": 0.85}')
    r = router.classify_intent("How to make pasta?")
    assert r["intent"] == "unclear"
run_test("Unknown intent label defaults to unclear", t5)

# 6. Markdown-fenced JSON parsed correctly
def t6():
    mock_client.chat.completions.create.return_value = make_resp('```json\n{"intent": "writing", "confidence": 0.88}\n```')
    r = router.classify_intent("Proofread my essay")
    assert r["intent"] == "writing"
run_test("Markdown-fenced JSON parsed correctly", t6)

# 7. Code intent → Code Expert system prompt
def t7():
    mock_client.chat.completions.create.return_value = make_resp("Here is your code...")
    router.route_and_respond("Implement quicksort", "code")
    msgs = mock_client.chat.completions.create.call_args[1]["messages"]
    assert "software engineer" in msgs[0]["content"].lower()
run_test("'code' routes to Code Expert system prompt", t7)

# 8. Unclear intent → clarification prompt
def t8():
    mock_client.chat.completions.create.return_value = make_resp("Could you clarify?")
    router.route_and_respond("I need help", "unclear")
    msgs = mock_client.chat.completions.create.call_args[1]["messages"]
    assert "clarif" in msgs[0]["content"].lower()
run_test("'unclear' uses clarification system prompt", t8)

# 9. Log file valid JSONL with required keys
def t9():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp = f.name
    orig = router.LOG_FILE
    router.LOG_FILE = tmp
    try:
        router.log_entry("test msg", "code", 0.95, "Here is the answer")
        router.log_entry("vague", "unclear", 0.0, "Clarify please")
        with open(tmp) as f:
            lines = f.readlines()
        assert len(lines) == 2
        for line in lines:
            entry = json.loads(line)
            for key in ["intent", "confidence", "user_message", "final_response"]:
                assert key in entry, f"Missing key: {key}"
    finally:
        router.LOG_FILE = orig
        os.unlink(tmp)
run_test("JSONL log contains all required keys", t9)

# 10. At least 4 expert prompts
def t10():
    experts = {k: v for k, v in router.EXPERT_PROMPTS.items() if k != "unclear"}
    assert len(experts) >= 4, f"Only {len(experts)} experts"
run_test(f"At least 4 expert prompts defined", t10)

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests")
print("🎉 All tests PASSED!" if failed == 0 else "⚠️  Some tests FAILED.")

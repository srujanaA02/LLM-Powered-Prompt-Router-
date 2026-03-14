"""
Interactive CLI for the LLM Prompt Router (Groq version)
Run: python cli.py
"""
from Router import process_request, EXPERT_PROMPTS, LOG_FILE, MODEL
def print_banner():
    print("\n" + "="*60)
    print("  🚀  LLM Prompt Router  —  Powered by Groq (Free!)")
    print(f"  🤖  Model: {MODEL}")
    print("="*60)
    print("Available expert domains:")
    for key, val in EXPERT_PROMPTS.items():
        if key != "unclear":
            print(f"  • [{key}]  {val['label']}")
    print(f"\nLogs: {LOG_FILE}")
    print("Type 'quit' to exit.\n")

def main():
    print_banner()
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye! 👋")
                break
            process_request(user_input)
        except KeyboardInterrupt:
            print("\nBye!")
            break

if __name__ == "__main__":
    main()

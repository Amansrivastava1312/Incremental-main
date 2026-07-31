# """
# main.py
# Top-level entry point for the LangChain agent project.

# Adds the `agent/` folder to sys.path so `src.tool_agent` imports work,
# then routes questions through the MarketPulse tool router.
# """

# import sys
# from pathlib import Path

# # Project root (LANGCHAIN_AGENT) where main.py lives
# ROOT_DIR = Path(__file__).resolve().parent

# # `agent` folder holds the `src` package -> needed for `from src.tool_agent import ...`
# AGENT_DIR = ROOT_DIR / "agent"

# # Add both root and agent folder to the path
# for p in (ROOT_DIR, AGENT_DIR):
#     if str(p) not in sys.path:
#         sys.path.insert(0, str(p))

# from src.tool_agent import run_tool_agent


# def main() -> None:
#     # ---- Example 1: sales forecast ----
#     q1 = "Give me a 30 day sarima sales forecast for product P100"
#     print("Q:", q1)
#     print(run_tool_agent(q1))

#     # ---- Example 2: review sentiment ----
#     q2 = "Is this review positive or negative: the product broke in two days"
#     print("\nQ:", q2)
#     print(run_tool_agent(q2))

#     # ---- Example 3: vision (needs a real image path) ----
#     q3 = "Analyze this image and detect the objects"
#     print("\nQ:", q3)

#     image_path = None  # e.g. r"C:\Users\10857884\Downloads\1.jpg"

#     if image_path:
#         print(run_tool_agent(q3, image_path=image_path))
#     else:
#         print("Skipping vision demo (set `image_path` to a real image to run it).")


# if __name__ == "__main__":
#     main()




import json
from pathlib import Path

from agent.script import run_marketpulse



def main():
    print("=" * 50)
    print("      Welcome to MarketPulse Agent CLI")
    print("=" * 50)

    try:
        question = input("\nAsk MarketPulse: ").strip()
        if not question:
            print("Error: Question cannot be empty.")
            sys.exit(1)

        raw_path = input("Image path (press Enter to skip): ").strip()
        image_path = raw_path if raw_path else None

        print("\n[Processing request...]")
        
        # Execute the agent workflow
        result = run_marketpulse(
        question=question,
        image_path=image_path or None
    )


        # Print structured JSON response
        print("\n--- Agent Response ---")
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"\nExecution Error: {e}")


if __name__ == "__main__":
    main()


 
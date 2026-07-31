import json
from LangChain.agent.src.tool_agent import agent


def run_tool_agent(question: str):

    if not question or not question.strip():
        raise ValueError("question is required")

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question.strip(),
                }
            ]
        }
    )

    final_msg = response["messages"][-1].content

    if (
        isinstance(final_msg, list)
        and len(final_msg) > 0
        and isinstance(final_msg[0], dict)
        and "text" in final_msg[0]
    ):
        text = final_msg[0]["text"]

        try:
            return json.loads(text)
        except Exception:
            return text

    return final_msg
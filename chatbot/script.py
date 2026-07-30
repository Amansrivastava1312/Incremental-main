from chatbot.src.chat_agent import create_education_agent

agent = create_education_agent()


def run_chatbot(question):

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    final_msg = response["messages"][-1].content
    return (final_msg[0]["text"])
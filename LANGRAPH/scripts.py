from LANGRAPH.src.agent import build_graph


def run_chatbot(question):

    state = {
        "question": question,
        "context": "",
        "research_notes": "",
        "answer": "",
        "approved": False,
        "issues": []
    }

    result = build_graph().invoke(state)

    return {
        "answer": result["answer"],
        
    }
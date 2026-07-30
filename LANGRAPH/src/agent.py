from typing import TypedDict, List
import json
from langgraph.graph import StateGraph, START, END
from rag.pipeline import RAGPipeline


# PIPELINE INITIALIZATION

pipeline = RAGPipeline()

if not pipeline.chroma_dir.exists():
    print("Creating vector database...")
    pipeline.ingest()


# LLM HELPER


def llm_call(prompt: str) -> str:
    response = pipeline.generator.generate(prompt)
    if isinstance(response, list):
        text_parts = []
        for item in response:
            if isinstance(item, dict):
                text_parts.append(item.get("text", ""))
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts)
    return str(response)

class AgentState(TypedDict):
    question: str
    context: str
    research_notes: str
    answer: str
    approved: bool
    issues: List[str]

# RESEARCH AGENT

def rag_search(question: str) -> str:
    """Retrieve relevant information using the RAG pipeline."""
    return pipeline.get_context(question)


def prepare_research(question: str, context: str) -> str:
    """
    Convert retrieved context into structured notes.
    """
    prompt = f"""
    You are a Research Assistant.
    Use ONLY the retrieved context.
    Question:
    {question}
    Context:
    {context}
    Prepare structured research notes.
    Include:
    - Important facts
    - Eligibility
    - Admission details
    - Policies
    - Fees (if available)
    - Deadlines (if available)
    Do not answer the user directly.
    Do not hallucinate.
    """
    return llm_call(prompt)


def researcher_node(state: AgentState):
    print("=" * 60)
    print("Running Research Agent")
    print("=" * 60)
    context = rag_search(state["question"])
    notes = prepare_research(
        state["question"],
        context
    )

    return {
        "context": context,
        "research_notes": notes
    }

# WRITER AGENT

def generate_response(question: str,context: str,research_notes: str) -> str:
    """Generate final answer."""
    prompt = f"""
    You are an Educational Assistant.
    Answer the user's question using ONLY the information below.
    ====================================
    Question
    ====================================
    {question}
    ====================================
    Retrieved Context
    ====================================
    {context}
    ====================================
    Research Notes
    ====================================
    {research_notes}
    Instructions:
    1. Answer naturally.
    2. Be professional.
    3. Do not hallucinate.
    4. Do not add extra information.
    5. If information is unavailable, clearly say so.
    6. Keep the answer concise and accurate.
    """

    return llm_call(prompt)


def format_response(answer: str) -> str:
    """Format response."""
    prompt = f"""
    Format the following response:
    {answer}
    Rules:
    - Use markdown
    - Use bullet points where appropriate
    - Keep spacing clean
    - Do not change meaning
    """

    return llm_call(prompt)


def writer_node(state: AgentState):

    print("=" * 60)
    print("Running Writer Agent")
    print("=" * 60)
    answer = generate_response(
        state["question"],
        state["context"],
        state["research_notes"]
    )

    formatted_answer = format_response(answer)

    return {
        "answer": formatted_answer
    }


# CRITIC AGENT

def response_checker(answer: str):
    issues = []
    if not answer.strip():
        issues.append("Empty response generated.")
    if len(answer.strip()) < 40:
        issues.append("Response is too short.")
    return {
        "approved": len(issues) == 0,
        "issues": issues
    }


def policy_checker(answer: str, context: str):

    prompt = f"""
    You are a Quality Assurance Agent.
    Retrieved Context:
    {context}
    Generated Answer:
    {answer}
    Check whether the answer is completely supported
    by the retrieved context.
    Return ONLY valid JSON.
    {{
        "approved": true,
        "reason": ""
    }}
    """
    response = llm_call(prompt)
    try:
        return json.loads(response)

    except Exception:
        return {
            "approved": True,
            "reason": ""
        }


def critic_node(state: AgentState):
    print("=" * 60)
    print("Running Critic Agent")
    print("=" * 60)
    response_result = response_checker(state["answer"])
    policy_result = policy_checker(
        state["answer"],
        state["context"]
    )
    issues = []
    issues.extend(response_result.get("issues", []))
    if not policy_result.get("approved", True):
        issues.append(
            policy_result.get(
                "reason",
                "Policy validation failed."
            )
        )

    return {
        "approved": len(issues) == 0,
        "issues": issues
    }


# BUILD GRAPH

def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("Research",researcher_node)
    builder.add_node("Writer",writer_node)
    builder.add_node("Critic",critic_node)
    builder.add_edge(START,"Research")
    builder.add_edge("Research","Writer")
    builder.add_edge("Writer","Critic")
    builder.add_edge("Critic",END)
    return builder.compile()


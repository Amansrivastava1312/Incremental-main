import os

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.pipeline import RAGPipeline

# =====================================================
# API KEY
# =====================================================
api_key = os.getenv("GEMINI_API_KEY")

# =====================================================
# RAG PIPELINE  (loads txt + pdf + docx knowledge base)
# =====================================================
pipeline = RAGPipeline()

# Build the vector DB once if it does not exist yet
if not pipeline.chroma_dir.exists():
    print("Creating vector database from txt / pdf / docx datasets...")
    pipeline.ingest()


# =====================================================
# RAG SEARCH HELPER
# =====================================================
def rag_search(query: str) -> str:
    context = pipeline.get_context(query)
    if not context or not context.strip():
        return "Information not found in the education documents."
    return context


# =====================================================
# TOOL 1 - General document search (RAG)
# =====================================================
@tool
def search_education_documents(query: str) -> str:
    """Search the education knowledge base (txt, pdf, docx) for any topic,
    concept explanation, or general question using RAG."""
    return rag_search(query)


# =====================================================
# TOOL 2 - Course details (RAG-backed, replaces all the old
#          hardcoded duration / prerequisites / resources /
#          careers / roadmap dictionaries)
# =====================================================
@tool
def get_course_details(course: str, aspect: str = "overview") -> str:
    """Get details about a course from the knowledge base.
    'aspect' can be: overview, duration, prerequisites, resources,
    next course / roadmap, or career opportunities.
    All information is retrieved from the datasets via RAG."""
    query = f"{course} {aspect}"
    return rag_search(query)


# =====================================================
# TOOL 3 - Quiz generator (dynamic, no static data)
# =====================================================
@tool
def generate_quiz(topic: str) -> str:
    """Generate simple practice quiz questions for a given topic."""
    return f"""Quiz on {topic}

1. What is {topic}?
2. Why do we use {topic}?
3. Give one real-world application of {topic}.
4. What is a common prerequisite before learning {topic}?
"""


# =====================================================
# TOOLS LIST  (reduced from 8 -> 3)
# =====================================================
tools = [
    search_education_documents,
    get_course_details,
    generate_quiz,
]


# =====================================================
# CREATE AGENT
# =====================================================
def create_education_agent():
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0,
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
You are an Education AI Assistant.

Rules:
1. Use search_education_documents for concept explanations or any general
   question that may be answered by the uploaded documents.
2. Use get_course_details when the user asks about a specific course's
   duration, prerequisites, learning resources, career opportunities, or
   what to learn next (roadmap). Pass the relevant 'aspect'.
3. Use generate_quiz for quizzes and practice questions.
4. Always prefer information retrieved from the documents (RAG).
5. Never hallucinate information. If it is not in the documents, say so.
""",
    )
    return agent


if __name__ == "__main__":
    agent = create_education_agent()
    print("Education agent ready with", len(tools), "tools.")

 
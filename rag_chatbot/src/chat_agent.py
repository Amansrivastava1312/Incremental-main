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
# RAG PIPELINE
# =====================================================

pipeline = RAGPipeline()

if not pipeline.chroma_dir.exists():
    print("Creating vector database...")
    pipeline.ingest()

# =====================================================
# RAG SEARCH
# =====================================================

def rag_search(query: str) -> str:

    context = pipeline.get_context(query)

    if not context or not context.strip():
        return "Information not found in the education documents."

    return context

# =====================================================
# TOOL 1
# =====================================================

@tool
def search_education_documents(query: str) -> str:
    """Search educational documents using RAG."""
    return rag_search(query)

# =====================================================
# TOOL 2
# =====================================================

@tool
def get_topic_information(topic: str) -> str:
    """Explain educational topics."""
    return rag_search(topic)

# =====================================================
# TOOL 3
# =====================================================

@tool
def get_learning_resources(topic: str) -> str:
    """Recommend learning resources."""

    resources = {
        "python": "Python Crash Course, W3Schools Python",
        "sql": "SQLBolt, PostgreSQL Documentation",
        "machine learning": "Hands-On Machine Learning Book",
        "java": "Oracle Java Documentation",
        "ai": "OpenAI Documentation"
    }

    return resources.get(
        topic.lower(),
        "Learning resources not available."
    )

# =====================================================
# TOOL 4
# =====================================================

@tool
def recommend_next_course(course: str) -> str:
    """Recommend next course."""

    roadmap = {
        "python": "Data Structures",
        "data structures": "Algorithms",
        "algorithms": "Machine Learning",
        "machine learning": "Deep Learning",
        "html": "CSS",
        "css": "JavaScript",
        "javascript": "React"
    }

    return roadmap.get(
        course.lower(),
        "No recommendation available."
    )

# =====================================================
# TOOL 5
# =====================================================

@tool
def get_course_duration(course: str) -> str:
    """Get course duration."""

    duration = {
        "python": "8 Weeks",
        "sql": "6 Weeks",
        "machine learning": "10 Weeks",
        "java": "8 Weeks",
        "ai": "12 Weeks"
    }

    return duration.get(
        course.lower(),
        "Duration not available."
    )

# =====================================================
# TOOL 6
# =====================================================

@tool
def get_course_prerequisites(course: str) -> str:
    """Get prerequisites for a course."""

    prerequisites = {
        "python": "Basic computer knowledge",
        "sql": "Basic computer knowledge",
        "machine learning": "Python and Mathematics",
        "deep learning": "Machine Learning"
    }

    return prerequisites.get(
        course.lower(),
        "Prerequisite not available."
    )

# =====================================================
# TOOL 7
# =====================================================

@tool
def generate_quiz(topic: str) -> str:
    """Generate quiz questions."""

    return f"""
Quiz on {topic}

1. What is {topic}?
2. Why do we use {topic}?
3. Give one real-world application of {topic}?
"""

# =====================================================
# TOOL 8
# =====================================================

@tool
def get_career_opportunities(course: str) -> str:
    """Suggest career opportunities."""

    careers = {
        "python": "Python Developer, Backend Developer",
        "sql": "Database Administrator, Data Analyst",
        "machine learning": "Machine Learning Engineer",
        "java": "Java Developer",
        "ai": "AI Engineer"
    }

    return careers.get(
        course.lower(),
        "Career information not available."
    )

# =====================================================
# TOOLS LIST
# =====================================================

tools = [
    search_education_documents,
    get_topic_information,
    get_learning_resources,
    recommend_next_course,
    get_course_duration,
    get_course_prerequisites,
    generate_quiz,
    get_career_opportunities,
]

# =====================================================
# CREATE AGENT
# =====================================================

def create_education_agent():

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
You are an Education AI Assistant.

Rules:

1. Use search_education_documents whenever information may exist in uploaded documents.

2. Use get_topic_information for concept explanations.

3. Use get_learning_resources for books, websites and study resources.

4. Use recommend_next_course when users ask what to learn next.

5. Use get_course_duration for duration related queries.

6. Use get_course_prerequisites for prerequisite questions.

7. Use generate_quiz for quizzes and practice questions.

8. Use get_career_opportunities for career guidance.

9. Prefer information from uploaded documents whenever possible.

10. Never hallucinate information.
"""
    )

    return agent
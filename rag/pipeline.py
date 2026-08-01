# pipeline.py

from rag.src.loader import DocumentLoader
from rag.src.chunking import Chunking
from rag.src.vector import Vectorizer
from rag.src.retrieval import Retriever
from rag.src.memory import Memory
from rag.src.generation import Generator

from langchain_huggingface import HuggingFaceEmbeddings
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "gemini-3.1-flash-lite"
API_KEY = os.getenv("GEMINI_API_KEY")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"


class RAGPipeline:

    def __init__(self):

        self.model_name = GENERATION_MODEL
        self.embedding_model = HuggingFaceEmbeddings(
                model_name = EMBEDDING_MODEL
                )
        self.chroma_dir = CHROMA_DIR
        self.api_key = API_KEY

        self.memory = Memory()

        self.generator = Generator(
            model_name=GENERATION_MODEL,
            api_key=API_KEY,
        )

        self.retriever = None

    def ingest(self):

        loader = DocumentLoader(DATA_DIR)
        documents = loader.load_documents()
        print(1)
        chunker = Chunking(
            documents,
            self.embedding_model,
        )
        chunks = chunker.semantic_chunking()
        print(1)
        vectorizer = Vectorizer(
            self.chroma_dir,
            self.embedding_model,
        )

        print(1)

        vector_store = vectorizer.build_vector(chunks)

        retriever_obj = vectorizer.create_retriever(vector_store)

        self.retriever = Retriever(retriever_obj)

    def get_context(self, question: str) -> str:

        if not Path(CHROMA_DIR).exists():
            print("chroma not exisit")
            self.ingest()

        if self.retriever is None:
            vectorizer = Vectorizer(
            self.chroma_dir,
            self.embedding_model,
            )

            vector_store = vectorizer.load_vector_store()

            retriever_obj = vectorizer.create_retriever(vector_store)

            self.retriever = Retriever(retriever_obj)

            

        rag_context = self.retriever.search_documents(question)

        return rag_context

    def get_answer(self, question: str) -> str:

        context = self.get_context(question)
        prompt = self.generator.build_prompt(question,self.memory.get_memory(),context)

        answer = self.generator.generate(prompt)

        self.memory.add_memory(
            question,
            answer[0]["text"],
        )

        return answer[0]["text"]

    def clear_memory(self):

        self.memory.clear()
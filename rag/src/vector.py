from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path


class Vectorizer:
    def __init__(self, CHROMA_DIR, embedding):
        self.CHROMA_DIR = CHROMA_DIR
        self.embedding = embedding

    def load_vector_store(self):
        vector_db = Chroma(
            persist_directory=self.CHROMA_DIR,
            embedding_function=self.embedding
        )
        return vector_db

    def build_vector(self, chunks = None):
        vector_db = Chroma.from_documents(
            documents=chunks,
            persist_directory=self.CHROMA_DIR,
            embedding=self.embedding
        )
        return vector_db
    
    def create_retriever(self, vector_store, k=3):
        return vector_store.as_retriever(
            search_kwargs={"k": k}
        )
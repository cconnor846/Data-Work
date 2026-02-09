"""
Construction Lessons-Learned RAG System

A pragmatic, Python-only proof-of-concept for semantic search 
over construction project lessons learned.
"""

__version__ = "0.1.0"

from .models import Lesson, lesson_to_canonical_text
from .ingest import load_lessons_from_excel
from .embeddings import create_embedding_client, EmbeddingClient
from .index import VectorIndex
from .simple_index import SimpleVectorIndex
from .retrieve import Retriever
from .generate import answer_question, LLMClient, StubLLMClient

__all__ = [
    "Lesson",
    "lesson_to_canonical_text",
    "load_lessons_from_excel",
    "create_embedding_client",
    "EmbeddingClient",
    "VectorIndex",
    "SimpleVectorIndex",
    "Retriever",
    "answer_question",
    "LLMClient",
    "StubLLMClient",
]

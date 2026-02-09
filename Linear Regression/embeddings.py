"""
Embedding client abstraction and implementations.
Default: sentence-transformers (local, free, semantic)
Alternative: OpenAI (API, best quality)
"""
from typing import List
import numpy as np
from abc import ABC, abstractmethod


class EmbeddingClient(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            float32 array of shape (n, dim) where n=len(texts)
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        pass


class SentenceTransformerClient(EmbeddingClient):
    """
    Local semantic embeddings using sentence-transformers.
    
    This is FREE, runs locally, and provides TRUE semantic search.
    No API calls, no costs, actual understanding of meaning.
    
    Usage:
        client = SentenceTransformerClient()
        vectors = client.embed_texts(["structural issues", "design errors"])
    
    The model downloads once (~100MB) then runs locally.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Args:
            model_name: sentence-transformers model to use
                - 'all-MiniLM-L6-v2': 384 dim, fast, good quality (default)
                - 'all-mpnet-base-v2': 768 dim, slower, better quality
                - 'paraphrase-MiniLM-L6-v2': 384 dim, good for paraphrasing
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Install with: pip install sentence-transformers"
            )
        
        print(f"Loading sentence-transformers model: {model_name}")
        print("(Downloads ~100MB on first run, then runs locally)")
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
        # Set dimension based on model
        if 'MiniLM-L6' in model_name:
            self._dim = 384
        elif 'mpnet-base' in model_name:
            self._dim = 768
        else:
            # Get dimension from model
            self._dim = self.model.get_sentence_embedding_dimension()
        
        print(f"✓ Model loaded: {self._dim} dimensions")
    
    @property
    def embedding_dim(self) -> int:
        return self._dim
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts using sentence-transformers.
        
        This runs locally - no API calls, completely free.
        """
        # Convert to embeddings (returns numpy array)
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embeddings.astype(np.float32)


class StubEmbeddingClient(EmbeddingClient):
    """
    Stub embedder for testing the pipeline.
    Uses simple TF-IDF-like approach with hashing.
    
    DO NOT use in production - replace with OpenAI/Cohere/etc.
    """
    
    def __init__(self, dim: int = 384):
        """
        Args:
            dim: Embedding dimensionality (default 384, similar to sentence-transformers)
        """
        self._dim = dim
    
    @property
    def embedding_dim(self) -> int:
        return self._dim
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Create deterministic but meaningless embeddings for testing.
        Uses hash-based approach - similar texts will have similar embeddings.
        """
        embeddings = []
        
        for text in texts:
            # Simple hash-based embedding
            # Split into words, hash each, aggregate
            words = text.lower().split()
            
            # Initialize embedding vector
            vec = np.zeros(self._dim, dtype=np.float32)
            
            # Add contribution from each word
            for word in words:
                # Hash word to indices
                word_hash = hash(word)
                idx = abs(word_hash) % self._dim
                vec[idx] += 1.0
                
                # Add second hash for better distribution
                idx2 = abs(word_hash >> 8) % self._dim
                vec[idx2] += 0.5
            
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            
            embeddings.append(vec)
        
        return np.array(embeddings, dtype=np.float32)


class OpenAIEmbeddingClient(EmbeddingClient):
    """
    OpenAI embeddings client.
    
    Example usage (when you have API key):
        client = OpenAIEmbeddingClient(
            api_key="sk-...",
            model="text-embedding-3-small"
        )
    """
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Args:
            api_key: OpenAI API key
            model: Embedding model name
                - text-embedding-3-small: 1536 dim, cheaper
                - text-embedding-3-large: 3072 dim, better quality
        """
        self.api_key = api_key
        self.model = model
        
        # Set dimension based on model
        self._dim = 1536 if "small" in model else 3072
        
        # NOTE: Actual implementation would initialize OpenAI client here
        # from openai import OpenAI
        # self.client = OpenAI(api_key=api_key)
        
        raise NotImplementedError(
            "OpenAI client requires openai package and API key setup. "
            "Use StubEmbeddingClient for now."
        )
    
    @property
    def embedding_dim(self) -> int:
        return self._dim
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts using OpenAI API.
        
        Implementation would be:
        
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)
        """
        raise NotImplementedError("See class docstring")


# Factory function for easy swapping
def create_embedding_client(
    provider: str = "sentence-transformers",
    **kwargs
) -> EmbeddingClient:
    """
    Factory to create embedding clients.
    
    Args:
        provider: "sentence-transformers", "openai", or "stub"
        **kwargs: Provider-specific arguments
        
    Returns:
        EmbeddingClient instance
        
    Examples:
        # Default: Local semantic embeddings (FREE)
        embedder = create_embedding_client()
        
        # OpenAI (requires API key)
        embedder = create_embedding_client(
            provider="openai",
            api_key="sk-..."
        )
        
        # Stub (testing only)
        embedder = create_embedding_client(provider="stub")
    """
    if provider == "sentence-transformers":
        return SentenceTransformerClient(**kwargs)
    elif provider == "stub":
        return StubEmbeddingClient(**kwargs)
    elif provider == "openai":
        return OpenAIEmbeddingClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'sentence-transformers', 'openai', or 'stub'")

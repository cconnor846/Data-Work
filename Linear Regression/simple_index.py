"""
Simple vector index implementation without FAISS dependency.
Uses numpy for basic similarity search - good enough for POC.

For production, replace with FAISS implementation in index.py
"""
from typing import List, Dict, Any, Optional
import numpy as np
import json
from pathlib import Path
import pickle


class SimpleVectorIndex:
    """
    Simple in-memory vector index using numpy.
    
    Good for POC/testing with <10k documents.
    For production, use FAISSVectorIndex from index.py
    """
    
    def __init__(self, dimension: int):
        """
        Args:
            dimension: Embedding dimensionality
        """
        self.dimension = dimension
        
        # Store everything in numpy arrays and lists
        self.vectors: Optional[np.ndarray] = None
        self.metadatas: List[Dict[str, Any]] = []
        self.texts: List[str] = []
    
    def add(
        self,
        vectors: np.ndarray,
        metadatas: List[Dict[str, Any]],
        texts: List[str]
    ) -> None:
        """
        Add vectors with metadata to the index.
        
        Args:
            vectors: float32 array of shape (n, dim)
            metadatas: List of metadata dicts (length n)
            texts: List of text strings (length n)
        """
        assert vectors.shape[0] == len(metadatas) == len(texts), \
            "Vectors, metadatas, and texts must have same length"
        assert vectors.shape[1] == self.dimension, \
            f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}"
        
        # Ensure float32
        vectors = vectors.astype(np.float32)
        
        # L2 normalize for cosine similarity
        vectors = self._normalize_vectors(vectors)
        
        # Add to index
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        
        # Store metadata and texts
        self.metadatas.extend(metadatas)
        self.texts.extend(texts)
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_vector: float32 array of shape (1, dim) or (dim,)
            k: Number of results to return
            score_threshold: Minimum similarity score (optional)
            
        Returns:
            List of dicts with keys: metadata, text, score, index
        """
        if self.vectors is None or len(self.vectors) == 0:
            return []
        
        # Ensure correct shape
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Ensure float32
        query_vector = query_vector.astype(np.float32)
        
        # L2 normalize
        query_vector = self._normalize_vectors(query_vector)
        
        # Compute cosine similarities (dot product of normalized vectors)
        # Shape: (1, dim) @ (dim, n) = (1, n)
        similarities = np.dot(query_vector, self.vectors.T).flatten()
        
        # Get top-k indices
        # argsort returns ascending order, so we reverse it
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            
            # Apply threshold if provided
            if score_threshold is not None and score < score_threshold:
                continue
            
            results.append({
                "metadata": self.metadatas[idx],
                "text": self.texts[idx],
                "score": score,
                "index": int(idx),
            })
        
        return results
    
    def save(self, save_dir: str | Path) -> None:
        """
        Persist index to disk.
        
        Args:
            save_dir: Directory to save to (will be created if needed)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all data
        data = {
            "vectors": self.vectors,
            "metadatas": self.metadatas,
            "texts": self.texts,
            "dimension": self.dimension,
        }
        
        save_path = save_dir / "index.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Saved index to {save_dir}")
        print(f"  - {len(self.metadatas)} documents")
        print(f"  - {self.dimension} dimensions")
    
    @classmethod
    def load(cls, save_dir: str | Path) -> "SimpleVectorIndex":
        """
        Load index from disk.
        
        Args:
            save_dir: Directory to load from
            
        Returns:
            Loaded SimpleVectorIndex instance
        """
        save_dir = Path(save_dir)
        
        if not save_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {save_dir}")
        
        # Load data
        save_path = save_dir / "index.pkl"
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        
        # Create instance
        instance = cls(dimension=data["dimension"])
        instance.vectors = data["vectors"]
        instance.metadatas = data["metadatas"]
        instance.texts = data["texts"]
        
        print(f"Loaded index from {save_dir}")
        print(f"  - {len(instance.metadatas)} documents")
        print(f"  - {instance.dimension} dimensions")
        
        return instance
    
    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def __len__(self) -> int:
        """Return number of vectors in index."""
        return len(self.metadatas)


# Use this as the default for POC
VectorIndex = SimpleVectorIndex

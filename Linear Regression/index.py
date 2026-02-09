"""
Vector index implementation using FAISS.
Handles storage, retrieval, and persistence of embeddings + metadata.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import json
from pathlib import Path
import pickle


class FAISSVectorIndex:
    """
    Vector index using FAISS for similarity search.
    Stores vectors in FAISS and metadata separately.
    
    Design choices:
    - Use IndexFlatIP (inner product) with normalized vectors = cosine similarity
    - Store metadata in parallel list (simple, debuggable)
    - Persist both FAISS index and metadata
    """
    
    def __init__(self, dimension: int):
        """
        Args:
            dimension: Embedding dimensionality
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed. Install with: pip install faiss-cpu"
            )
        
        self.dimension = dimension
        self.faiss = faiss
        
        # Use IndexFlatIP for exact inner product search
        # With L2-normalized vectors, IP = cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        
        # Store metadata and texts in parallel arrays
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
        
        # Add to FAISS
        self.index.add(vectors)
        
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
        Search for similar vectors.
        
        Args:
            query_vector: float32 array of shape (1, dim) or (dim,)
            k: Number of results to return
            score_threshold: Minimum similarity score (optional)
            
        Returns:
            List of dicts with keys: metadata, text, score, index
        """
        # Ensure correct shape
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Ensure float32
        query_vector = query_vector.astype(np.float32)
        
        # L2 normalize
        query_vector = self._normalize_vectors(query_vector)
        
        # Search
        # scores are cosine similarities (higher = more similar)
        scores, indices = self.index.search(query_vector, k)
        
        # Convert to list of results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # FAISS returns -1 for not found
            if idx == -1:
                continue
            
            # Apply threshold if provided
            if score_threshold is not None and score < score_threshold:
                continue
            
            results.append({
                "metadata": self.metadatas[idx],
                "text": self.texts[idx],
                "score": float(score),
                "index": int(idx),
            })
        
        return results
    
    def save(self, save_dir: str | Path) -> None:
        """
        Persist index and metadata to disk.
        
        Args:
            save_dir: Directory to save to (will be created if needed)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = save_dir / "faiss.index"
        self.faiss.write_index(self.index, str(index_path))
        
        # Save metadata and texts
        data = {
            "metadatas": self.metadatas,
            "texts": self.texts,
            "dimension": self.dimension,
        }
        
        metadata_path = save_dir / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Saved index to {save_dir}")
        print(f"  - {len(self.metadatas)} documents")
        print(f"  - {self.dimension} dimensions")
    
    @classmethod
    def load(cls, save_dir: str | Path) -> "FAISSVectorIndex":
        """
        Load index and metadata from disk.
        
        Args:
            save_dir: Directory to load from
            
        Returns:
            Loaded FAISSVectorIndex instance
        """
        save_dir = Path(save_dir)
        
        if not save_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {save_dir}")
        
        # Load metadata first to get dimension
        metadata_path = save_dir / "metadata.pkl"
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
        
        # Create instance
        instance = cls(dimension=data["dimension"])
        
        # Load FAISS index
        index_path = save_dir / "faiss.index"
        instance.index = instance.faiss.read_index(str(index_path))
        
        # Restore metadata
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


# Type alias for the vector index interface
# Try to use FAISS if available, otherwise fall back to simple index
try:
    import faiss
    VectorIndex = FAISSVectorIndex
    print("Using FAISS for vector indexing")
except ImportError:
    from .simple_index import SimpleVectorIndex
    VectorIndex = SimpleVectorIndex
    print("FAISS not available - using simple numpy-based index (good for POC)")

"""
Retrieval logic with filtering and scoring.
Orchestrates embedding + vector search + filtering.
"""
from typing import List, Dict, Any, Optional
import re
from .embeddings import EmbeddingClient
from .index import VectorIndex


# Known values for deterministic filtering
DISCIPLINES = {
    "electrical", "mechanical", "civil", "piping", 
    "instrumentation", "controls", "structural", "hvac"
}

PHASES = {
    "design", "procurement", "construction", 
    "commissioning", "startup", "closeout"
}

PROJECT_TYPES = {
    "industrial", "commercial", "residential", 
    "infrastructure", "renovation", "new_build"
}


class Retriever:
    """
    Orchestrates semantic retrieval with optional metadata filtering.
    
    Flow:
    1. Embed query
    2. Vector search
    3. Apply metadata filters
    4. Apply score threshold
    5. Return ranked results
    """
    
    def __init__(
        self,
        embedder: EmbeddingClient,
        index: VectorIndex,
        default_k: int = 5,
        score_threshold: float = 0.05  # Lower threshold for stub embedder
    ):
        """
        Args:
            embedder: Embedding client
            index: Vector index
            default_k: Default number of results
            score_threshold: Minimum similarity score (0-1, cosine similarity)
        """
        self.embedder = embedder
        self.index = index
        self.default_k = default_k
        self.score_threshold = score_threshold
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        auto_filter: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant lessons for a query.
        
        Args:
            query: User question
            k: Number of results (uses default if None)
            filters: Explicit metadata filters (e.g., {"discipline": "electrical"})
            auto_filter: If True, extract filters from query text
            
        Returns:
            List of hits with metadata, text, and score
        """
        if k is None:
            k = self.default_k
        
        # Combine explicit and auto-extracted filters
        all_filters = filters or {}
        if auto_filter:
            auto_filters = extract_filters_from_query(query)
            # Explicit filters take precedence
            for key, value in auto_filters.items():
                if key not in all_filters:
                    all_filters[key] = value
        
        # Log filters if any were detected
        if all_filters:
            print(f"Applying filters: {all_filters}")
        
        # Embed query
        query_vec = self.embedder.embed_texts([query])
        
        # Search vector index
        # Request more results than k to allow for filtering
        search_k = k * 3 if all_filters else k
        hits = self.index.search(
            query_vector=query_vec,
            k=search_k,
            score_threshold=self.score_threshold
        )
        
        # Apply metadata filters
        if all_filters:
            hits = [h for h in hits if metadata_matches(h["metadata"], all_filters)]
        
        # Limit to k results
        hits = hits[:k]
        
        return hits
    
    def retrieve_with_context(
        self,
        query: str,
        k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieve with additional context for debugging.
        
        Returns:
            Dict with hits, filters, scores, etc.
        """
        filters = kwargs.get("filters")
        auto_filter = kwargs.get("auto_filter", True)
        
        # Extract filters for logging
        all_filters = filters or {}
        if auto_filter:
            auto_filters = extract_filters_from_query(query)
            all_filters.update(auto_filters)
        
        # Retrieve
        hits = self.retrieve(query, k=k, **kwargs)
        
        return {
            "query": query,
            "hits": hits,
            "filters_applied": all_filters,
            "num_results": len(hits),
            "top_score": hits[0]["score"] if hits else None,
        }


def extract_filters_from_query(query: str) -> Dict[str, Any]:
    """
    Extract metadata filters from query text using keyword matching.
    
    This is deterministic and explainable - no NLP magic.
    
    Args:
        query: User question
        
    Returns:
        Dict of filters (e.g., {"discipline": "electrical"})
    """
    query_lower = query.lower()
    filters = {}
    
    # Check for disciplines
    for discipline in DISCIPLINES:
        if discipline in query_lower:
            filters["discipline"] = discipline.title()
            break  # Take first match
    
    # Check for phases
    for phase in PHASES:
        if phase in query_lower:
            filters["phase"] = phase.title()
            break
    
    # Check for project types
    for project_type in PROJECT_TYPES:
        # Handle multi-word types
        pattern = project_type.replace("_", " ")
        if pattern in query_lower:
            filters["project_type"] = project_type.replace("_", " ").title()
            break
    
    # Check for vendor mentions
    # Only filter if query explicitly mentions a vendor name
    vendor_pattern = r'vendor[:\s]+([a-zA-Z0-9\s&]+?)(?:\s|$|,|\?)'
    vendor_match = re.search(vendor_pattern, query, re.IGNORECASE)
    if vendor_match:
        filters["vendor"] = vendor_match.group(1).strip().title()
    
    return filters


def metadata_matches(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """
    Check if metadata matches all filters.
    
    Args:
        metadata: Document metadata
        filters: Required filter values
        
    Returns:
        True if all filters match
    """
    for key, value in filters.items():
        # Case-insensitive comparison
        meta_val = str(metadata.get(key, "")).lower()
        filter_val = str(value).lower()
        
        if meta_val != filter_val:
            return False
    
    return True


def apply_score_threshold(
    hits: List[Dict[str, Any]],
    threshold: float
) -> List[Dict[str, Any]]:
    """
    Filter hits by minimum score.
    
    Args:
        hits: List of search results
        threshold: Minimum score
        
    Returns:
        Filtered list
    """
    return [h for h in hits if h["score"] >= threshold]


def compute_score_stats(hits: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute statistics about retrieval scores.
    
    Useful for calibrating thresholds.
    """
    if not hits:
        return {"min": 0, "max": 0, "mean": 0, "median": 0}
    
    scores = [h["score"] for h in hits]
    scores.sort()
    
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": sum(scores) / len(scores),
        "median": scores[len(scores) // 2],
    }

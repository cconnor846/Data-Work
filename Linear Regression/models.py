"""
Data models for the construction lessons-learned RAG system.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class Lesson:
    """
    Immutable lesson record with all required and optional fields.
    
    Fields match the real lessons-learned database schema:
    - lesson_id: Unique identifier (IDENTIFIER column)
    - title: Brief summary (TITLE column)
    - description: What happened / the issue (DESCRIPTION column)
    - actions_taken: What was done to address it (ACTIONS TAKEN column)
    - proposed_solution: Planned fixes (PROPOSED SOLUTION column)
    - operations: Area of work / discipline (OPERATIONS INVOLVED column)
    - business_area: Business area (BUSINESS AREA INVOLVED column)
    - project: Project name (PROJECT column)
    - event_date: When it occurred (EVENT DATE column)
    - identified_by: Who reported it (IDENTIFIED BY column)
    """
    lesson_id: str
    title: str
    description: str
    operations: str
    business_area: str
    project: str
    actions_taken: Optional[str] = None
    proposed_solution: Optional[str] = None
    event_date: Optional[str] = None
    identified_by: Optional[str] = None

    def to_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata for vector index storage.
        Excludes the text content fields (title, description, actions, solution).
        """
        return {
            "lesson_id": self.lesson_id,
            "operations": self.operations,
            "business_area": self.business_area,
            "project": self.project,
            "event_date": self.event_date or "Unknown",
            "identified_by": self.identified_by or "Unknown",
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


def lesson_to_canonical_text(lesson: Lesson) -> str:
    """
    Convert a lesson to canonical text format for embedding and retrieval.
    
    Format:
    - Metadata first (for context)
    - Content second (title/description/actions/solution)
    
    This is the single document that gets embedded and retrieved.
    Keeping it readable helps with debugging and LLM grounding.
    """
    parts = [
        f"Lesson ID: {lesson.lesson_id}",
        f"Project: {lesson.project}",
        f"Operations: {lesson.operations}",
        f"Business Area: {lesson.business_area}",
        f"Date: {lesson.event_date or 'Unknown'}",
        f"Identified By: {lesson.identified_by or 'Unknown'}",
        "",
        f"Title: {lesson.title}",
        "",
        f"Description: {lesson.description}",
    ]
    
    # Add actions if present
    if lesson.actions_taken:
        parts.extend(["", f"Actions Taken: {lesson.actions_taken}"])
    
    # Add proposed solution if present
    if lesson.proposed_solution:
        parts.extend(["", f"Proposed Solution: {lesson.proposed_solution}"])
    
    return "\n".join(parts)

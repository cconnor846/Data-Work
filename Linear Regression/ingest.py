"""
Data ingestion from Excel files.
Handles loading, validation, and normalization of lessons-learned data.
"""
import pandas as pd
from typing import List, Optional
from pathlib import Path
from .models import Lesson


# Required columns in the Excel file
REQUIRED_COLUMNS = [
    "IDENTIFIER",      # lesson_id
    "TITLE",          # title
    "DESCRIPTION",    # description
    "OPERATIONS INVOLVED",     # operations (discipline)
    "BUSINESS AREA INVOLVED",  # business_area
    "PROJECT",        # project
]

# Optional columns
OPTIONAL_COLUMNS = [
    "ACTIONS TAKEN",   # actions_taken
    "PROPOSED SOLUTION",  # proposed_solution
    "EVENT DATE",      # event_date
    "IDENTIFIED BY",   # identified_by
]


def load_lessons_from_excel(excel_path: str | Path) -> List[Lesson]:
    """
    Load lessons from an Excel file.
    
    Args:
        excel_path: Path to Excel file containing lessons
        
    Returns:
        List of validated Lesson objects
        
    Raises:
        ValueError: If required columns are missing or data is invalid
        FileNotFoundError: If Excel file doesn't exist
    """
    excel_path = Path(excel_path)
    
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    # Load Excel - handle both .xlsx and .xls
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}")
    
    # Validate required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )
    
    # Note: Keep original column names (don't lowercase) since they're uppercase in the file
    
    # Ensure optional columns exist (fill with None if missing)
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    
    # Normalize data
    lessons = []
    for idx, row in df.iterrows():
        try:
            lesson = _row_to_lesson(row, idx)
            lessons.append(lesson)
        except Exception as e:
            print(f"Warning: Skipping row {idx} due to error: {e}")
            continue
    
    if not lessons:
        raise ValueError("No valid lessons loaded from Excel file")
    
    print(f"Successfully loaded {len(lessons)} lessons from {excel_path}")
    return lessons


def _row_to_lesson(row: pd.Series, idx: int) -> Lesson:
    """
    Convert a DataFrame row to a Lesson object.
    
    Args:
        row: Pandas Series representing one row
        idx: Row index (for error messages)
        
    Returns:
        Validated Lesson object
    """
    # Handle NaN values
    def clean_str(val) -> str:
        if pd.isna(val):
            return ""
        return str(val).strip()
    
    def clean_optional_str(val) -> Optional[str]:
        if pd.isna(val):
            return None
        s = str(val).strip()
        return s if s else None
    
    # Extract and validate required fields
    lesson_id = clean_str(row["IDENTIFIER"])
    title = clean_str(row["TITLE"])
    description = clean_str(row["DESCRIPTION"])
    operations = clean_str(row["OPERATIONS INVOLVED"])
    business_area = clean_str(row["BUSINESS AREA INVOLVED"])
    project = clean_str(row["PROJECT"])
    
    # Validate required fields are not empty
    if not all([lesson_id, title, description, operations, business_area, project]):
        raise ValueError(f"Row {idx}: Required field is empty")
    
    # Handle optional fields
    actions_taken = clean_optional_str(row.get("ACTIONS TAKEN"))
    proposed_solution = clean_optional_str(row.get("PROPOSED SOLUTION"))
    identified_by = clean_optional_str(row.get("IDENTIFIED BY"))
    
    # Handle event date - convert to ISO string if it's a datetime
    date_val = row.get("EVENT DATE")
    if pd.notna(date_val):
        if hasattr(date_val, "isoformat"):
            event_date = date_val.isoformat().split('T')[0]  # Just the date part
        else:
            event_date = str(date_val).strip()
    else:
        event_date = None
    
    return Lesson(
        lesson_id=lesson_id,
        title=title,
        description=description,
        operations=operations,
        business_area=business_area,
        project=project,
        actions_taken=actions_taken,
        proposed_solution=proposed_solution,
        event_date=event_date,
        identified_by=identified_by,
    )


def validate_lessons(lessons: List[Lesson]) -> bool:
    """
    Run validation checks on loaded lessons.
    
    Args:
        lessons: List of lessons to validate
        
    Returns:
        True if all validations pass
        
    Raises:
        ValueError: If validation fails
    """
    if not lessons:
        raise ValueError("No lessons to validate")
    
    # Check for duplicate lesson IDs
    lesson_ids = [l.lesson_id for l in lessons]
    duplicates = set([x for x in lesson_ids if lesson_ids.count(x) > 1])
    if duplicates:
        raise ValueError(f"Duplicate lesson IDs found: {duplicates}")
    
    # Check that all required fields are non-empty
    for lesson in lessons:
        if not all([
            lesson.lesson_id,
            lesson.title,
            lesson.description,
            lesson.operations,
            lesson.business_area,
            lesson.project,
        ]):
            raise ValueError(f"Lesson {lesson.lesson_id} has empty required fields")
    
    print(f"Validation passed for {len(lessons)} lessons")
    return True

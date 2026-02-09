"""
Generate example lessons-learned Excel file for testing.
"""
import pandas as pd
from pathlib import Path


def create_example_data():
    """Create sample lessons learned data."""
    
    lessons = [
        {
            "lesson_id": "LESSON_0001",
            "mistake": "Electrical conduit installed before structural framing complete",
            "reason": "Poor coordination between electrical and structural teams. Schedule pressure led to premature installation.",
            "resolution": "Conduit had to be relocated after framing. Implemented mandatory coordination meetings every Monday.",
            "discipline": "Electrical",
            "project_type": "Commercial",
            "phase": "Construction",
            "date": "2024-03-15",
            "vendor": "ABC Electric"
        },
        {
            "lesson_id": "LESSON_0002",
            "mistake": "Mechanical HVAC ductwork clashed with sprinkler piping",
            "reason": "3D coordination model was not updated with latest sprinkler revisions",
            "resolution": "Redesigned duct routing in field. Now require BIM updates within 48 hours of any change.",
            "discipline": "Mechanical",
            "project_type": "Industrial",
            "phase": "Construction",
            "date": "2024-02-20",
            "vendor": "Cool Air Systems"
        },
        {
            "lesson_id": "LESSON_0003",
            "mistake": "Civil grading did not account for required drainage slopes",
            "reason": "Site survey data was outdated by 6 months. Ground conditions had changed.",
            "resolution": "Re-graded entire site. Now require fresh survey within 30 days of mobilization.",
            "discipline": "Civil",
            "project_type": "Infrastructure",
            "phase": "Construction",
            "date": "2024-01-10",
            "vendor": None
        },
        {
            "lesson_id": "LESSON_0004",
            "mistake": "Instrumentation cables pulled before cable tray was inspected",
            "reason": "Contractor started work before receiving QC approval due to schedule pressure",
            "resolution": "Some cables had to be re-pulled. Implemented hard stop: no work without QC signoff.",
            "discipline": "Instrumentation",
            "project_type": "Industrial",
            "phase": "Construction",
            "date": "2024-04-05",
            "vendor": "Precision Controls Inc"
        },
        {
            "lesson_id": "LESSON_0005",
            "mistake": "Electrical panel locations conflicted with door swing clearances",
            "reason": "Panel locations were designed without field verification of actual door hardware",
            "resolution": "Panels relocated at significant cost. Now require field verification during design phase.",
            "discipline": "Electrical",
            "project_type": "Commercial",
            "phase": "Design",
            "date": "2023-11-30",
            "vendor": None
        },
        {
            "lesson_id": "LESSON_0006",
            "mistake": "Piping supports designed for wrong load conditions",
            "reason": "Design assumed empty pipe weight, did not account for fluid density during operation",
            "resolution": "All supports had to be reinforced. Added load calculation review checklist.",
            "discipline": "Piping",
            "project_type": "Industrial",
            "phase": "Design",
            "date": "2024-02-28",
            "vendor": None
        },
        {
            "lesson_id": "LESSON_0007",
            "mistake": "Control system programming started before final P&IDs approved",
            "reason": "Programming team wanted to get ahead of schedule, assumed P&IDs wouldn't change",
            "resolution": "30% of logic had to be rewritten. Now enforce design freeze before programming starts.",
            "discipline": "Controls",
            "project_type": "Industrial",
            "phase": "Design",
            "date": "2024-03-20",
            "vendor": "AutomationPro"
        },
        {
            "lesson_id": "LESSON_0008",
            "mistake": "Structural steel arrived 3 weeks late causing project delay",
            "reason": "Vendor was not given formal notice to proceed, operated on verbal authorization only",
            "resolution": "Implemented formal PO and NTP system. All vendors now require written authorization.",
            "discipline": "Structural",
            "project_type": "Commercial",
            "phase": "Procurement",
            "date": "2024-01-25",
            "vendor": "SteelCo Industries"
        },
        {
            "lesson_id": "LESSON_0009",
            "mistake": "Commissioning revealed 40% of instrumentation out of calibration",
            "reason": "Instruments sat in warehouse for 8 months before installation. Calibration dates expired.",
            "resolution": "All instruments recalibrated. Now track calibration dates in procurement system.",
            "discipline": "Instrumentation",
            "project_type": "Industrial",
            "phase": "Commissioning",
            "date": "2024-04-15",
            "vendor": "Precision Controls Inc"
        },
        {
            "lesson_id": "LESSON_0010",
            "mistake": "HVAC system failed to maintain temperature during startup",
            "reason": "Design calculations used incorrect outdoor air temperature assumptions",
            "resolution": "Added supplemental cooling capacity. Now require site-specific weather data for all HVAC designs.",
            "discipline": "Mechanical",
            "project_type": "Commercial",
            "phase": "Startup",
            "date": "2024-05-01",
            "vendor": "Cool Air Systems"
        },
    ]
    
    # Create DataFrame
    df = pd.DataFrame(lessons)
    
    # Save to Excel
    output_path = Path(__file__).parent / "example_lessons.xlsx"
    df.to_excel(output_path, index=False, engine='openpyxl')
    
    print(f"Created example data: {output_path}")
    print(f"Contains {len(lessons)} lessons")
    
    return output_path


if __name__ == "__main__":
    create_example_data()

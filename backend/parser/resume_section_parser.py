"""
Resume Section Parser - Extract skills from explicitly labeled resume sections
This handles resumes where skills are listed under clear headers like "TECHNICAL SKILLS"
"""

import re
from typing import Dict, List


def extract_skills_from_sections(resume_text: str) -> Dict[str, List[str]]:
    """
    Extract skills from labeled sections in resume

    Args:
        resume_text: Full resume text

    Returns:
        Dictionary of categorized skills
    """

    # Find the TECHNICAL SKILLS section
    # Look for section header, then capture everything until next major section (all caps line, no colon)
    tech_skills_pattern = r'(?:TECHNICAL\s+SKILLS|SKILLS|TECHNOLOGIES|TECH\s+STACK)\s*\n(.*?)(?=\n\s*[A-Z][A-Z\s]{5,}\s*\n|\Z)'
    match = re.search(tech_skills_pattern, resume_text, re.DOTALL | re.IGNORECASE)

    if not match:
        return {}

    skills_section = match.group(1)

    # Extract subsections with patterns like "Programming:", "Data Analysis:", etc.
    subsection_pattern = r'([A-Za-z\s&]+):\s*([^\n]+)'
    subsections = re.findall(subsection_pattern, skills_section)

    categorized_skills = {
        'programming_languages': [],
        'frameworks_libraries': [],
        'tools_software': [],
        'databases': [],
        'soft_skills': []
    }

    for header, skills_text in subsections:
        header_lower = header.lower().strip()

        # Split skills by common delimiters
        skills = re.split(r'[,;|]', skills_text)
        skills = [s.strip() for s in skills if s.strip()]

        # Categorize based on section header
        if any(keyword in header_lower for keyword in ['programming', 'language', 'coding']):
            categorized_skills['programming_languages'].extend(skills)

        elif any(keyword in header_lower for keyword in ['framework', 'library', 'libraries']):
            categorized_skills['frameworks_libraries'].extend(skills)

        elif any(keyword in header_lower for keyword in ['data analysis', 'ml', 'machine learning', 'ai', 'analytics']):
            # Data science libraries go into frameworks
            categorized_skills['frameworks_libraries'].extend(skills)

        elif any(keyword in header_lower for keyword in ['visualization', 'viz', 'reporting']):
            categorized_skills['frameworks_libraries'].extend(skills)

        elif any(keyword in header_lower for keyword in ['tool', 'software', 'platform', 'environment']):
            categorized_skills['tools_software'].extend(skills)

        elif any(keyword in header_lower for keyword in ['database', 'db', 'data store']):
            categorized_skills['databases'].extend(skills)

        elif any(keyword in header_lower for keyword in ['design', 'creative']):
            categorized_skills['tools_software'].extend(skills)

        else:
            # Default: if it mentions specific known tools, categorize accordingly
            categorized_skills['tools_software'].extend(skills)

    return categorized_skills


def merge_skill_dicts(nlp_skills: Dict, section_skills: Dict) -> Dict:
    """
    Merge skills from NLP extraction and section extraction
    Section skills take priority

    Args:
        nlp_skills: Skills from NLP extraction
        section_skills: Skills from section parsing

    Returns:
        Merged skill dictionary
    """
    merged = nlp_skills.copy()

    for category in section_skills:
        if category in merged:
            # Combine and deduplicate
            all_skills = list(set(
                merged[category] + section_skills[category]
            ))
            merged[category] = all_skills
        else:
            merged[category] = section_skills[category]

    return merged

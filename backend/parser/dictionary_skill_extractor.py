"""
Dictionary-Based Skill Extractor
Simple and accurate skill extraction using comprehensive skill dictionary
"""

import re
from typing import Dict, List
from skill_dictionary import SKILL_DICTIONARY


def extract_skills_from_text(text: str) -> Dict[str, List[str]]:
    """
    Extract skills from text by matching against skill dictionary

    Args:
        text: Resume text or any text to extract skills from

    Returns:
        Dictionary of categorized skills found in the text
    """
    text_lower = text.lower()

    found_skills = {
        'programming_languages': [],
        'frameworks_libraries': [],
        'tools_software': [],
        'databases': [],
        'soft_skills': []
    }

    for category, skills_list in SKILL_DICTIONARY.items():
        for skill in skills_list:
            # Special handling for single-letter programming languages (C and R)
            if skill in ['c', 'r'] and len(skill) == 1:
                # Match only when uppercase or surrounded by specific patterns
                # C: look for " C," or " C " or " C++"
                if skill == 'c':
                    if re.search(r'[\s,;:]C[\s,;:]', text) or re.search(r'[\s,;:]C$', text):
                        found_skills[category].append(skill)
                # R: look for " R," or " R "
                elif skill == 'r':
                    if re.search(r'[\s,;:]R[\s,;:]', text) or re.search(r'[\s,;:]R$', text):
                        found_skills[category].append(skill)
                continue

            # Create word boundary pattern for exact matching
            # Handle special regex characters
            escaped_skill = re.escape(skill)

            # Create pattern that matches the skill as a whole word/phrase
            # Allows for surrounding punctuation, whitespace, or parentheses
            pattern = r'(?:^|[\s,;:()\[\]{}|/<>"\'])' + escaped_skill + r'(?:$|[\s,;:()\[\]{}|/<>"\'])'

            if re.search(pattern, text_lower, re.IGNORECASE):
                # Store the skill in proper case (first occurrence format)
                found_skills[category].append(skill)

    # Remove duplicates while preserving order
    for category in found_skills:
        seen = set()
        unique_skills = []
        for skill in found_skills[category]:
            skill_normalized = skill.lower()
            if skill_normalized not in seen:
                seen.add(skill_normalized)
                unique_skills.append(skill)
        found_skills[category] = unique_skills

    return found_skills


def extract_skills_fuzzy(text: str) -> Dict[str, List[str]]:
    """
    More lenient extraction that handles variations
    Example: "React.js" matches "react", "Next" matches "next.js"

    Args:
        text: Resume text

    Returns:
        Dictionary of categorized skills
    """
    text_lower = text.lower()

    # Remove common non-alphanumeric except dots, pluses, hashes, hyphens
    text_normalized = re.sub(r'[^\w\s.+#-]', ' ', text_lower)

    found_skills = {
        'programming_languages': [],
        'frameworks_libraries': [],
        'tools_software': [],
        'databases': [],
        'soft_skills': []
    }

    for category, skills_list in SKILL_DICTIONARY.items():
        for skill in skills_list:
            # Normalize skill for fuzzy matching
            skill_normalized = skill.replace('.', '').replace('-', '').replace(' ', '').lower()
            text_search = text_normalized.replace('.', '').replace('-', '').replace(' ', '').lower()

            # For very short skills (1-2 chars), require exact word boundary
            if len(skill_normalized) <= 2:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills[category].append(skill)
            else:
                # For longer skills, allow fuzzy matching
                if skill_normalized in text_search:
                    found_skills[category].append(skill)

    # Remove duplicates
    for category in found_skills:
        found_skills[category] = list(set(found_skills[category]))

    return found_skills


def merge_extracted_skills(dict1: Dict[str, List[str]], dict2: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Merge two skill dictionaries, removing duplicates

    Args:
        dict1: First skill dictionary
        dict2: Second skill dictionary

    Returns:
        Merged dictionary
    """
    merged = {}

    all_categories = set(list(dict1.keys()) + list(dict2.keys()))

    for category in all_categories:
        skills1 = dict1.get(category, [])
        skills2 = dict2.get(category, [])

        # Combine and deduplicate
        combined = skills1 + skills2
        seen = set()
        unique = []

        for skill in combined:
            skill_lower = skill.lower()
            if skill_lower not in seen:
                seen.add(skill_lower)
                unique.append(skill)

        merged[category] = unique

    return merged

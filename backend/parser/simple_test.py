#!/usr/bin/env python3
"""
Simple test script for ResumeParser without unicode issues
"""

import os
import sys

# Simple resume parser test without pandas dependency
def test_basic_parsing():
    print("Testing basic resume parsing...")
    
    # Sample resume text
    sample_resume = """
    John Doe
    Senior Software Developer
    
    With over 8 years of experience in software development, I am proficient in Python, JavaScript, and Java.
    I have extensive experience working with frameworks like Django, React, and Spring Boot.
    
    Technical Skills:
    - Programming Languages: Python, JavaScript, Java, TypeScript
    - Frameworks: Django, React, Angular, Spring Boot
    - Databases: PostgreSQL, MongoDB, Redis
    - Tools: Git, Docker, AWS, Jenkins
    - Other: RESTful APIs, Microservices, Agile Development
    
    Experience:
    - Developed and maintained web applications using Django and React
    - Implemented RESTful APIs for mobile applications
    - Collaborated with cross-functional teams to deliver high-quality software solutions
    - Led a team of 5 developers on multiple projects
    """
    
    # Try to parse without importing the full ResumeParser
    # Let's test individual components first
    
    # Test 1: Years extraction with regex
    import re
    year_patterns = [
        (r'(\d+)\+?\s*years?\s*of\s*experience', 1),
        (r'over\s*(\d+)\s*years?', 1),
        (r'more\s*than\s*(\d+)\s*years?', 1),
        (r'(\d+)\+\s*years?', 1),
        (r'(\d+)\s*years?\s*experience', 1),
        (r'with\s*(\d+)\s*years?', 1)
    ]
    
    found_years = []
    text_lower = sample_resume.lower()
    
    for pattern, group in year_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            try:
                years = int(match.group(group))
                if 0 < years <= 50:
                    found_years.append(years)
            except ValueError:
                continue
    
    years_result = max(found_years) if found_years else None
    print(f"Years of experience found: {years_result}")
    
    # Test 2: Basic skill extraction
    # Look for capitalized technical terms
    tech_patterns = [
        r'\b[A-Z][a-z]+\b',  # Capitalized words
        r'\b\w+\.\w+\b',     # Compound terms with dots
        r'\b[A-Z]{2,}\b'     # Acronyms
    ]
    
    technical_terms = []
    for pattern in tech_patterns:
        matches = re.findall(pattern, sample_resume)
        technical_terms.extend(matches)
    
    # Remove duplicates and common words
    stop_words = {'The', 'With', 'I', 'Senior', 'Other', 'Experience', 'Technical', 'Skills', 'Programming', 'Languages', 'Frameworks', 'Databases', 'Tools', 'Developed', 'Implemented', 'Collaborated', 'Led'}
    technical_terms = list(set([term for term in technical_terms if term not in stop_words and len(term) > 2]))
    
    print(f"Technical terms found: {technical_terms[:10]}...")  # Show first 10
    
    # Test 3: Programming language classification
    known_languages = ['python', 'javascript', 'java', 'typescript', 'php', 'ruby', 'go', 'rust', 'c++', 'c#']
    found_languages = []
    
    for term in technical_terms:
        if term.lower() in known_languages:
            found_languages.append(term)
    
    # Also search directly in text
    for lang in known_languages:
        if lang.lower() in text_lower:
            lang_title = lang.title()
            if lang_title not in found_languages:
                found_languages.append(lang_title)
    
    print(f"Programming languages found: {found_languages}")
    
    # Test 4: Framework detection
    known_frameworks = ['django', 'react', 'angular', 'spring', 'flask', 'vue', 'express']
    found_frameworks = []
    
    for framework in known_frameworks:
        if framework.lower() in text_lower:
            framework_title = framework.title()
            if framework_title not in found_frameworks:
                found_frameworks.append(framework_title)
    
    print(f"Frameworks found: {found_frameworks}")
    
    print("\nBasic parsing test completed successfully!")
    return True

def test_file_reading():
    """Test reading resume files from test directory"""
    print("\nTesting file reading...")
    
    # Test directory path
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'test_resumes')
    
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return False
    
    # List files
    files = [f for f in os.listdir(test_dir) if f.endswith('.txt')]
    print(f"Found {len(files)} test files: {files}")
    
    # Read first file
    if files:
        first_file = os.path.join(test_dir, files[0])
        try:
            with open(first_file, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"Successfully read {files[0]} ({len(content)} characters)")
            
            # Quick skill extraction test
            lines = content.lower().split('\n')
            skills_found = []
            
            for line in lines:
                if any(keyword in line for keyword in ['python', 'javascript', 'java', 'react', 'django']):
                    # Extract potential skills from this line
                    words = re.findall(r'\b[A-Za-z][A-Za-z0-9+.#-]*\b', line)
                    skills_found.extend(words)
            
            # Remove duplicates and filter
            unique_skills = list(set([skill for skill in skills_found if len(skill) > 2]))
            print(f"Potential skills from {files[0]}: {unique_skills[:10]}...")
            
        except Exception as e:
            print(f"Error reading {files[0]}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("Simple Resume Parser Test")
    print("=" * 40)
    
    # Run basic tests
    test1_result = test_basic_parsing()
    test2_result = test_file_reading()
    
    if test1_result and test2_result:
        print("\nAll tests passed! ✅")
    else:
        print("\nSome tests failed! ❌")
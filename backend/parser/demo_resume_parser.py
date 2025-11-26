#!/usr/bin/env python3
"""
Demo Resume Parser - Core functionality without pandas dependency
Demonstrates the key features of resume parsing and skill categorization
"""

import sqlite3
import re
import os
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter

class DemoResumeParser:
    """
    Simplified Resume parser that extracts and categorizes skills from individual resume files.
    Similar structure to csv_parser.py but without pandas dependency.
    """
    
    def __init__(self):
        print("🚀 Initializing Demo Resume Parser")
        
        # Basic stop words
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                              'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'])
        
        # Skill classification patterns
        self.programming_languages = [
            'python', 'javascript', 'java', 'typescript', 'php', 'ruby', 'go', 'rust',
            'c++', 'c#', 'swift', 'kotlin', 'scala', 'html', 'css', 'sql', 'r'
        ]
        
        self.frameworks_libraries = [
            'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'node',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'jquery', 'bootstrap'
        ]
        
        self.databases = [
            'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle', 'cassandra', 
            'elasticsearch', 'dynamodb', 'snowflake', 'bigquery'
        ]
        
        self.tools_software = [
            'git', 'docker', 'kubernetes', 'jenkins', 'aws', 'azure', 'gcp', 'terraform',
            'ansible', 'jira', 'confluence', 'slack', 'figma', 'photoshop', 'tableau',
            'excel', 'powerbi', 'linux', 'windows', 'macos'
        ]

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"❌ Error reading file {file_path}: {e}")
            return ""

    def extract_years_experience(self, text: str) -> Optional[int]:
        """Extract years of experience using regex patterns"""
        if not text:
            return None
            
        year_patterns = [
            (r'(\d+)\+?\s*years?\s*of\s*experience', 1),
            (r'over\s*(\d+)\s*years?', 1),
            (r'more\s*than\s*(\d+)\s*years?', 1),
            (r'(\d+)\+\s*years?', 1),
            (r'(\d+)\s*years?\s*experience', 1),
            (r'experience\s*:\s*(\d+)\s*years?', 1),
            (r'with\s*(\d+)\s*years?', 1)
        ]
        
        found_years = []
        text_lower = text.lower()
        
        for pattern, group in year_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                try:
                    years = int(match.group(group))
                    if 0 < years <= 50:  # Reasonable range
                        # Check context for experience indicators
                        context_start = max(0, match.start() - 20)
                        context_end = min(len(text_lower), match.end() + 20)
                        context = text_lower[context_start:context_end]
                        
                        experience_indicators = ['experience', 'work', 'career', 'professional', 'industry']
                        if any(indicator in context for indicator in experience_indicators):
                            found_years.append(years)
                except ValueError:
                    continue
        
        return max(found_years) if found_years else None

    def extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms using basic patterns"""
        patterns = [
            r'\b[A-Z][a-z]+\b',  # Capitalized words
            r'\b\w+\.\w+\b',     # Compound terms with dots
            r'\b\w+-\w+\b',      # Compound terms with hyphens
            r'\b\w+\+\+?\b',     # Terms with plus signs
            r'\b\w+#\b',         # Terms with hash
            r'\b[A-Z]{2,}\b'     # Acronyms
        ]
        
        technical_terms = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            technical_terms.extend(matches)
        
        # Filter out common words
        common_words = {'The', 'With', 'Senior', 'Experience', 'Technical', 'Skills', 'Programming', 
                       'Languages', 'Frameworks', 'Databases', 'Tools', 'Developed', 'Implemented', 
                       'Collaborated', 'Led', 'Using', 'Working', 'Project', 'Team', 'Software',
                       'Development', 'Application', 'System', 'Management', 'Analysis', 'Design'}
        
        return list(set([term for term in technical_terms if term not in common_words and len(term) > 2]))

    def classify_skills(self, terms: List[str], text: str) -> Dict[str, List[str]]:
        """Classify extracted terms into skill categories"""
        classified = {
            'programming_languages': [],
            'frameworks_libraries': [],
            'tools_software': [],
            'databases': [],
            'soft_skills': [],
            'other_skills': []
        }
        
        text_lower = text.lower()
        
        # Classify by matching against known lists
        for term in terms:
            term_lower = term.lower()
            classified_flag = False
            
            # Check programming languages
            if any(lang in term_lower for lang in self.programming_languages):
                classified['programming_languages'].append(term)
                classified_flag = True
            
            # Check frameworks/libraries
            elif any(fw in term_lower for fw in self.frameworks_libraries):
                classified['frameworks_libraries'].append(term)
                classified_flag = True
            
            # Check databases
            elif any(db in term_lower for db in self.databases):
                classified['databases'].append(term)
                classified_flag = True
            
            # Check tools/software
            elif any(tool in term_lower for tool in self.tools_software):
                classified['tools_software'].append(term)
                classified_flag = True
            
            # If not classified yet, put in other_skills
            if not classified_flag:
                classified['other_skills'].append(term)
        
        # Also search directly in text for known skills
        for lang in self.programming_languages:
            if lang in text_lower and lang.title() not in classified['programming_languages']:
                classified['programming_languages'].append(lang.title())
        
        for fw in self.frameworks_libraries:
            if fw in text_lower and fw.title() not in classified['frameworks_libraries']:
                classified['frameworks_libraries'].append(fw.title())
        
        for db in self.databases:
            if db in text_lower and db.title() not in classified['databases']:
                classified['databases'].append(db.title())
        
        for tool in self.tools_software:
            if tool in text_lower and tool.title() not in classified['tools_software']:
                classified['tools_software'].append(tool.title())
        
        # Extract soft skills
        classified['soft_skills'] = self.extract_soft_skills(text)
        
        # Remove duplicates
        for category in classified:
            classified[category] = list(set(classified[category]))
        
        return classified

    def extract_soft_skills(self, text: str) -> List[str]:
        """Extract soft skills using pattern matching"""
        soft_skills_found = []
        text_lower = text.lower()
        
        soft_skill_patterns = {
            'Problem Solving': [r'problem[- ]solving', r'analytical', r'critical\s+thinking', r'troubleshooting'],
            'Team Collaboration': [r'team\s+player', r'collaborative?', r'teamwork', r'cross[- ]functional'],
            'Communication': [r'communication', r'interpersonal', r'presentation'],
            'Leadership': [r'leadership', r'leading\s+teams', r'mentoring', r'coaching'],
            'Project Management': [r'project\s+management', r'time\s+management', r'deadline'],
            'Adaptability': [r'adaptable', r'flexible', r'fast[- ]paced'],
            'Initiative': [r'proactive', r'self[- ]motivated', r'initiative']
        }
        
        for skill_name, patterns in soft_skill_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    soft_skills_found.append(skill_name)
                    break
        
        return soft_skills_found

    def extract_experience_descriptions(self, text: str) -> List[str]:
        """Extract experience descriptions"""
        sentences = re.split(r'[.!?]+', text)
        experience_sentences = []
        
        action_verbs = [
            'developed', 'created', 'built', 'designed', 'implemented', 'managed',
            'led', 'coordinated', 'supervised', 'achieved', 'delivered', 'optimized'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:
                continue
                
            sentence_lower = sentence.lower()
            
            if any(verb in sentence_lower for verb in action_verbs):
                if len(sentence) <= 200:
                    experience_sentences.append(sentence)
        
        return experience_sentences[:5]

    def parse_resume_text(self, resume_text: str, resume_id: int, category: str = "Unknown") -> Dict:
        """Parse a single resume text"""
        # Extract years of experience
        years_exp = self.extract_years_experience(resume_text)
        
        # Extract technical terms
        technical_terms = self.extract_technical_terms(resume_text)
        
        # Classify skills
        classified_skills = self.classify_skills(technical_terms, resume_text)
        
        # Extract experience descriptions
        experience_desc = self.extract_experience_descriptions(resume_text)
        
        return {
            'id': resume_id,
            'category': category,
            'years_experience': years_exp,
            'programming_languages': classified_skills['programming_languages'],
            'frameworks_libraries': classified_skills['frameworks_libraries'],
            'tools_software': classified_skills['tools_software'],
            'databases': classified_skills['databases'],
            'soft_skills': classified_skills['soft_skills'],
            'other_skills': classified_skills['other_skills'],
            'experience_descriptions': experience_desc,
            'raw_extracted_terms': technical_terms
        }

    def parse_resume_file(self, file_path: str, resume_id: int, category: str = "Unknown") -> Dict:
        """Parse a resume from file"""
        print(f"📄 Processing resume file: {os.path.basename(file_path)}")
        
        resume_text = self.extract_text_from_file(file_path)
        if not resume_text.strip():
            print(f"❌ No text extracted from {file_path}")
            return None
        
        return self.parse_resume_text(resume_text, resume_id, category)

    def create_sqlite_database(self, parsed_resumes: List[Dict], db_file: str = 'demo_resume_results.db'):
        """Create SQLite database with parsing results"""
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Create tables
        table_schemas = {
            'persons': 'id INTEGER PRIMARY KEY, category TEXT, years_experience INTEGER',
            'programming_languages': 'id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, programming_languages TEXT',
            'frameworks_libraries': 'id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, frameworks_libraries TEXT',
            'tools_software': 'id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, tools_software TEXT',
            'databases': 'id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, databases TEXT',
            'soft_skills': 'id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, soft_skills TEXT',
            'other_skills': 'id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, other_skills TEXT',
            'experience_descriptions': 'id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, description_order INTEGER, description TEXT'
        }
        
        # Drop and create tables
        for table_name, schema in table_schemas.items():
            cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
            cursor.execute(f'CREATE TABLE {table_name} ({schema})')
        
        # Insert data
        for resume in parsed_resumes:
            if resume is None:
                continue
                
            # Insert person
            cursor.execute(
                'INSERT INTO persons VALUES (?, ?, ?)',
                (resume['id'], resume['category'], resume['years_experience'])
            )
            
            # Insert skills
            for skill_type in ['programming_languages', 'frameworks_libraries', 'tools_software', 
                             'databases', 'soft_skills', 'other_skills']:
                for skill in resume[skill_type]:
                    if skill:
                        cursor.execute(
                            f'INSERT INTO {skill_type} (person_id, {skill_type}) VALUES (?, ?)',
                            (resume['id'], skill)
                        )
            
            # Insert experience descriptions
            for idx, desc in enumerate(resume['experience_descriptions']):
                cursor.execute(
                    'INSERT INTO experience_descriptions (person_id, description_order, description) VALUES (?, ?, ?)',
                    (resume['id'], idx + 1, desc)
                )
        
        conn.commit()
        conn.close()
        print(f"💾 Database created successfully: {db_file}")

    def generate_analytics(self, parsed_resumes: List[Dict]) -> Dict:
        """Generate analytics on parsed resumes"""
        analytics = {}
        
        # Collect all skills
        all_programming_languages = []
        all_frameworks = []
        all_tools = []
        all_databases = []
        all_soft_skills = []
        
        for resume in parsed_resumes:
            if resume is None:
                continue
            all_programming_languages.extend(resume['programming_languages'])
            all_frameworks.extend(resume['frameworks_libraries'])
            all_tools.extend(resume['tools_software'])
            all_databases.extend(resume['databases'])
            all_soft_skills.extend(resume['soft_skills'])
        
        # Generate counts
        for skill_type, skills in [
            ('programming_languages', all_programming_languages),
            ('frameworks_libraries', all_frameworks),
            ('tools_software', all_tools),
            ('databases', all_databases)
        ]:
            if skills:
                skill_counts = Counter(skills)
                analytics[skill_type] = {
                    'total_mentions': len(skills),
                    'unique_skills': len(skill_counts),
                    'top_10': skill_counts.most_common(10)
                }
        
        return analytics

def main():
    """Demo the resume parser functionality"""
    print("📋 Demo Resume Parser")
    print("=" * 50)
    
    parser = DemoResumeParser()
    
    # Test with sample resume
    sample_resume = """
    Sarah Martinez
    Senior Software Developer
    
    With over 6 years of experience in software development, I am proficient in Python, JavaScript, and Java.
    I have extensive experience working with frameworks like Django, React, and Spring Boot.
    
    Technical Skills:
    - Programming Languages: Python, JavaScript, Java, TypeScript
    - Frameworks: Django, React, Angular, Spring Boot
    - Databases: PostgreSQL, MongoDB, Redis
    - Tools: Git, Docker, AWS, Jenkins
    
    Experience:
    - Developed web applications using Django and React
    - Implemented RESTful APIs for mobile applications
    - Led a team of 4 developers on multiple projects
    - Collaborated with cross-functional teams to deliver solutions
    """
    
    # Parse sample resume
    print("\n🔍 Parsing Sample Resume:")
    parsed = parser.parse_resume_text(sample_resume, 1, "Software Developer")
    
    print(f"   Years Experience: {parsed['years_experience']}")
    print(f"   Programming Languages: {parsed['programming_languages']}")
    print(f"   Frameworks: {parsed['frameworks_libraries']}")
    print(f"   Tools: {parsed['tools_software']}")
    print(f"   Databases: {parsed['databases']}")
    print(f"   Soft Skills: {parsed['soft_skills']}")
    print(f"   Experience Descriptions: {len(parsed['experience_descriptions'])} found")
    
    # Test with actual resume files
    test_resumes_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'test_resumes')
    
    if os.path.exists(test_resumes_dir):
        print(f"\n📁 Testing with resume files from: {test_resumes_dir}")
        
        resume_files = [f for f in os.listdir(test_resumes_dir) if f.endswith('.txt')]
        print(f"Found {len(resume_files)} resume files")
        
        all_parsed = [parsed]  # Include sample resume
        
        for i, filename in enumerate(resume_files[:3], 2):  # Test first 3 files
            file_path = os.path.join(test_resumes_dir, filename)
            category = filename.replace('.txt', '').replace('_', ' ').title()
            
            parsed_resume = parser.parse_resume_file(file_path, i, category)
            if parsed_resume:
                all_parsed.append(parsed_resume)
                print(f"   ✅ {filename}: {len(parsed_resume['raw_extracted_terms'])} terms extracted")
        
        # Generate analytics
        print(f"\n📊 Analytics Summary:")
        analytics = parser.generate_analytics(all_parsed)
        
        for skill_type, stats in analytics.items():
            print(f"   {skill_type.replace('_', ' ').title()}:")
            print(f"     Total: {stats['total_mentions']}, Unique: {stats['unique_skills']}")
            if stats['top_10']:
                top_3 = [f"{skill}({count})" for skill, count in stats['top_10'][:3]]
                print(f"     Top 3: {', '.join(top_3)}")
        
        # Create database
        parser.create_sqlite_database(all_parsed)
        
        print(f"\n✅ Demo completed successfully!")
        print(f"   Processed {len(all_parsed)} resumes")
        print(f"   Database saved as: demo_resume_results.db")
    
    else:
        print(f"\n⚠️  Test resumes directory not found: {test_resumes_dir}")
        print("Creating database with sample resume only...")
        parser.create_sqlite_database([parsed])

if __name__ == "__main__":
    main()
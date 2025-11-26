# simple_parser.py - Works with your current requirements (no spaCy)

import pandas as pd
import sqlite3
import re
from typing import Dict, List, Optional, Tuple
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

class SimpleResumeParser:
    """Resume parser that works without spaCy - uses regex and keyword matching"""
    
    def __init__(self):
        print("🚀 Initializing Simple Resume Parser (no spaCy dependency)")
        
        # Comprehensive skill dictionaries
        self.programming_languages = {
            'python', 'javascript', 'java', 'typescript', 'php', 'ruby', 'go', 'rust',
            'c++', 'c#', 'csharp', 'swift', 'kotlin', 'scala', 'html', 'css', 'sql',
            'r', 'matlab', 'perl', 'shell', 'bash', 'powershell', 'dart', 'lua',
            'objective-c', 'assembly', 'fortran', 'cobol', 'vb.net', 'f#', 'clojure',
            'haskell', 'erlang', 'elixir', 'groovy', 'coffeescript'
        }
        
        self.frameworks_libraries = {
            'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 
            'nodejs', 'node.js', 'laravel', 'rails', 'ruby on rails', 'asp.net',
            'fastapi', 'nextjs', 'next.js', 'nuxt', 'svelte', 'ember', 'backbone',
            'jquery', 'bootstrap', 'tailwind', 'material-ui', 'ant design',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
            'matplotlib', 'seaborn', 'plotly', 'streamlit', 'dash'
        }
        
        self.tools_software = {
            'git', 'github', 'gitlab', 'bitbucket', 'docker', 'kubernetes', 'k8s',
            'jenkins', 'aws', 'azure', 'gcp', 'google cloud', 'linux', 'windows',
            'macos', 'jira', 'confluence', 'slack', 'teams', 'tableau', 'powerbi',
            'excel', 'photoshop', 'figma', 'sketch', 'vs code', 'visual studio',
            'intellij', 'eclipse', 'pycharm', 'sublime', 'atom', 'vim', 'emacs',
            'postman', 'insomnia', 'swagger', 'terraform', 'ansible', 'vagrant',
            'nginx', 'apache', 'tomcat', 'iis'
        }
        
        self.databases = {
            'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'sqlite',
            'oracle', 'mssql', 'sql server', 'cassandra', 'elasticsearch',
            'dynamodb', 'firestore', 'couchdb', 'neo4j', 'influxdb',
            'clickhouse', 'snowflake', 'bigquery', 'redshift'
        }
        
        self.soft_skills = {
            'leadership', 'communication', 'teamwork', 'collaboration', 'management',
            'analytical', 'creative', 'problem solving', 'critical thinking',
            'project management', 'time management', 'mentoring', 'coaching',
            'presentation', 'negotiation', 'adaptability', 'innovation'
        }

    def extract_years_experience(self, text: str) -> Optional[int]:
        """Extract years of experience using comprehensive regex patterns"""
        patterns = [
            r'(\d+)\+?\s*years?\s*of\s*experience',
            r'over\s*(\d+)\s*years?',
            r'more\s*than\s*(\d+)\s*years?',
            r'(\d+)\+\s*years?',
            r'(\d+)\s*years?\s*experience',
            r'(\d+)\s*year\s*experience',
            r'experience\s*of\s*(\d+)\s*years?',
            r'(\d+)\s*years?\s*in\s*\w+',
            r'with\s*(\d+)\s*years?',
            r'having\s*(\d+)\s*years?'
        ]
        
        text_lower = text.lower()
        found_years = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    years = int(match)
                    if 0 < years <= 50:  # Reasonable range
                        found_years.append(years)
                except ValueError:
                    continue
        
        return max(found_years) if found_years else None

    def extract_comprehensive_skills(self, text: str) -> Dict[str, List[str]]:
        """Comprehensive skill extraction using multiple techniques"""
        text_lower = text.lower()
        
        # Method 1: Direct keyword matching
        found_skills = {
            'programming_languages': [],
            'frameworks_libraries': [],
            'tools_software': [],
            'databases': [],
            'soft_skills': [],
            'other_skills': []
        }
        
        # Extract various text patterns
        words = re.findall(r'\b\w+\b', text_lower)
        compound_terms = re.findall(r'\b[\w\-\.+#]+\b', text_lower)
        phrases = re.findall(r'\b\w+\s+\w+\b', text_lower)
        
        all_terms = set(words + compound_terms + phrases)
        
        # Classify terms
        for term in all_terms:
            term_clean = self.clean_term(term)
            
            if self.is_programming_language(term, term_clean):
                found_skills['programming_languages'].append(term_clean)
            elif self.is_framework_library(term, term_clean):
                found_skills['frameworks_libraries'].append(term_clean)
            elif self.is_database(term, term_clean):
                found_skills['databases'].append(term_clean)
            elif self.is_tool_software(term, term_clean):
                found_skills['tools_software'].append(term_clean)
            elif self.is_soft_skill(term, term_clean):
                found_skills['soft_skills'].append(term_clean)
        
        # Method 2: Context-based extraction
        self.extract_context_skills(text_lower, found_skills)
        
        # Clean and deduplicate
        for category in found_skills:
            found_skills[category] = sorted(list(set(found_skills[category])))
            found_skills[category] = [skill for skill in found_skills[category] if len(skill) > 1]
        
        return found_skills

    def clean_term(self, term: str) -> str:
        """Clean and normalize terms"""
        # Handle special cases
        term = term.lower().strip()
        
        # Common replacements
        replacements = {
            'c++': 'c++', 'c#': 'c#', 'csharp': 'c#',
            'node.js': 'nodejs', 'vue.js': 'vue',
            'react.js': 'react', 'angular.js': 'angular'
        }
        
        return replacements.get(term, term)

    def is_programming_language(self, original: str, cleaned: str) -> bool:
        """Check if term is a programming language"""
        return (cleaned in self.programming_languages or 
                original in self.programming_languages or
                any(lang in original for lang in ['c++', 'c#', '.net']))

    def is_framework_library(self, original: str, cleaned: str) -> bool:
        """Check if term is a framework or library"""
        return (cleaned in self.frameworks_libraries or 
                original in self.frameworks_libraries or
                original.endswith('.js') or
                any(fw in original for fw in ['react', 'angular', 'vue', 'django']))

    def is_database(self, original: str, cleaned: str) -> bool:
        """Check if term is a database"""
        return (cleaned in self.databases or 
                original in self.databases or
                'sql' in original)

    def is_tool_software(self, original: str, cleaned: str) -> bool:
        """Check if term is a tool or software"""
        return (cleaned in self.tools_software or 
                original in self.tools_software)

    def is_soft_skill(self, original: str, cleaned: str) -> bool:
        """Check if term is a soft skill"""
        return (cleaned in self.soft_skills or 
                original in self.soft_skills or
                any(skill in original for skill in ['leadership', 'management', 'communication']))

    def extract_context_skills(self, text: str, found_skills: Dict):
        """Extract skills using context patterns"""
        # Patterns for skill extraction
        skill_patterns = [
            r'skilled in ([^.]+)',
            r'proficient in ([^.]+)',
            r'experience with ([^.]+)',
            r'expertise in ([^.]+)',
            r'knowledge of ([^.]+)',
            r'familiar with ([^.]+)',
            r'worked with ([^.]+)',
            r'using ([^.]+)',
            r'technologies: ([^.]+)',
            r'tools: ([^.]+)',
            r'languages: ([^.]+)'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Split by common separators
                skills = re.split(r'[,;and&]+', match)
                for skill in skills:
                    skill = skill.strip()
                    if len(skill) > 1:
                        # Try to classify this extracted skill
                        if any(lang in skill for lang in self.programming_languages):
                            found_skills['programming_languages'].append(skill)
                        elif any(fw in skill for fw in self.frameworks_libraries):
                            found_skills['frameworks_libraries'].append(skill)
                        # Add more classification logic as needed

    def extract_experience_descriptions(self, text: str) -> List[str]:
        """Extract key experience descriptions"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        experience_keywords = [
            'responsible for', 'developed', 'created', 'built', 'designed',
            'implemented', 'managed', 'led', 'worked on', 'experience in',
            'achieved', 'delivered', 'coordinated', 'collaborated', 'supervised'
        ]
        
        experience_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (any(keyword in sentence.lower() for keyword in experience_keywords) and
                20 <= len(sentence) <= 200):
                experience_sentences.append(sentence)
        
        return experience_sentences[:5]

    def parse_single_resume(self, resume_text: str, category: str, resume_id: int) -> Dict:
        """Parse a single resume"""
        years_exp = self.extract_years_experience(resume_text)
        skills = self.extract_comprehensive_skills(resume_text)
        experience_desc = self.extract_experience_descriptions(resume_text)
        
        return {
            'id': resume_id,
            'category': category,
            'years_experience': years_exp,
            'programming_languages': skills['programming_languages'],
            'frameworks_libraries': skills['frameworks_libraries'],
            'tools_software': skills['tools_software'],
            'databases': skills['databases'],
            'soft_skills': skills['soft_skills'],
            'other_skills': skills['other_skills'],
            'experience_descriptions': experience_desc,
            'raw_extracted_terms': (skills['programming_languages'] + 
                                  skills['frameworks_libraries'] + 
                                  skills['tools_software'] + 
                                  skills['databases'])
        }

    def load_and_parse_dataset(self, csv_file: str) -> Tuple[List[Dict], Dict]:
        """Load and parse the entire dataset"""
        print(f"📁 Loading dataset from {csv_file}...")
        
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['Category', 'Resume'])
        df['Category'] = df['Category'].str.strip()
        df['Resume'] = df['Resume'].str.strip()
        
        print(f"🔄 Processing {len(df)} resumes...")
        
        parsed_resumes = []
        for idx, row in df.iterrows():
            if idx % 50 == 0:  # Progress indicator
                print(f"   Progress: {idx + 1}/{len(df)} resumes processed...")
            
            parsed = self.parse_single_resume(row['Resume'], row['Category'], idx + 1)
            parsed_resumes.append(parsed)
        
        print(f"✅ Completed parsing {len(parsed_resumes)} resumes!")
        
        tables = self.create_normalized_tables(parsed_resumes)
        return parsed_resumes, tables

    def create_normalized_tables(self, parsed_resumes: List[Dict]) -> Dict:
        """Create normalized table structures"""
        tables = {
            'persons': [],
            'programming_languages': [],
            'frameworks_libraries': [],
            'tools_software': [],
            'databases': [],
            'soft_skills': [],
            'other_skills': [],
            'experience_descriptions': [],
            'extracted_terms_summary': []
        }
        
        for resume in parsed_resumes:
            tables['persons'].append({
                'id': resume['id'],
                'category': resume['category'],
                'years_experience': resume['years_experience']
            })
            
            # Add skills to respective tables
            for skill_type in ['programming_languages', 'frameworks_libraries', 
                             'tools_software', 'databases', 'soft_skills', 'other_skills']:
                for skill in resume[skill_type]:
                    if skill:  # Only add non-empty skills
                        tables[skill_type].append({
                            'person_id': resume['id'],
                            skill_type: skill
                        })
            
            # Add experience descriptions
            for idx, desc in enumerate(resume['experience_descriptions']):
                tables['experience_descriptions'].append({
                    'person_id': resume['id'],
                    'description_order': idx + 1,
                    'description': desc
                })
            
            # Add summary
            tables['extracted_terms_summary'].append({
                'person_id': resume['id'],
                'total_terms_extracted': len(resume['raw_extracted_terms']),
                'all_terms': ', '.join(resume['raw_extracted_terms'])
            })
        
        return tables

    def create_sqlite_database(self, tables: Dict, db_file: str = 'simple_resume_data.db'):
        """Create SQLite database"""
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
            'experience_descriptions': 'id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, description_order INTEGER, description TEXT',
            'extracted_terms_summary': 'id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, total_terms_extracted INTEGER, all_terms TEXT'
        }
        
        for table_name, schema in table_schemas.items():
            cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
            cursor.execute(f'CREATE TABLE {table_name} ({schema})')
        
        # Insert data
        for table_name, data in tables.items():
            if not data:
                continue
            
            if table_name == 'persons':
                for record in data:
                    cursor.execute(
                        'INSERT INTO persons VALUES (?, ?, ?)',
                        (record['id'], record['category'], record['years_experience'])
                    )
            else:
                for record in data:
                    if table_name in ['programming_languages', 'frameworks_libraries', 'tools_software', 
                                    'databases', 'soft_skills', 'other_skills']:
                        cursor.execute(
                            f'INSERT INTO {table_name} (person_id, {table_name}) VALUES (?, ?)',
                            (record['person_id'], record[table_name])
                        )
                    elif table_name == 'experience_descriptions':
                        cursor.execute(
                            'INSERT INTO experience_descriptions (person_id, description_order, description) VALUES (?, ?, ?)',
                            (record['person_id'], record['description_order'], record['description'])
                        )
                    elif table_name == 'extracted_terms_summary':
                        cursor.execute(
                            'INSERT INTO extracted_terms_summary (person_id, total_terms_extracted, all_terms) VALUES (?, ?, ?)',
                            (record['person_id'], record['total_terms_extracted'], record['all_terms'])
                        )
        
        conn.commit()
        conn.close()
        print(f"💾 Database created successfully: {db_file}")

    def generate_analytics(self, tables: Dict) -> Dict:
        """Generate skill analytics"""
        analytics = {}
        
        for skill_type in ['programming_languages', 'frameworks_libraries', 'tools_software', 'databases']:
            if skill_type in tables and tables[skill_type]:
                skills = [item[skill_type] for item in tables[skill_type]]
                skill_counts = Counter(skills)
                analytics[skill_type] = {
                    'total_mentions': len(skills),
                    'unique_skills': len(skill_counts),
                    'top_10': skill_counts.most_common(10)
                }
        
        return analytics

def main():
    """Main function"""
    parser = SimpleResumeParser()
    
    try:
        # Parse the dataset
        parsed_resumes, tables = parser.load_and_parse_dataset('csdataset.csv')
        
        # Create database
        parser.create_sqlite_database(tables)
        
        # Generate analytics
        analytics = parser.generate_analytics(tables)
        
        # Print results
        print("\n" + "="*60)
        print("🎉 SIMPLE RESUME PARSING COMPLETE")
        print("="*60)
        
        print(f"\n📊 Summary:")
        print(f"   Processed: {len(parsed_resumes)} resumes")
        print(f"   Database: simple_resume_data.db")
        
        print(f"\n🏆 Top Skills Found:")
        for skill_type, stats in analytics.items():
            print(f"\n   {skill_type.replace('_', ' ').title()}:")
            print(f"     Total mentions: {stats['total_mentions']}")
            print(f"     Unique skills: {stats['unique_skills']}")
            if stats['top_10']:
                top_5 = [f"{skill}({count})" for skill, count in stats['top_10'][:5]]
                print(f"     Top 5: {', '.join(top_5)}")
        
        # Example query
        print(f"\n🔍 Sample Query Results:")
        conn = sqlite3.connect('simple_resume_data.db')
        
        # Find Python developers
        query = """
        SELECT p.category, COUNT(*) as count
        FROM persons p
        JOIN programming_languages pl ON p.id = pl.person_id
        WHERE pl.programming_languages LIKE '%python%'
        GROUP BY p.category
        ORDER BY count DESC
        """
        
        result = pd.read_sql_query(query, conn)
        print(f"   Python developers by category:")
        for _, row in result.iterrows():
            print(f"     {row['category']}: {row['count']}")
        
        conn.close()
        print(f"\n✅ Parsing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
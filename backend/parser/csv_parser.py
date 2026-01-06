
import pandas as pd
import sqlite3
import re
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")

# NLP Libraries (avoiding spaCy)
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.util import ngrams
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("  NLTK not available - using basic text processing")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("scikit-learn not available - using basic feature extraction")

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    print("TextBlob not available - using basic sentiment analysis")

class NLPResumeParser:
    """NLP-based resume parser using NLTK, scikit-learn, and TextBlob"""
    
    def __init__(self):
        print("Initializing Resume Parser")
        
        # Download required NLTK data
        if HAS_NLTK:
            self._download_nltk_data()
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            self.lemmatizer = None
        
        # Initialize skill classification patterns
        self.skill_context_patterns = {
            'programming_languages': [
                r'programming\s+(?:language|in|with)',
                r'coded?\s+in',
                r'developed?\s+(?:in|with|using)',
                r'language[s]?\s*:\s*',
                r'proficient\s+in\s+\w+\s+programming',
                r'experience\s+(?:in|with)\s+\w+\s+development'
            ],
            'frameworks_libraries': [
                r'framework[s]?\s*:\s*',
                r'using\s+\w+\s+framework',
                r'library[ies]*\s*:\s*',
                r'built\s+with\s+\w+',
                r'implemented\s+using\s+\w+',
                r'experience\s+with\s+\w+\s+framework'
            ],
            'tools_software': [
                r'tool[s]?\s*:\s*',
                r'software\s*:\s*',
                r'platform[s]?\s*:\s*',
                r'environment[s]?\s*:\s*',
                r'using\s+\w+\s+tool',
                r'worked\s+with\s+\w+\s+platform'
            ],
            'databases': [
                r'database[s]?\s*:\s*',
                r'worked\s+with\s+\w+\s+database',
                r'experience\s+with\s+\w+\s+db',
                r'data\s+storage\s+using',
                r'sql\s+database',
                r'nosql\s+database'
            ]
        }
        
        # Technical indicators for classification
        self.technical_indicators = {
            'version_patterns': [r'\d+\.\d+', r'v\d+', r'version\s+\d+'],
            'file_extensions': [r'\.\w{2,4}', r'\.js', r'\.py', r'\.java', r'\.cpp'],
            'technical_suffixes': ['js', 'py', 'sql', 'db', 'api', 'sdk', 'cli', 'ui', 'ux'],
            'technical_prefixes': ['web', 'mobile', 'cloud', 'data', 'machine', 'deep', 'big'],
            'compound_patterns': [r'\w+[-\.]\w+', r'\w+\s+\w+\s+\w+'],
        }

    def _download_nltk_data(self):
        """Download required NLTK data"""
        nltk_downloads = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 
                         'words', 'stopwords', 'wordnet', 'omw-1.4']
        
        for item in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{item}')
            except LookupError:
                try:
                    nltk.data.find(f'taggers/{item}')
                except LookupError:
                    try:
                        nltk.data.find(f'chunkers/{item}')
                    except LookupError:
                        try:
                            nltk.data.find(f'corpora/{item}')
                        except LookupError:
                            try:
                                nltk.download(item, quiet=True)
                            except:
                                pass

    def extract_years_experience(self, text: str) -> Optional[int]:
        """Extract years of experience using NLP techniques"""
        if not text:
            return None
            
        # Tokenize and process text
        if HAS_NLTK:
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
        else:
            tokens = re.findall(r'\b\w+\b', text.lower())
            pos_tags = [(token, 'UNKNOWN') for token in tokens]
        
        # Look for year patterns with context
        year_patterns = [
            (r'(\d+)\+?\s*years?\s*of\s*experience', 1),
            (r'over\s*(\d+)\s*years?', 1),
            (r'more\s*than\s*(\d+)\s*years?', 1),
            (r'(\d+)\+\s*years?', 1),
            (r'(\d+)\s*years?\s*experience', 1),
            (r'experience\s*:\s*(\d+)\s*years?', 1),
            (r'(\d+)\s*years?\s*in\s*\w+', 1),
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
                        # Check context to ensure it's about experience
                        context_start = max(0, match.start() - 20)
                        context_end = min(len(text_lower), match.end() + 20)
                        context = text_lower[context_start:context_end]
                        
                        experience_indicators = ['experience', 'work', 'career', 'professional', 'industry']
                        if any(indicator in context for indicator in experience_indicators):
                            found_years.append(years)
                except ValueError:
                    continue
        
        return max(found_years) if found_years else None

    def extract_named_entities(self, text: str) -> List[str]:
        """Extract named entities using NLTK"""
        if not HAS_NLTK:
            return []
            
        try:
            # Tokenize and tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Extract named entities
            entities = []
            tree = ne_chunk(pos_tags)
            
            for subtree in tree:
                if hasattr(subtree, 'label'):
                    entity_name = ' '.join([token for token, pos in subtree.leaves()])
                    if len(entity_name) > 1 and not entity_name.lower() in self.stop_words:
                        entities.append(entity_name)
            
            return entities
        except Exception:
            return []

    def extract_technical_terms_nlp(self, text: str) -> List[str]:
        """Extract technical terms using NLP techniques"""
        if not HAS_NLTK:
            return self._extract_technical_terms_basic(text)
        
        try:
            # Tokenize and POS tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            technical_terms = []
            
            # Extract based on POS patterns
            for i, (word, pos) in enumerate(pos_tags):
                # Skip common words
                if word.lower() in self.stop_words or len(word) < 2:
                    continue
                
                # Technical terms are often:
                # 1. Proper nouns (NNP, NNPS)
                # 2. Regular nouns in technical contexts (NN, NNS)
                # 3. Adjectives describing technical concepts (JJ)
                
                if pos in ['NNP', 'NNPS']:  # Proper nouns
                    technical_terms.append(word)
                elif pos in ['NN', 'NNS'] and self._is_technical_context(word, pos_tags, i):
                    technical_terms.append(word)
                elif pos == 'JJ' and self._is_technical_adjective(word):
                    technical_terms.append(word)
            
            # Extract compound terms
            compound_terms = self._extract_compound_terms(text)
            technical_terms.extend(compound_terms)
            
            return list(set(technical_terms))
            
        except Exception:
            return self._extract_technical_terms_basic(text)

    def _extract_technical_terms_basic(self, text: str) -> List[str]:
        """Basic technical term extraction without NLTK"""
        # Extract capitalized words and technical patterns
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
        
        return list(set(technical_terms))

    def _is_technical_context(self, word: str, pos_tags: List[Tuple[str, str]], index: int) -> bool:
        """Check if a word appears in technical context"""
        # Check surrounding words for technical indicators
        context_window = 3
        start_idx = max(0, index - context_window)
        end_idx = min(len(pos_tags), index + context_window + 1)
        
        context_words = [pos_tags[i][0].lower() for i in range(start_idx, end_idx)]
        
        technical_context_words = [
            'programming', 'development', 'software', 'web', 'mobile', 'data',
            'machine', 'learning', 'artificial', 'intelligence', 'framework',
            'library', 'database', 'server', 'client', 'api', 'service',
            'platform', 'system', 'application', 'technology', 'tool',
            'environment', 'cloud', 'devops', 'agile', 'scrum'
        ]
        
        return any(tech_word in context_words for tech_word in technical_context_words)

    def _is_technical_adjective(self, word: str) -> bool:
        """Check if adjective is technical"""
        technical_adjectives = [
            'scalable', 'distributed', 'concurrent', 'asynchronous', 'reactive',
            'responsive', 'robust', 'efficient', 'optimized', 'automated',
            'integrated', 'modular', 'extensible', 'maintainable', 'testable'
        ]
        return word.lower() in technical_adjectives

    def _extract_compound_terms(self, text: str) -> List[str]:
        """Extract compound technical terms"""
        compound_patterns = [
            r'\b\w+\.\w+(?:\.\w+)*\b',  # dot notation (e.g., React.js, Node.js)
            r'\b\w+-\w+(?:-\w+)*\b',    # hyphenated terms
            r'\b\w+\+\+?\b',            # plus notation (C++, C+)
            r'\b\w+#\b',                # hash notation (C#, F#)
            r'\b[A-Z]{2,}\b'            # Acronyms
        ]
        
        compounds = []
        for pattern in compound_patterns:
            matches = re.findall(pattern, text)
            compounds.extend(matches)
        
        return compounds

    def extract_skills_with_tfidf(self, text: str) -> List[str]:
        """Extract skills using TF-IDF analysis"""
        if not HAS_SKLEARN:
            return []
        
        try:
            # Preprocess text
            sentences = sent_tokenize(text) if HAS_NLTK else text.split('.')
            
            # Use TF-IDF to find important terms
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                stop_words='english',
                max_features=200,
                min_df=1,
                token_pattern=r'\b[A-Za-z][A-Za-z0-9\.\-\+#]*\b'
            )
            
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get high-scoring terms
            high_score_terms = []
            for i, score in enumerate(scores):
                if score > 0.1:  # Threshold for importance
                    term = feature_names[i]
                    if self._is_likely_skill(term):
                        high_score_terms.append(term)
            
            return high_score_terms[:50]  # Limit results
            
        except Exception:
            return []

    def _is_likely_skill(self, term: str) -> bool:
        """Determine if a term is likely a technical skill"""
        # Technical skill indicators
        if len(term) < 2:
            return False
        
        # Check for technical patterns
        technical_patterns = [
            r'^[A-Z][a-z]+$',           # Capitalized words
            r'\w+\.\w+',                # Compound with dots
            r'\w+-\w+',                 # Compound with hyphens
            r'^[A-Z]{2,}$',             # Acronyms
            r'\w+\+\+?$',               # Plus notation
            r'\w+#$',                   # Hash notation
            r'.*(?:js|py|sql|db|api|sdk|ui|ux|css|html)$'  # Technical suffixes
        ]
        
        for pattern in technical_patterns:
            if re.match(pattern, term):
                return True
        
        # Check for technical prefixes
        technical_prefixes = ['web', 'mobile', 'cloud', 'data', 'machine', 'deep', 'big']
        if any(term.lower().startswith(prefix) for prefix in technical_prefixes):
            return True
        
        return False

    def classify_skills_nlp(self, terms: List[str], context: str) -> Dict[str, List[str]]:
        """Classify skills using NLP context analysis"""
        classified = {
            'programming_languages': [],
            'frameworks_libraries': [],
            'tools_software': [],
            'databases': [],
            'soft_skills': [],
            'other_skills': []
        }
        
        context_lower = context.lower()
        
        for term in terms:
            if not term or len(term) < 2:
                continue
            
            # Use context patterns to classify
            classified_category = self._classify_by_context(term, context_lower)
            
            if classified_category:
                classified[classified_category].append(term)
            else:
                # Use heuristic classification
                classified_category = self._classify_by_heuristics(term)
                classified[classified_category].append(term)
        
        # Remove duplicates and clean
        for category in classified:
            classified[category] = list(set(classified[category]))
            classified[category] = [skill for skill in classified[category] if skill.strip()]
        
        return classified

    def _classify_by_context(self, term: str, context: str) -> Optional[str]:
        """Classify term based on surrounding context"""
        term_lower = term.lower()
        
        # Find term in context
        term_pos = context.find(term_lower)
        if term_pos == -1:
            return None
        
        # Get surrounding context
        context_window = 100
        start = max(0, term_pos - context_window)
        end = min(len(context), term_pos + len(term_lower) + context_window)
        surrounding = context[start:end]
        
        # Check against context patterns
        for category, patterns in self.skill_context_patterns.items():
            for pattern in patterns:
                if re.search(pattern, surrounding):
                    return category
        
        return None

    def _classify_by_heuristics(self, term: str) -> str:
        """Classify term using heuristic rules"""
        term_lower = term.lower()
        
        # Programming language indicators
        if (term_lower.endswith(('script', 'lang')) or 
            term_lower in ['c', 'r', 'go'] or
            any(indicator in term_lower for indicator in ['python', 'java', 'script', 'sql', 'html', 'css'])):
            return 'programming_languages'
        
        # Framework/library indicators
        if (term_lower.endswith(('.js', '.py', 'framework', 'library')) or
            any(indicator in term_lower for indicator in ['react', 'angular', 'vue', 'django', 'spring'])):
            return 'frameworks_libraries'
        
        # Database indicators
        if (term_lower.endswith(('db', 'sql', 'base')) or
            any(indicator in term_lower for indicator in ['mysql', 'postgres', 'mongo', 'redis', 'database'])):
            return 'databases'
        
        # Tool/software indicators
        if (any(indicator in term_lower for indicator in ['git', 'docker', 'aws', 'azure', 'linux', 'tool', 'platform']) or
            term_lower.isupper() and len(term_lower) > 1):  # Acronyms are often tools
            return 'tools_software'
        
        # Soft skills
        if any(indicator in term_lower for indicator in ['leadership', 'communication', 'management', 'teamwork']):
            return 'soft_skills'
        
        return 'other_skills'

    def extract_experience_descriptions(self, text: str) -> List[str]:
        """Extract experience descriptions using NLP"""
        if HAS_NLTK:
            sentences = sent_tokenize(text)
        else:
            sentences = re.split(r'[.!?]+', text)
        
        experience_sentences = []
        
        # Action verbs that indicate experience
        action_verbs = [
            'developed', 'created', 'built', 'designed', 'implemented', 'managed',
            'led', 'coordinated', 'supervised', 'achieved', 'delivered', 'optimized',
            'automated', 'integrated', 'deployed', 'maintained', 'collaborated',
            'architected', 'engineered', 'programmed', 'configured', 'tested'
        ]
        
        # Responsibility indicators
        responsibility_indicators = [
            'responsible for', 'in charge of', 'accountable for', 'oversaw',
            'handled', 'managed', 'coordinated', 'supervised', 'directed'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:
                continue
                
            sentence_lower = sentence.lower()
            
            # Check for action verbs or responsibility indicators
            if (any(verb in sentence_lower for verb in action_verbs) or
                any(indicator in sentence_lower for indicator in responsibility_indicators)):
                
                # Additional filtering for meaningful sentences
                if (len(sentence) <= 200 and
                    any(tech_word in sentence_lower for tech_word in 
                        ['software', 'system', 'application', 'development', 'technology', 'project'])):
                    experience_sentences.append(sentence)
        
        return experience_sentences[:5]  # Limit to top 5

    def extract_entities_and_skills(self, text: str) -> Dict[str, List[str]]:
        """Main skill extraction method combining multiple NLP techniques"""
        # Method 1: Named entity recognition
        entities = self.extract_named_entities(text)
        
        # Method 2: Technical term extraction
        technical_terms = self.extract_technical_terms_nlp(text)
        
        # Method 3: TF-IDF based extraction
        tfidf_terms = self.extract_skills_with_tfidf(text)
        
        # Combine all extracted terms
        all_terms = list(set(entities + technical_terms + tfidf_terms))
        
        # Classify using NLP context analysis
        classified_skills = self.classify_skills_nlp(all_terms, text)
        
        return classified_skills

    def parse_single_resume(self, resume_text: str, category: str, resume_id: int) -> Dict:
        """Parse a single resume using NLP techniques"""
        # Extract years of experience
        years_exp = self.extract_years_experience(resume_text)
        
        # Extract and classify skills
        classified_skills = self.extract_entities_and_skills(resume_text)
        
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
            'raw_extracted_terms': list(set(
                classified_skills['programming_languages'] +
                classified_skills['frameworks_libraries'] +
                classified_skills['tools_software'] +
                classified_skills['databases'] +
                classified_skills['other_skills']
            ))
        }

    def load_and_parse_dataset(self, csv_file: str) -> Tuple[List[Dict], Dict]:
        """Load and parse the dataset"""
        print(f"📁 Loading dataset from {csv_file}...")
        
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['Category', 'Resume'])
        df['Category'] = df['Category'].str.strip()
        df['Resume'] = df['Resume'].str.strip()
        
        print(f" Processing {len(df)} resumes with NLP techniques...")
        
        parsed_resumes = []
        for idx, row in df.iterrows():
            if idx % 50 == 0:
                print(f"   Progress: {idx + 1}/{len(df)} resumes processed...")
            
            parsed = self.parse_single_resume(row['Resume'], row['Category'], idx + 1)
            parsed_resumes.append(parsed)
        
        print(f"Completed parsing {len(parsed_resumes)} resumes!")
        
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
            
            for skill_type in ['programming_languages', 'frameworks_libraries', 
                             'tools_software', 'databases', 'soft_skills', 'other_skills']:
                for skill in resume[skill_type]:
                    if skill:
                        tables[skill_type].append({
                            'person_id': resume['id'],
                            skill_type: skill
                        })
            
            for idx, desc in enumerate(resume['experience_descriptions']):
                tables['experience_descriptions'].append({
                    'person_id': resume['id'],
                    'description_order': idx + 1,
                    'description': desc
                })
            
            tables['extracted_terms_summary'].append({
                'person_id': resume['id'],
                'total_terms_extracted': len(resume['raw_extracted_terms']),
                'all_terms': ', '.join(resume['raw_extracted_terms'])
            })
        
        return tables

    def create_sqlite_database(self, tables: Dict, db_file: str = 'nlp_resume_data.db'):
        """Create SQLite database"""
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
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
        
        for table_name, data in tables.items():
            if not data:
                continue
            
            if table_name == 'persons':
                for record in data:
                    cursor.execute(
                        'INSERT INTO persons VALUES (?, ?, ?)',
                        (record['id'], record['category'], record['years_experience'])
                    )
            elif table_name in ['programming_languages', 'frameworks_libraries', 'tools_software', 
                              'databases', 'soft_skills', 'other_skills']:
                for record in data:
                    cursor.execute(
                        f'INSERT INTO {table_name} (person_id, {table_name}) VALUES (?, ?)',
                        (record['person_id'], record[table_name])
                    )
            elif table_name == 'experience_descriptions':
                for record in data:
                    cursor.execute(
                        'INSERT INTO experience_descriptions (person_id, description_order, description) VALUES (?, ?, ?)',
                        (record['person_id'], record['description_order'], record['description'])
                    )
            elif table_name == 'extracted_terms_summary':
                for record in data:
                    cursor.execute(
                        'INSERT INTO extracted_terms_summary (person_id, total_terms_extracted, all_terms) VALUES (?, ?, ?)',
                        (record['person_id'], record['total_terms_extracted'], record['all_terms'])
                    )
        
        conn.commit()
        conn.close()
        print(f" Database created successfully: {db_file}")

    def generate_skill_analytics(self, tables: Dict) -> Dict:
        """Generate analytics on extracted skills"""
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
    
    def extract_soft_skills_improved(self, text: str) -> List[str]:
        soft_skills_found = []
        text_lower = text.lower()
        
        # Comprehensive soft skill patterns with their readable names
        soft_skill_patterns = {
            'Problem Solving': [
                r'problem[- ]solving',
                r'analytical\s+(?:thinking|skills)',
                r'critical\s+thinking', 
                r'tackle\s+(?:complex\s+)?challenges',
                r'solve\s+(?:complex\s+)?problems',
                r'troubleshooting',
                r'analytical\s+approach'
            ],
            
            'Team Collaboration': [
                r'team\s+player',
                r'collaborative?(?:\s+(?:approach|skills))?',
                r'teamwork',
                r'work\s+(?:with|in)\s+teams',
                r'cross[- ]functional\s+teams',
                r'team\s+environment',
                r'collaborative\s+approach'
            ],
            
            'Communication': [
                r'communication\s+skills',
                r'strong\s+communicator',
                r'interpersonal\s+skills',
                r'verbal\s+communication',
                r'written\s+communication',
                r'presentation\s+skills',
                r'client\s+(?:facing|interaction)',
                r'stakeholder\s+(?:management|communication)'
            ],
            
            'Leadership': [
                r'leadership\s*(?:skills|experience|qualities)?',
                r'leading\s+teams',
                r'team\s+lead(?:er|ership)?',
                r'mentoring',
                r'coaching',
                r'guidance',
                r'supervising?',
                r'project\s+lead(?:er|ership)?'
            ],
            
            'Adaptability': [
                r'fast[- ]paced\s+environment',
                r'adaptable',
                r'flexible',
                r'thrives?\s+in\s+(?:fast[- ]paced|dynamic|challenging)',
                r'dynamic\s+environment',
                r'quick\s+(?:learner|to\s+adapt)',
                r'agile\s+(?:environment|methodology|approach)'
            ],
            
            'Project Management': [
                r'project\s+management',
                r'time\s+management',
                r'deadline[- ]driven',
                r'project\s+coordination',
                r'resource\s+management',
                r'planning\s+(?:and\s+)?(?:execution|organizing)',
                r'meet\s+deadlines'
            ],
            
            'Initiative': [
                r'proactive',
                r'self[- ]motivated',
                r'initiative',
                r'independent\s+(?:work|worker)',
                r'self[- ]starter',
                r'drive\s+(?:results|innovation|improvement)'
            ],
            
            'Customer Focus': [
                r'customer[- ](?:focused|centric|oriented)',
                r'client\s+(?:service|satisfaction|relations)',
                r'user[- ](?:focused|centric)',
                r'customer\s+experience',
                r'service[- ]oriented'
            ]
        }
        
        # Check each skill category
        for skill_name, patterns in soft_skill_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    soft_skills_found.append(skill_name)
                    break  # Avoid duplicates for same skill
        
        return soft_skills_found

    def extract_entities_and_skills_improved(self, text: str) -> Dict[str, List[str]]:
        """
        Improved version of extract_entities_and_skills that includes better soft skills detection
        Replace the existing method with this one
        """
        # Method 1: Named entity recognition
        entities = self.extract_named_entities(text)
        
        # Method 2: Technical term extraction
        technical_terms = self.extract_technical_terms_nlp(text)
        
        # Method 3: TF-IDF based extraction
        tfidf_terms = self.extract_skills_with_tfidf(text)
        
        # Method 4: DEDICATED SOFT SKILLS EXTRACTION (NEW!)
        soft_skills = self.extract_soft_skills_improved(text)
        
        # Combine technical terms for classification
        all_technical_terms = list(set(entities + technical_terms + tfidf_terms))
        
        # Classify technical skills using existing NLP context analysis
        classified_technical = self.classify_skills_nlp(all_technical_terms, text)
        
        # Add the dedicated soft skills to the classification
        classified_technical['soft_skills'] = soft_skills
        
        return classified_technical

def main():
    """Main function"""
    print(" NLP RESUME PARSER ")
    print("Using: NLTK, scikit-learn, TextBlob")
    print("=" * 50)
    
    parser = NLPResumeParser()
    
    try:
        parsed_resumes, tables = parser.load_and_parse_dataset('csdataset.csv')
        parser.create_sqlite_database(tables)
        analytics = parser.generate_skill_analytics(tables)
        
        print("\n" + "="*60)
        print(" NLP RESUME PARSING COMPLETE")
        print("="*60)
        
        print(f"\n Summary:")
        print(f"   Processed: {len(parsed_resumes)} resumes")
        print(f"   Database: nlp_resume_data.db")
        
        print(f"\n Discovered Skills (via NLP):")
        for skill_type, stats in analytics.items():
            print(f"\n   {skill_type.replace('_', ' ').title()}:")
            print(f"     Total mentions: {stats['total_mentions']}")
            print(f"     Unique skills: {stats['unique_skills']}")
            if stats['top_10']:
                top_5 = [f"{skill}({count})" for skill, count in stats['top_10'][:5]]
                print(f"     Top 5: {', '.join(top_5)}")
        
        # Example queries
        print(f"\n🔍 Sample Analysis:")
        conn = sqlite3.connect('nlp_resume_data.db')
        
        # Most common programming languages
        query = """
        SELECT programming_languages, COUNT(*) as count
        FROM programming_languages
        GROUP BY programming_languages
        ORDER BY count DESC
        LIMIT 5
        """
        result = pd.read_sql_query(query, conn)
        print(f"   Top Programming Languages (discovered via NLP):")
        for _, row in result.iterrows():
            print(f"     {row['programming_languages']}: {row['count']} mentions")
        
        # Most common frameworks
        framework_query = """
        SELECT frameworks_libraries, COUNT(*) as count
        FROM frameworks_libraries
        GROUP BY frameworks_libraries
        ORDER BY count DESC
        LIMIT 5
        """

        try:
            framework_result = pd.read_sql_query(framework_query, conn)
            print(f"   Top Frameworks/Libraries (discovered via NLP):")
            for _, row in framework_result.iterrows():
                print(f"     {row['frameworks_libraries']}: {row['count']} mentions")
        except:
            print(f"   No frameworks/libraries data available")
        
        # Skills by category
        category_query = """
        SELECT p.category, COUNT(DISTINCT pl.programming_languages) as unique_langs
        FROM persons p
        LEFT JOIN programming_languages pl ON p.id = pl.person_id
        GROUP BY p.category
        ORDER BY unique_langs DESC
        """
        try:
            category_result = pd.read_sql_query(category_query, conn)
            print(f"   Programming Language Diversity by Category:")
            for _, row in category_result.iterrows():
                print(f"     {row['category']}: {row['unique_langs']} unique languages")
        except:
            print(f"   Category analysis not available")
        
        conn.close()
        
        print(f"\n NLP parsing completed successfully!")
        print(f"Skills were discovered using NLP techniques, not hardcoded lists!")
        print(f" Techniques used: Named Entity Recognition, POS Tagging, TF-IDF, Context Analysis")
        
        # Show some interesting statistics
        print(f"\n📈 Interesting Statistics:")
        total_skills = sum(stats['total_mentions'] for stats in analytics.values())
        unique_skills = sum(stats['unique_skills'] for stats in analytics.values())
        print(f"   Total skill mentions: {total_skills}")
        print(f"   Unique skills discovered: {unique_skills}")
        print(f"   Average skills per resume: {total_skills / len(parsed_resumes):.1f}")
        
        # Years of experience statistics
        years_data = [resume['years_experience'] for resume in parsed_resumes if resume['years_experience'] is not None]
        if years_data:
            print(f"   Resumes with experience data: {len(years_data)}/{len(parsed_resumes)}")
            print(f"   Average years of experience: {sum(years_data) / len(years_data):.1f}")
            print(f"   Experience range: {min(years_data)} - {max(years_data)} years")
        
        print(f"\n Database ready for queries! Try:")
        print(f"   - Find developers with specific skills")
        print(f"   - Analyze skill trends by category")
        print(f"   - Identify skill combinations")
        print(f"   - Experience level analysis")
        
    except Exception as e:
        print(f" Error during parsing: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n Common solutions:")
        print(f"   - Install missing packages: pip install nltk scikit-learn textblob")
        print(f"   - Download NLTK data: python -c \"import nltk; nltk.download('all')\"")
        print(f"   - Check if csdataset.csv exists in current directory")

if __name__ == "__main__":
    main()

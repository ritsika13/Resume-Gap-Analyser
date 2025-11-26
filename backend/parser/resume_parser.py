import sqlite3
import re
import os
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")

# Make pandas optional
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("WARNING: pandas not available - some analytics features will be limited")

# NLP Libraries
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("WARNING: NLTK not available - using basic text processing")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not available - using basic feature extraction")

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    print("WARNING: TextBlob not available - using basic sentiment analysis")

# File processing libraries
try:
    import PyPDF2
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("WARNING: PDF libraries not available - install PyPDF2 and pdfplumber for PDF support")

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("WARNING: python-docx not available - install for DOCX support")

class ResumeParser:
    """
    Resume parser that extracts and categorizes skills from individual resume files.
    Similar structure to csv_parser.py but focuses on single resume processing.
    """
    
    def __init__(self):
        print("=� Initializing Resume Parser")
        
        # Download required NLTK data
        if HAS_NLTK:
            self._download_nltk_data()
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            self.lemmatizer = None
        
        # Initialize skill classification patterns (same as csv_parser.py)
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

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            
            elif file_extension == '.pdf' and HAS_PDF:
                return self._extract_from_pdf(file_path)
            
            elif file_extension == '.docx' and HAS_DOCX:
                return self._extract_from_docx(file_path)
            
            else:
                print(f"L Unsupported file format: {file_extension}")
                return ""
                
        except Exception as e:
            print(f"L Error reading file {file_path}: {e}")
            return ""

    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        
        # Try pdfplumber first (better text extraction)
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text
        except:
            pass
        
        # Fallback to PyPDF2
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"L Error extracting PDF text: {e}")
        
        return text

    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"L Error extracting DOCX text: {e}")
            return ""

    def extract_years_experience(self, text: str) -> Optional[int]:
        """Extract years of experience using regex patterns"""
        if not text:
            return None
            
        # Year patterns with context (same as csv_parser.py)
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
                
                # Technical terms are often proper nouns or nouns in technical contexts
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
        """Classify skills using NLP context analysis (same as csv_parser.py)"""
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
        """Classify term using heuristic rules (same as csv_parser.py)"""
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

    def extract_soft_skills_improved(self, text: str) -> List[str]:
        """Enhanced soft skills extraction (same as csv_parser.py)"""
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
            ]
        }
        
        # Check each skill category
        for skill_name, patterns in soft_skill_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    soft_skills_found.append(skill_name)
                    break  # Avoid duplicates for same skill
        
        return soft_skills_found

    def extract_experience_descriptions(self, text: str) -> List[str]:
        """Extract experience descriptions using NLP (same as csv_parser.py)"""
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
        
        # Method 4: Dedicated soft skills extraction
        soft_skills = self.extract_soft_skills_improved(text)
        
        # Combine technical terms for classification
        all_technical_terms = list(set(entities + technical_terms + tfidf_terms))
        
        # Classify technical skills using NLP context analysis
        classified_technical = self.classify_skills_nlp(all_technical_terms, text)
        
        # Add the dedicated soft skills to the classification
        classified_technical['soft_skills'] = soft_skills
        
        return classified_technical

    def parse_resume_text(self, resume_text: str, resume_id: int, category: str = "Unknown") -> Dict:
        """Parse a single resume text using NLP techniques"""
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

    def parse_resume_file(self, file_path: str, resume_id: int, category: str = "Unknown") -> Dict:
        """Parse a resume from file"""
        print(f"=� Processing resume file: {os.path.basename(file_path)}")
        
        # Extract text from file
        resume_text = self.extract_text_from_file(file_path)
        
        if not resume_text.strip():
            print(f"L No text extracted from {file_path}")
            return None
        
        # Parse the extracted text
        return self.parse_resume_text(resume_text, resume_id, category)

    def create_normalized_tables(self, parsed_resumes: List[Dict]) -> Dict:
        """Create normalized table structures (same as csv_parser.py)"""
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
            if resume is None:  # Skip failed parses
                continue
                
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

    def create_sqlite_database(self, tables: Dict, db_file: str = 'resume_parsing_results.db'):
        """Create SQLite database (same schema as csv_parser.py)"""
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
        print(f"=� Database created successfully: {db_file}")

    def generate_skill_analytics(self, tables: Dict) -> Dict:
        """Generate analytics on extracted skills (same as csv_parser.py)"""
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
    """Example usage of the ResumeParser"""
    print("=� Resume Parser - Individual Resume Processing")
    print("=" * 60)
    
    parser = ResumeParser()
    
    # Example: Parse a single resume text
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
    
    try:
        # Parse the sample resume
        parsed_resume = parser.parse_resume_text(sample_resume, resume_id=1, category="Software Developer")
        
        print("\n=� Parsing Results:")
        print(f"Years of Experience: {parsed_resume['years_experience']}")
        print(f"Programming Languages: {parsed_resume['programming_languages']}")
        print(f"Frameworks/Libraries: {parsed_resume['frameworks_libraries']}")
        print(f"Tools/Software: {parsed_resume['tools_software']}")
        print(f"Databases: {parsed_resume['databases']}")
        print(f"Soft Skills: {parsed_resume['soft_skills']}")
        print(f"Other Skills: {parsed_resume['other_skills']}")
        print(f"Experience Descriptions: {len(parsed_resume['experience_descriptions'])} found")
        
        # Create tables and database
        tables = parser.create_normalized_tables([parsed_resume])
        parser.create_sqlite_database(tables)
        
        # Generate analytics
        analytics = parser.generate_skill_analytics(tables)
        
        print("\n=� Analytics:")
        for skill_type, stats in analytics.items():
            if stats['top_10']:
                print(f"{skill_type.replace('_', ' ').title()}: {[skill for skill, count in stats['top_10'][:3]]}")
        
        print("\n Resume parsing completed successfully!")
        
    except Exception as e:
        print(f"L Error during parsing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Tech Stack & Architecture

SkillSight is a career insight platform with a multi-component architecture:

- **Backend**: Python 3.10.13, FastAPI, SQLite with SQLAlchemy ORM
- **Frontend**: React.js (legacy), HTML, CSS  
- **Voice Dashboard**: Next.js 15.3.5 with TypeScript, TailwindCSS, React 19
- **Data Processing**: NLP-based resume parsing using NLTK, scikit-learn, TextBlob

## Development Environment Setup

### Python Environment
```bash
# Install Python 3.10.13 (required for compatibility)
pyenv install 3.10.13
pyenv local 3.10.13

# Create and activate virtual environment
python3 -m venv skillsight-env
source skillsight-env/bin/activate  # macOS/Linux

# Install backend dependencies
pip install -r backend/requirements.txt
```

### Frontend Development
```bash
# Legacy React frontend
cd frontend
npm install
npm start

# Next.js Voice Dashboard
cd voice-dashboard
npm install
npm run dev
```

## Key Development Commands

### Backend
```bash
# Run FastAPI server (from project root)
cd backend
uvicorn main:app --reload

# Run resume parser and NLP processing
cd backend/parser
python csv_parser.py

# Run database tests
python backend/parser/full_tester.py
python backend/parser/quick_test.py
```

### Voice Dashboard
```bash
cd voice-dashboard
npm run dev          # Development server
npm run build        # Production build
npm run start        # Production server
npm run lint         # ESLint checks
```

## Database Architecture

The system uses SQLAlchemy models (`backend/database/models.py`):

- **User**: User accounts with authentication
- **Skill**: User skills with confidence scores and types
- **Role**: Job roles in the system
- **RoleSkill**: Skills required for specific roles
- **UserRoleInterest**: Tracks user interest in roles

Database connection configured in `backend/database/db.py` using SQLite (`skillsight.db`).

## Resume Processing Pipeline

The NLP resume parser (`backend/parser/csv_parser.py`) includes:

1. **Text Processing**: NLTK tokenization, POS tagging, named entity recognition
2. **Skill Extraction**: TF-IDF analysis, technical term extraction, context-based classification
3. **Classification**: Programming languages, frameworks, tools, databases, soft skills
4. **Experience Analysis**: Years of experience extraction, experience descriptions
5. **Database Storage**: Normalized tables for skills, analytics, and relationships

Key parser methods:
- `extract_years_experience()`: Extract experience from text
- `extract_entities_and_skills()`: Main skill extraction pipeline
- `classify_skills_nlp()`: Context-based skill classification
- `extract_soft_skills_improved()`: Dedicated soft skills detection

## Project Structure

```
SkillSight/
├── backend/
│   ├── main.py                    # FastAPI application entry
│   ├── database/
│   │   ├── models.py             # SQLAlchemy models
│   │   ├── crud.py               # Database operations
│   │   └── db.py                 # Database configuration
│   ├── parser/
│   │   ├── csv_parser.py         # NLP resume parser
│   │   ├── resume_parser.py      # Parser interface
│   │   └── utils.py              # Parsing utilities
│   └── matcher/
│       └── skill_matcher.py      # Skill matching algorithms
├── frontend/                     # Legacy React app
├── voice-dashboard/              # Next.js TypeScript app
└── data/                         # CSV datasets and resumes
```

## Data Processing

- **Dataset**: `data/csdataset.csv` contains resume data for processing
- **Skills Database**: `data/skills_database.json` for skill definitions
- **Resume Storage**: `data/resumes/` for uploaded resume files
- **Processing Output**: Creates `nlp_resume_data.db` with extracted skills

## Development Notes

- **Python Version**: Locked to 3.10.13 for NLP library compatibility
- **Database**: SQLite for development, easily migrated to PostgreSQL for production
- **NLP Dependencies**: Parser gracefully degrades if optional libraries (spaCy) unavailable
- **Skill Classification**: Uses context-aware NLP rather than hardcoded skill lists
- **Error Handling**: Database operations include integrity error handling for duplicates

## Testing & Validation

Run skill extraction tests:
```bash
cd backend/parser
python full_tester.py     # Complete dataset processing
python quick_test.py      # Quick validation test
```

The parser includes analytics generation for skill frequency analysis and validation of extraction quality.

## Large Codebase Analysis with Gemini CLI

When analyzing large codebases or multiple files that might exceed context limits, use the Gemini CLI with its massive context window:

```bash
# Single file analysis
gemini -p "@backend/parser/csv_parser.py Explain this file's purpose and structure"

# Multiple files
gemini -p "@backend/database/models.py @backend/database/crud.py Analyze the database implementation"

# Entire directory analysis
gemini -p "@backend/ Summarize the backend architecture"

# Multiple directories
gemini -p "@backend/ @frontend/ Analyze the full-stack implementation"

# Current directory and subdirectories
gemini -p "@./ Give me an overview of the entire SkillSight project"

# Or use --all_files flag
gemini --all_files -p "Analyze the project structure and dependencies"
```

### Implementation Verification Examples

```bash
# Check if authentication is implemented
gemini -p "@backend/ @frontend/ Is user authentication implemented? Show relevant files and functions"

# Verify NLP processing implementation
gemini -p "@backend/parser/ Is resume parsing with NLP fully implemented? List all processing steps"

# Check database relationships
gemini -p "@backend/database/ Are skill-role relationships properly implemented in the database models?"

# Verify API endpoints
gemini -p "@backend/ Are all CRUD operations implemented for users, skills, and roles?"

# Check frontend components
gemini -p "@frontend/ @voice-dashboard/ What UI components are implemented for the skill matching interface?"
```

### When to Use Gemini CLI

Use `gemini -p` when:
- Analyzing entire codebases or large directories
- Need to understand project-wide patterns or architecture  
- Current context window is insufficient for the task
- Working with files totaling more than 100KB
- Verifying if specific features are implemented across the codebase
- Checking for coding patterns across multiple components

**Note**: Paths in @ syntax are relative to your current working directory when invoking gemini.
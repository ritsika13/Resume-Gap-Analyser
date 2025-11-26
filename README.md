# SkillSight

SkillSight is a career insight platform that helps users plan their career growth. Once users indicate which roles they are interested in, it compares the user's resume to the skills required for those roles and recommends jobs, online courses, and other next steps to help them close the gap.

The goal is to provide personalized, data-driven guidance to help users confidently progress toward their career goals.

## 🚀 Tech Stack

- **Backend:** Python, FastAPI, SQLite, SQLAlchemy
- **Frontend:** React.js, HTML, CSS


## 📦 Python Version Notes

We're using Python 3.10.13 to ensure compatibility with core libraries and our backend modules.

Later on, if we build LLM-based features (e.g. chatbots, PDF parsing with LlamaIndex), we may use Python 3.11+ in a separate environment or feature branch to support those tools.

This keeps our base system stable while giving us flexibility for future expansion.

---

## 🛠️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/sadac1/skillsight.git
cd skillsight
```

### 2. Install correct Python version

```bash
brew install pyenv
pyenv install 3.10.13
pyenv local 3.10.13  # setting it locally for this project
```

To verify: 
```bash
python --version
```

### 3. Create a virtual environment

```bash
python3 -m venv skillsight-env
source skillsight-env/bin/activate  # on macOS/Linux
```

### 4. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 5. Run the backend (not yet built so skip)
```bash
uvicorn main:app --reload
```

---

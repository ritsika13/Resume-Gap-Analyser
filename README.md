# CareerBridge

CareerBridge is an AI-powered career development platform that analyzes your resume against real-world professionals in your target role. Simply
paste your resume and select your desired job title to discover which skills you already have and which ones you need to learn. The platform
compares your skillset against a database of 400+ parsed resumes from various tech roles, providing personalized learning recommendations with
curated resources from top platforms like Udemy, Coursera, and YouTube.

## 🚀 Tech Stack

- **Backend:** Python 3.10+, FastAPI, SQLite
- **Frontend:** Vanilla JavaScript, HTML5, CSS3
- **NLP & Data Processing**: NLTK, scikit-learn, TextBlob
- **Design**: Montserrat font, custom color palette
- **Data**: 400+ professionally parsed resumes across multiple tech roles


## How to Run 

### Start the Backend
  cd backend
  python3 main.py
  The API will be available at http://localhost:8000

### Start the Frontend

  cd frontend
  python3 -m http.server 3000
  Open http://localhost:3000 in your browser

  Features

  Hybrid skill extraction using section parsing, dictionary matching, and NLP Real-world skill gap analysis based on actual professionals📊
  Match percentage and detailed skill breakdowns by category. Curated learning resources for missing skills. Modern, responsive UI with
  professional design



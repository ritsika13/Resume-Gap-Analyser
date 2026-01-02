# CareerBridge - Setup Instructions

## Quick Start Guide

### Step 1: Install Dependencies

```bash
# Install Python dependencies
pip install fastapi uvicorn python-multipart

# Verify existing dependencies are installed
pip install nltk scikit-learn textblob pandas
```

### Step 2: Start the Backend Server

```bash
cd backend
python3 main.py
```

You should see:
```
Starting CareerBridge API...
API will be available at: http://localhost:8000
API documentation at: http://localhost:8000/docs
```

### Step 3: Open the Frontend

Open `frontend/index.html` in your web browser:

**Option A: Double-click the file**
- Navigate to the `frontend` folder
- Double-click `index.html`

**Option B: Use a local server (recommended)**
```bash
cd frontend
python3 -m http.server 3000
# Then open: http://localhost:3000
```

### Step 4: Test the Application

1. **Paste a resume** in the text area (or use the sample below)
2. **Select a target role** from the dropdown
3. **Click "Analyze My Skills"**
4. **View your results!**

---

## Sample Resume for Testing

```
Frontend Developer with 3 years of experience building responsive web applications.

SKILLS:
- HTML, CSS, JavaScript
- React.js for building user interfaces
- Git for version control
- Responsive design and CSS frameworks

EXPERIENCE:
Developed multiple React-based single-page applications
Collaborated with backend developers to integrate RESTful APIs
Implemented responsive designs using modern CSS techniques
```

---

## API Endpoints

The backend provides the following endpoints:

### GET /roles
Get all available job roles
```bash
curl http://localhost:8000/roles
```

### POST /analyze
Analyze resume and get skill gaps
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Your resume text here...",
    "target_role": "Frontend Developer"
  }'
```

### GET /roles/{role_name}/top-skills
Get top skills for a specific role
```bash
curl http://localhost:8000/roles/Frontend%20Developer/top-skills
```

### GET /docs
Interactive API documentation (Swagger UI)
- Open: http://localhost:8000/docs

---

## Project Structure

```
CareerBridge/
├── backend/
│   ├── main.py                    # FastAPI server
│   ├── parser/
│   │   └── csv_parser.py          # NLP resume parser
│   ├── matcher/
│   │   ├── skill_matcher.py       # Skill matching engine
│   │   └── skill_matcher_tester.py
│   └── database/
│       └── models.py              # Database models
├── frontend/
│   └── index.html                 # Web interface
├── data/
│   └── csdataset.csv             # Resume dataset (400 resumes)
└── parser/
    └── nlp_resume_data.db        # Parsed skills database
```

---

## How It Works

1. **User Input**: User pastes resume text and selects target role
2. **NLP Parsing**: Backend extracts skills using NLTK, TF-IDF, and context analysis
3. **Skill Matching**: Compares user skills against 400 real professionals in the database
4. **Gap Analysis**: Identifies missing skills and calculates match percentage
5. **Recommendations**: Provides prioritized learning resources

---

## Available Roles

- Frontend Developer (54 professionals)
- Backend Developer (57 professionals)
- Python Developer (45 professionals)
- Data Scientist (53 professionals)
- Full Stack Developer (47 professionals)
- Machine Learning Engineer (43 professionals)
- Mobile App Developer (45 professionals)
- Cloud Engineer (56 professionals)

---

## Troubleshooting

### "Error loading roles"
- Make sure the backend server is running on port 8000
- Check console for CORS errors

### "CORS policy error"
- The backend is configured to allow all origins
- If issues persist, try using a local server for the frontend instead of opening the file directly

### "Analysis failed"
- Ensure resume text is at least 50 characters
- Check that a target role is selected
- Verify the backend server is running

### Port 8000 already in use
```bash
# Find and kill the process using port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn main:app --port 8001
# Then update API_URL in frontend/index.html to http://localhost:8001
```

---

## Testing the Skill Matcher

You can test the skill matcher independently:

```bash
cd backend/matcher
python3 skill_matcher_tester.py
```

This will show detailed analysis for 4 different user profiles.

---

## Next Steps

### Enhancements You Can Add:

1. **File Upload**: Allow PDF/DOCX resume uploads
2. **Export Results**: Download results as PDF
3. **Multiple Roles**: Compare against multiple roles simultaneously
4. **Skill Trends**: Show trending skills in each role
5. **Learning Progress**: Track which skills you've learned
6. **Share Results**: Generate shareable links

### Production Deployment:

1. **Database**: Migrate from SQLite to PostgreSQL
2. **Authentication**: Add user accounts (optional)
3. **Caching**: Add Redis for faster responses
4. **Hosting**: Deploy to Heroku, AWS, or Vercel

---

## Support

For issues or questions:
- Check the troubleshooting section above
- Review the API docs: http://localhost:8000/docs
- Examine the browser console for frontend errors
- Check the terminal for backend errors

---

**Built with:**
- FastAPI (Backend)
- NLTK, scikit-learn, TextBlob (NLP)
- Vanilla JavaScript (Frontend)
- SQLite (Database)

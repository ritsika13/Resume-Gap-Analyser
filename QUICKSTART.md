# CareerBridge - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Start the Backend Server

```bash
cd backend
python3 main.py
```

**Expected output:**
```
Starting CareerBridge API...
API will be available at: http://localhost:8000
API documentation at: http://localhost:8000/docs
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ **Keep this terminal window open!**

---

### Step 2: Open the Frontend

**Option A - Direct File (Simple)**
- Open `frontend/index.html` in your browser (Chrome/Firefox/Safari)

**Option B - Local Server (Recommended)**
```bash
# In a NEW terminal window
cd frontend
python3 -m http.server 3000
```
Then open: http://localhost:3000

---

### Step 3: Test with Sample Resume

1. **Paste this sample resume:**

```
Experienced Frontend Developer with 3 years building modern web applications.

TECHNICAL SKILLS:
- HTML5, CSS3, JavaScript (ES6+)
- React.js for component-based UIs
- Git and GitHub for version control
- Responsive design using Flexbox and Grid

EXPERIENCE:
Built 5+ React applications with Redux state management
Collaborated with design teams to implement pixel-perfect UIs
Integrated RESTful APIs with Axios
Optimized web performance achieving 95+ Lighthouse scores

Seeking to expand skills in Vue.js and Angular frameworks
```

2. **Select role:** Frontend Developer

3. **Click:** "Analyze My Skills"

4. **View your results!**

---

## 📊 What You'll See

- **Match Score**: Your % match with real professionals
- **Skills You Have**: Green checkmarks for matched skills
- **Skills to Learn**: Red X for missing skills with priority levels
- **Learning Path**: Direct links to courses on Coursera, Codecademy, freeCodeCamp

---

## 🔧 Troubleshooting

**Backend won't start?**
```bash
# Install dependencies
pip install fastapi uvicorn

# Try again
cd backend
python3 main.py
```

**"Error loading roles"?**
- Check backend is running on http://localhost:8000
- Visit http://localhost:8000/roles to test directly

**CORS errors?**
- Use Option B (local server) instead of opening HTML file directly

---

## 📚 API Documentation

Interactive API docs available at: **http://localhost:8000/docs**

Test endpoints directly in your browser!

---

## ✨ Ready for More?

See `SETUP_INSTRUCTIONS.md` for:
- Detailed API documentation
- Testing the skill matcher independently
- Production deployment guide
- Enhancement ideas

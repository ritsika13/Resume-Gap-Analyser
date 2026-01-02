"""
CareerBridge - FastAPI Backend
Simple API for resume skill gap analysis
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
import os

# Add parser and matcher to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'parser'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'matcher'))

from parser.csv_parser import NLPResumeParser
from matcher.skill_matcher import SkillMatcher
from parser.skill_filter import filter_skill_dict
from parser.resume_section_parser import extract_skills_from_sections, merge_skill_dicts
from parser.dictionary_skill_extractor import extract_skills_from_text, merge_extracted_skills

# Initialize FastAPI app
app = FastAPI(
    title="CareerBridge API",
    description="Resume skill gap analysis API",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NLP parser and skill matcher
nlp_parser = NLPResumeParser()

# Determine database path relative to backend directory
db_path = os.path.join(os.path.dirname(__file__), 'parser', 'nlp_resume_data.db')
skill_matcher = SkillMatcher(db_path)

# Request/Response Models
class ResumeAnalysisRequest(BaseModel):
    resume_text: str
    target_role: str
    min_frequency: Optional[int] = 3

class SkillInfo(BaseModel):
    skill: str
    frequency: Optional[int] = None
    percentage: Optional[float] = None

class LearningResource(BaseModel):
    platform: str
    course: str
    url: str

class SkillRecommendation(BaseModel):
    skill: str
    category: str
    importance: str
    possessed_by_percentage: float
    resources: List[LearningResource]

class ResumeAnalysisResponse(BaseModel):
    role: str
    match_percentage: float
    matched_skills: Dict[str, List[SkillInfo]]
    missing_skills: Dict[str, List[SkillInfo]]
    summary: Dict[str, int]
    learning_recommendations: List[SkillRecommendation]
    role_metadata: Dict


@app.get("/")
async def root():
    return {
        "message": "CareerBridge API",
        "version": "1.0.0",
        "endpoints": {
            "/roles": "Get available job roles",
            "/analyze": "Analyze resume and get skill gaps",
            "/docs": "API documentation"
        }
    }


@app.get("/roles")
async def get_available_roles():
    try:
        roles = skill_matcher.get_available_roles()
        return {
            "success": True,
            "roles": roles,
            "total_roles": len(roles)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching roles: {str(e)}")


@app.get("/roles/{role_name}/top-skills")
async def get_role_top_skills(role_name: str, top_n: int = 10):
    try:
        top_skills = skill_matcher.get_top_skills_by_role(role_name, top_n=top_n)
        return {
            "success": True,
            "role": role_name,
            "top_skills": top_skills
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching top skills: {str(e)}")


@app.post("/analyze")
async def analyze_resume(request: ResumeAnalysisRequest):
    try:
        if not request.resume_text or len(request.resume_text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Resume text is too short. Please provide at least 50 characters."
            )

        if not request.target_role:
            raise HTTPException(
                status_code=400,
                detail="Target role is required."
            )

        print(f"Parsing resume for role: {request.target_role}")

        # Method 1: Section-based extraction
        section_skills = extract_skills_from_sections(request.resume_text)
        print(f"✓ Section-based extraction: {sum(len(v) for v in section_skills.values())} skills")

        # Method 2: Dictionary-based extraction
        dictionary_skills = extract_skills_from_text(request.resume_text)
        print(f"✓ Dictionary-based extraction: {sum(len(v) for v in dictionary_skills.values())} skills")

        # Method 3: Soft skills extraction
        soft_skills = nlp_parser.extract_soft_skills_improved(request.resume_text)
        print(f"✓ Soft skills extraction: {len(soft_skills)} skills")

        # Merge all three methods
        extracted_skills = merge_extracted_skills(section_skills, dictionary_skills)
        extracted_skills['soft_skills'] = soft_skills

        print(f"✓ Total unique skills extracted: {sum(len(v) for v in extracted_skills.values())}")

        # Apply filter
        extracted_skills = filter_skill_dict(extracted_skills)
        print(f"✓ After filtering: {sum(len(v) for v in extracted_skills.values())} skills")

        user_skills = {
            'programming_languages': extracted_skills.get('programming_languages', []),
            'frameworks_libraries': extracted_skills.get('frameworks_libraries', []),
            'tools_software': extracted_skills.get('tools_software', []),
            'databases': extracted_skills.get('databases', []),
            'soft_skills': extracted_skills.get('soft_skills', [])
        }

        print(f"Extracted skills: {sum(len(v) for v in user_skills.values())} total")

        comparison = skill_matcher.compare_skills(
            user_skills,
            request.target_role,
            min_frequency=request.min_frequency
        )

        learning_recommendations = skill_matcher.get_learning_resources(
            comparison['missing_skills']
        )

        response = {
            "success": True,
            "role": comparison['role'],
            "match_percentage": comparison['match_percentage'],
            "matched_skills": comparison['matched_skills'],
            "missing_skills": comparison['missing_skills'],
            "summary": comparison['summary'],
            "learning_recommendations": learning_recommendations[:10],
            "role_metadata": comparison['role_metadata'],
            "user_skills_extracted": user_skills
        }

        return response

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing resume: {str(e)}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "parser_loaded": nlp_parser is not None,
        "matcher_loaded": skill_matcher is not None
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting CareerBridge API...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

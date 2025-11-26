from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List
from . import models

# 1. Create a new user
def create_user(db: Session, email: str, hashed_password: str, name: str):
    user = models.User(
        email=email,
        hashed_password=hashed_password,
        name=name
    )
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
        return user
    except IntegrityError:
        db.rollback()
        return None  # handle duplicate email gracefully

# 2. Get user by email
def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

# 3. Insert skills parsed from resume
def insert_user_skills(db: Session, user_id: int, skills: List[str], skill_type: str = None):
    for skill_name in skills:
        skill = models.Skill(
            user_id=user_id,
            skill_name=skill_name,
            confidence=1.0,  # can update later if needed
            skill_type=skill_type
        )
        try:
            db.add(skill)
            db.commit()
        except IntegrityError:
            db.rollback()  # skip duplicate entries

# 4. Insert a new role + its associated skills
def insert_role_skills(db: Session, role_name: str, skills: List[str], skill_type: str = None):
    # Check if role already exists
    role = db.query(models.Role).filter(models.Role.role_name == role_name).first()
    if not role:
        role = models.Role(role_name=role_name)
        db.add(role)
        db.commit()
        db.refresh(role)

    # Insert role skills
    for skill_name in skills:
        role_skill = models.RoleSkill(
            role_id=role.id,
            skill_name=skill_name,
            skill_type=skill_type
        )
        try:
            db.add(role_skill)
            db.commit()
        except IntegrityError:
            db.rollback()  # skip duplicates

# 5. Record a user's interest in a specific role
def add_user_role_interest(db: Session, user_id: int, role_id: int, match_score: float = None):
    interest = models.UserRoleInterest(
        user_id=user_id,
        role_id=role_id,
        match_score=match_score
    )
    try:
        db.add(interest)
        db.commit()
    except IntegrityError:
        db.rollback()
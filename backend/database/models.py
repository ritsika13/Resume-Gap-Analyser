from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)

class Skill(Base):
    __tablename__ = "skills"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    skill_name = Column(String, nullable=False)
    confidence = Column(Float, nullable=True) #questioning
    skill_type = Column(String, nullable=True)

    user = relationship("User", backref="skills")

    # make sure same skill isn't added twice (handle errors in parser)
    __table_args__ = (
        UniqueConstraint("user_id", "skill_name", name="_user_skill_uc"),
    )

class UserRoleInterest(Base):
    __tablename__ = "user_role_interests"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    role_id = Column(Integer, ForeignKey("roles.id"))

    user = relationship("User", backref="interested_roles")
    role = relationship("Role", backref="interested_users")

class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    role_name = Column(String, unique=True, nullable=False)

    skills = relationship("RoleSkill", backref="role")

class RoleSkill(Base):
    __tablename__ = "role_skills"

    id = Column(Integer, primary_key=True)
    role_id = Column(Integer, ForeignKey("roles.id"))
    skill_name = Column(String, nullable=False)
    skill_type = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint("role_id", "skill_name", name="_role_skill_uc"),
    )
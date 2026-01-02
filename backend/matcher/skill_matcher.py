# -*- coding: utf-8 -*-
import sqlite3
from typing import Dict, List, Set, Tuple
from collections import Counter, defaultdict
import re
import sys
import os

# Add parser to path to import skill filter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'parser'))
from skill_filter import is_valid_skill


class SkillMatcher:
    """
    Matches user skills against aggregated skills from real professionals in the target role.
    Uses the parsed resume data from nlp_resume_data.db to provide real-world skill comparisons.
    """

    def __init__(self, db_path: str = 'nlp_resume_data.db'):
        """
        Initialize the SkillMatcher with database connection.

        Args:
            db_path: Path to the SQLite database with parsed resume data
        """
        self.db_path = db_path
        self.skill_categories = [
            'programming_languages',
            'frameworks_libraries',
            'tools_software',
            'databases',
            'soft_sk'
            'ills'
        ]

    def normalize_skill(self, skill: str) -> str:
        """
        Normalize skill name for better matching.
        Example: 'Python', 'python', 'Python3' all become 'python'

        Args:
            skill: Raw skill name

        Returns:
            Normalized skill name
        """
        if not skill:
            return ""

        # Convert to lowercase
        normalized = skill.lower().strip()

        # Remove version numbers (Python3 -> python, Node.js 14 -> node.js)
        normalized = re.sub(r'\s*\d+(\.\d+)*\s*$', '', normalized)
        normalized = re.sub(r'\s*v?\d+(\.\d+)*\s*', ' ', normalized)

        # Remove common suffixes
        normalized = normalized.replace('.js', 'js')

        # Handle common variations
        variations = {
            'react.js': 'react',
            'reactjs': 'react',
            'vue.js': 'vue',
            'vuejs': 'vue',
            'node.js': 'nodejs',
            'express.js': 'express',
            'expressjs': 'express',
            'mongodb': 'mongo',
            'postgresql': 'postgres',
            'css3': 'css',
            'html5': 'html',
            'javascript': 'js',
        }

        normalized = variations.get(normalized, normalized)

        return normalized.strip()

    def get_role_skills(self, role_name: str, min_frequency: int = 2) -> Dict[str, Dict[str, int]]:
        """
        Aggregate all skills from people with the specified role in the database.

        Args:
            role_name: Target job role (e.g., 'Frontend Developer')
            min_frequency: Minimum number of times a skill must appear to be included

        Returns:
            Dictionary with skill categories and their frequencies:
            {
                'programming_languages': {'python': 50, 'javascript': 45, ...},
                'frameworks_libraries': {'react': 30, 'django': 25, ...},
                ...
            }
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # First, get all person IDs with this role
        cursor.execute(
            "SELECT id FROM persons WHERE category = ?",
            (role_name,)
        )
        person_ids = [row[0] for row in cursor.fetchall()]

        if not person_ids:
            conn.close()
            raise ValueError(f"No data found for role: {role_name}")

        role_skills = {}

        # For each skill category, aggregate skills
        for category in self.skill_categories:
            # Use dict of sets to track unique people per skill (prevents duplicates)
            skill_people = defaultdict(set)

            # Query all skills in this category for people with this role
            placeholders = ','.join('?' * len(person_ids))
            query = f"""
                SELECT person_id, {category}
                FROM {category}
                WHERE person_id IN ({placeholders})
            """

            cursor.execute(query, person_ids)

            for row in cursor.fetchall():
                person_id, skill = row
                if skill:
                    # Normalize the skill name
                    normalized = self.normalize_skill(skill)
                    # **FILTER OUT NOISE WORDS** from database
                    if normalized and is_valid_skill(normalized):
                        # Track unique people (not total occurrences)
                        skill_people[normalized].add(person_id)

            # Convert to counts (number of unique people per skill)
            skill_counter = {skill: len(people) for skill, people in skill_people.items()}

            # Filter by minimum frequency and convert to dict
            role_skills[category] = {
                skill: count
                for skill, count in skill_counter.items()
                if count >= min_frequency
            }

        conn.close()

        # Add metadata
        role_skills['_metadata'] = {
            'role_name': role_name,
            'total_professionals': len(person_ids),
            'min_frequency': min_frequency
        }

        return role_skills

    def get_top_skills_by_role(self, role_name: str, top_n: int = 10) -> Dict[str, List[Tuple[str, int]]]:
        """
        Get the top N most common skills for each category in a role.

        Args:
            role_name: Target job role
            top_n: Number of top skills to return per category

        Returns:
            Dictionary with top skills per category:
            {
                'programming_languages': [('python', 50), ('javascript', 45), ...],
                ...
            }
        """
        role_skills = self.get_role_skills(role_name, min_frequency=1)

        top_skills = {}
        for category in self.skill_categories:
            if category in role_skills:
                # Sort by frequency and get top N
                sorted_skills = sorted(
                    role_skills[category].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_n]
                top_skills[category] = sorted_skills

        return top_skills

    def compare_skills(
        self,
        user_skills: Dict[str, List[str]],
        role_name: str,
        min_frequency: int = 3
    ) -> Dict:
        """
        Compare user skills against professionals in the target role.

        Args:
            user_skills: User's extracted skills:
                {
                    'programming_languages': ['Python', 'JavaScript', ...],
                    'frameworks_libraries': ['React', 'Django', ...],
                    ...
                }
            role_name: Target job role
            min_frequency: Minimum frequency for role skills to be considered important

        Returns:
            Comprehensive comparison result:
            {
                'role': role_name,
                'match_percentage': 75.5,
                'matched_skills': {...},
                'missing_skills': {...},
                'extra_skills': {...},
                'role_metadata': {...},
                'summary': {...}
            }
        """
        # Get aggregated skills for the target role
        role_skills = self.get_role_skills(role_name, min_frequency=min_frequency)

        # Normalize user skills
        normalized_user_skills = {}
        for category in self.skill_categories:
            if category in user_skills:
                normalized_user_skills[category] = set(
                    self.normalize_skill(skill)
                    for skill in user_skills[category]
                    if skill
                )
            else:
                normalized_user_skills[category] = set()

        # Create a flat set of ALL user skills (across all categories) for cross-category matching
        all_user_skills = set()
        for skills_set in normalized_user_skills.values():
            all_user_skills.update(skills_set)

        # Compare skills in each category with CROSS-CATEGORY MATCHING
        matched_skills = defaultdict(list)
        missing_skills = defaultdict(list)
        extra_skills = defaultdict(list)
        already_matched = set()  # Track skills we've already matched to avoid duplicates

        for category in self.skill_categories:
            role_skill_set = set(role_skills.get(category, {}).keys())
            user_skill_set = normalized_user_skills.get(category, set())

            # Find matches - check if skill is in user's skills (ANY category, not just matching category)
            for role_skill in role_skill_set:
                if role_skill in all_user_skills and role_skill not in already_matched:
                    # Skill found! (even if it's in a different category in user's resume)
                    frequency = role_skills[category].get(role_skill, 0)
                    matched_skills[category].append({
                        'skill': role_skill,
                        'frequency': frequency,
                        'percentage': round((frequency / role_skills['_metadata']['total_professionals']) * 100, 1)
                    })
                    already_matched.add(role_skill)
                elif role_skill not in all_user_skills:
                    # Truly missing
                    frequency = role_skills[category].get(role_skill, 0)
                    missing_skills[category].append({
                        'skill': role_skill,
                        'frequency': frequency,
                        'percentage': round((frequency / role_skills['_metadata']['total_professionals']) * 100, 1)
                    })

            # Find extra skills (in user but not in role at all)
            for user_skill in user_skill_set:
                # Check if this skill appears in ANY role category
                found_in_role = any(user_skill in role_skills.get(cat, {}) for cat in self.skill_categories)
                if not found_in_role and user_skill not in already_matched:
                    extra_skills[category].append({'skill': user_skill})

        # Sort missing skills by frequency (most important first)
        for category in missing_skills:
            missing_skills[category] = sorted(
                missing_skills[category],
                key=lambda x: x['frequency'],
                reverse=True
            )

        # Calculate match percentage
        total_role_skills = sum(
            len(role_skills.get(cat, {}))
            for cat in self.skill_categories
        )
        total_matched_skills = sum(
            len(matched_skills.get(cat, []))
            for cat in self.skill_categories
        )

        match_percentage = (
            (total_matched_skills / total_role_skills * 100)
            if total_role_skills > 0
            else 0
        )

        # Generate summary statistics
        summary = {
            'total_role_skills': total_role_skills,
            'total_user_skills': sum(len(normalized_user_skills.get(cat, [])) for cat in self.skill_categories),
            'total_matched': total_matched_skills,
            'total_missing': sum(len(missing_skills.get(cat, [])) for cat in self.skill_categories),
            'total_extra': sum(len(extra_skills.get(cat, [])) for cat in self.skill_categories),
        }

        return {
            'role': role_name,
            'match_percentage': round(match_percentage, 2),
            'matched_skills': dict(matched_skills),
            'missing_skills': dict(missing_skills),
            'extra_skills': dict(extra_skills),
            'role_metadata': role_skills['_metadata'],
            'summary': summary
        }

    def get_learning_resources(self, missing_skills: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Generate learning resource recommendations for missing skills.

        Args:
            missing_skills: Missing skills from compare_skills() result

        Returns:
            List of learning resources:
            [
                {
                    'skill': 'react',
                    'category': 'frameworks_libraries',
                    'importance': 'high',
                    'resources': [...]
                },
                ...
            ]
        """
        # Learning platform mappings for common skills
        resource_map = {
            # Programming Languages
            'python': [
                {'platform': 'Coursera', 'course': 'Python for Everybody', 'url': 'https://www.coursera.org/specializations/python'},
                {'platform': 'Codecademy', 'course': 'Learn Python 3', 'url': 'https://www.codecademy.com/learn/learn-python-3'},
                {'platform': 'freeCodeCamp', 'course': 'Python Tutorials', 'url': 'https://www.freecodecamp.org/news/tag/python/'},
            ],
            'javascript': [
                {'platform': 'freeCodeCamp', 'course': 'JavaScript Algorithms and Data Structures', 'url': 'https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/'},
                {'platform': 'Codecademy', 'course': 'Learn JavaScript', 'url': 'https://www.codecademy.com/learn/introduction-to-javascript'},
                {'platform': 'MDN', 'course': 'JavaScript Guide', 'url': 'https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide'},
            ],
            'java': [
                {'platform': 'Coursera', 'course': 'Java Programming and Software Engineering', 'url': 'https://www.coursera.org/specializations/java-programming'},
                {'platform': 'Codecademy', 'course': 'Learn Java', 'url': 'https://www.codecademy.com/learn/learn-java'},
            ],
            'typescript': [
                {'platform': 'Official Docs', 'course': 'TypeScript Handbook', 'url': 'https://www.typescriptlang.org/docs/handbook/intro.html'},
                {'platform': 'Udemy', 'course': 'Understanding TypeScript', 'url': 'https://www.udemy.com/course/understanding-typescript/'},
            ],

            # Frameworks
            'react': [
                {'platform': 'Official Docs', 'course': 'React Tutorial', 'url': 'https://react.dev/learn'},
                {'platform': 'Scrimba', 'course': 'Learn React for Free', 'url': 'https://scrimba.com/learn/learnreact'},
                {'platform': 'freeCodeCamp', 'course': 'React Course', 'url': 'https://www.freecodecamp.org/news/tag/react/'},
            ],
            'angular': [
                {'platform': 'Official Docs', 'course': 'Angular Tutorial', 'url': 'https://angular.io/tutorial'},
                {'platform': 'Udemy', 'course': 'Angular - The Complete Guide', 'url': 'https://www.udemy.com/course/the-complete-guide-to-angular-2/'},
            ],
            'vue': [
                {'platform': 'Official Docs', 'course': 'Vue.js Guide', 'url': 'https://vuejs.org/guide/introduction.html'},
                {'platform': 'Vue Mastery', 'course': 'Intro to Vue 3', 'url': 'https://www.vuemastery.com/courses/intro-to-vue-3/'},
            ],
            'django': [
                {'platform': 'Official Docs', 'course': 'Django Tutorial', 'url': 'https://docs.djangoproject.com/en/stable/intro/tutorial01/'},
                {'platform': 'Coursera', 'course': 'Django for Everybody', 'url': 'https://www.coursera.org/specializations/django'},
            ],
            'flask': [
                {'platform': 'Official Docs', 'course': 'Flask Tutorial', 'url': 'https://flask.palletsprojects.com/en/stable/tutorial/'},
                {'platform': 'freeCodeCamp', 'course': 'Flask Course', 'url': 'https://www.freecodecamp.org/news/tag/flask/'},
            ],
            'nodejs': [
                {'platform': 'Official Docs', 'course': 'Node.js Guides', 'url': 'https://nodejs.org/en/docs/guides/'},
                {'platform': 'freeCodeCamp', 'course': 'Node.js and Express', 'url': 'https://www.freecodecamp.org/news/tag/nodejs/'},
            ],
            'express': [
                {'platform': 'Official Docs', 'course': 'Express Guide', 'url': 'https://expressjs.com/en/starter/guide.html'},
                {'platform': 'Codecademy', 'course': 'Learn Express', 'url': 'https://www.codecademy.com/learn/learn-express'},
            ],

            # Databases
            'sql': [
                {'platform': 'Codecademy', 'course': 'Learn SQL', 'url': 'https://www.codecademy.com/learn/learn-sql'},
                {'platform': 'Khan Academy', 'course': 'SQL Basics', 'url': 'https://www.khanacademy.org/computing/computer-programming/sql'},
            ],
            'mongodb': [
                {'platform': 'MongoDB University', 'course': 'MongoDB Basics', 'url': 'https://university.mongodb.com/'},
                {'platform': 'freeCodeCamp', 'course': 'MongoDB Course', 'url': 'https://www.freecodecamp.org/news/tag/mongodb/'},
            ],
            'postgres': [
                {'platform': 'Official Docs', 'course': 'PostgreSQL Tutorial', 'url': 'https://www.postgresql.org/docs/current/tutorial.html'},
                {'platform': 'Codecademy', 'course': 'Learn PostgreSQL', 'url': 'https://www.codecademy.com/learn/learn-postgresql'},
            ],
            'mysql': [
                {'platform': 'MySQL Tutorial', 'course': 'MySQL for Beginners', 'url': 'https://dev.mysql.com/doc/mysql-tutorial-excerpt/8.0/en/'},
                {'platform': 'Codecademy', 'course': 'Learn MySQL', 'url': 'https://www.codecademy.com/learn/learn-mysql'},
            ],

            # Tools
            'git': [
                {'platform': 'Git Official', 'course': 'Pro Git Book', 'url': 'https://git-scm.com/book/en/v2'},
                {'platform': 'GitHub', 'course': 'Git Handbook', 'url': 'https://guides.github.com/introduction/git-handbook/'},
            ],
            'docker': [
                {'platform': 'Docker Docs', 'course': 'Get Started with Docker', 'url': 'https://docs.docker.com/get-started/'},
                {'platform': 'freeCodeCamp', 'course': 'Docker Tutorial', 'url': 'https://www.freecodecamp.org/news/tag/docker/'},
            ],
            'aws': [
                {'platform': 'AWS Training', 'course': 'AWS Cloud Practitioner', 'url': 'https://aws.amazon.com/training/'},
                {'platform': 'Coursera', 'course': 'AWS Fundamentals', 'url': 'https://www.coursera.org/specializations/aws-fundamentals'},
            ],
            'kubernetes': [
                {'platform': 'Kubernetes Docs', 'course': 'Kubernetes Basics', 'url': 'https://kubernetes.io/docs/tutorials/kubernetes-basics/'},
                {'platform': 'Udemy', 'course': 'Kubernetes for Beginners', 'url': 'https://www.udemy.com/course/learn-kubernetes/'},
            ],
        }

        # Generic resources for categories
        generic_resources = {
            'programming_languages': [
                {'platform': 'Codecademy', 'course': 'Programming Courses', 'url': 'https://www.codecademy.com/catalog/language/python'},
                {'platform': 'Coursera', 'course': 'Programming Specializations', 'url': 'https://www.coursera.org/browse/computer-science'},
            ],
            'frameworks_libraries': [
                {'platform': 'Official Documentation', 'course': 'Framework Docs', 'url': 'https://www.google.com/search?q={skill}+documentation'},
                {'platform': 'YouTube', 'course': 'Framework Tutorials', 'url': 'https://www.youtube.com/results?search_query={skill}+tutorial'},
            ],
            'databases': [
                {'platform': 'Codecademy', 'course': 'Database Courses', 'url': 'https://www.codecademy.com/catalog/subject/databases'},
                {'platform': 'Khan Academy', 'course': 'SQL and Databases', 'url': 'https://www.khanacademy.org/computing/computer-programming/sql'},
            ],
            'tools_software': [
                {'platform': 'Official Documentation', 'course': 'Tool Documentation', 'url': 'https://www.google.com/search?q={skill}+documentation'},
                {'platform': 'YouTube', 'course': 'Tool Tutorials', 'url': 'https://www.youtube.com/results?search_query={skill}+tutorial'},
            ],
        }

        recommendations = []

        # Process each category of missing skills
        for category, skills in missing_skills.items():
            for skill_info in skills:
                skill = skill_info['skill']
                frequency = skill_info['frequency']
                percentage = skill_info['percentage']

                # Determine importance based on how many professionals have this skill
                if percentage >= 50:
                    importance = 'critical'
                elif percentage >= 30:
                    importance = 'high'
                elif percentage >= 15:
                    importance = 'medium'
                else:
                    importance = 'low'

                # Get resources for this skill
                resources = resource_map.get(skill, [])

                # If no specific resources, use generic ones
                if not resources and category in generic_resources:
                    resources = [
                        {
                            'platform': res['platform'],
                            'course': res['course'],
                            'url': res['url'].replace('{skill}', skill)
                        }
                        for res in generic_resources[category]
                    ]

                recommendations.append({
                    'skill': skill,
                    'category': category,
                    'importance': importance,
                    'possessed_by_percentage': percentage,
                    'resources': resources[:3]  # Limit to top 3 resources
                })

        # Sort by importance (critical > high > medium > low)
        importance_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: importance_order[x['importance']])

        return recommendations

    def get_available_roles(self) -> List[Dict[str, any]]:
        """
        Get all available roles from the database with counts.

        Returns:
            List of roles with metadata:
            [
                {'role': 'Frontend Developer', 'count': 54},
                {'role': 'Backend Developer', 'count': 57},
                ...
            ]
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT category, COUNT(*) as count
            FROM persons
            GROUP BY category
            ORDER BY count DESC
        """)

        roles = [
            {'role': row[0], 'professional_count': row[1]}
            for row in cursor.fetchall()
        ]

        conn.close()
        return roles


def main():
    """Demo usage of SkillMatcher"""
    print("<� SKILL MATCHER - Demo")
    print("=" * 60)

    matcher = SkillMatcher('nlp_resume_data.db')

    # Example 1: Get available roles
    print("\n=� Available Roles:")
    roles = matcher.get_available_roles()
    for role_info in roles:
        print(f"   - {role_info['role']}: {role_info['professional_count']} professionals")

    # Example 2: Get top skills for a role
    print("\n\n<� Top Skills for 'Frontend Developer':")
    top_skills = matcher.get_top_skills_by_role('Frontend Developer', top_n=5)
    for category, skills in top_skills.items():
        if skills:
            print(f"\n   {category.replace('_', ' ').title()}:")
            for skill, count in skills:
                print(f"      - {skill}: {count} professionals")

    # Example 3: Compare user skills
    print("\n\n=== Skill Gap Analysis:")
    print("-" * 60)

    # Simulated user skills (would come from resume parsing)
    user_skills = {
        'programming_languages': ['HTML', 'CSS', 'JavaScript'],
        'frameworks_libraries': ['React'],
        'tools_software': ['Git'],
        'databases': [],
        'soft_skills': ['Team Collaboration', 'Communication']
    }

    print(f"\nUser Skills: {sum(len(skills) for skills in user_skills.values())} total")
    print("Target Role: Frontend Developer")

    comparison = matcher.compare_skills(user_skills, 'Frontend Developer', min_frequency=3)

    print(f"\n( Match Percentage: {comparison['match_percentage']}%")
    print(f"   Professionals analyzed: {comparison['role_metadata']['total_professionals']}")
    print(f"   Skills matched: {comparison['summary']['total_matched']}")
    print(f"   Skills missing: {comparison['summary']['total_missing']}")

    # Show top missing skills
    print("\nL Top Missing Skills:")
    for category in ['programming_languages', 'frameworks_libraries', 'tools_software']:
        missing = comparison['missing_skills'].get(category, [])
        if missing:
            print(f"\n   {category.replace('_', ' ').title()}:")
            for skill_info in missing[:5]:
                print(f"      - {skill_info['skill']}: {skill_info['percentage']}% of professionals have this")

    # Get learning resources
    print("\n\n=� Learning Recommendations:")
    print("-" * 60)

    resources = matcher.get_learning_resources(comparison['missing_skills'])

    # Show top 5 most critical skills to learn
    for rec in resources[:5]:
        print(f"\n<� {rec['skill'].upper()} ({rec['importance']} priority)")
        print(f"   Category: {rec['category'].replace('_', ' ')}")
        print(f"   {rec['possessed_by_percentage']}% of {comparison['role']}s have this skill")
        print(f"   Resources:")
        for resource in rec['resources']:
            print(f"      - {resource['platform']}: {resource['course']}")

    print("\n" + "=" * 60)
    print(" Skill matching demo complete!")


if __name__ == "__main__":
    main()

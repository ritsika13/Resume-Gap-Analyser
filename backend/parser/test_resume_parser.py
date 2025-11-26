#!/usr/bin/env python3
"""
Test script for ResumeParser - validates parsing accuracy with example resumes
"""

import os
import sys
import json
from typing import Dict, List
import sqlite3
import pandas as pd

# Add the parent directory to the path so we can import resume_parser
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from resume_parser import ResumeParser

class ResumeParserTester:
    """Test class for validating ResumeParser accuracy"""
    
    def __init__(self):
        self.parser = ResumeParser()
        self.test_results = []
        
        # Expected results for validation (based on our test resumes)
        self.expected_results = {
            'software_developer.txt': {
                'category': 'Software Developer',
                'expected_languages': ['JavaScript', 'Python', 'Java', 'TypeScript'],
                'expected_frameworks': ['React', 'Django', 'Angular', 'Spring'],
                'expected_databases': ['PostgreSQL', 'MySQL', 'MongoDB', 'Redis'],
                'expected_tools': ['Git', 'Docker', 'AWS', 'Jenkins'],
                'expected_years': 6,
                'min_skills_total': 15
            },
            'data_scientist.txt': {
                'category': 'Data Scientist', 
                'expected_languages': ['Python', 'R', 'SQL'],
                'expected_frameworks': ['TensorFlow', 'PyTorch', 'Scikit'],
                'expected_databases': ['PostgreSQL', 'MongoDB'],
                'expected_tools': ['AWS', 'Tableau', 'Hadoop', 'Spark'],
                'expected_years': 8,
                'min_skills_total': 20
            },
            'devops_engineer.txt': {
                'category': 'DevOps Engineer',
                'expected_languages': ['Python', 'Bash'],
                'expected_frameworks': [],
                'expected_databases': [],
                'expected_tools': ['Docker', 'Kubernetes', 'AWS', 'Jenkins', 'Terraform', 'Ansible'],
                'expected_years': 7,
                'min_skills_total': 15
            },
            'ui_ux_designer.txt': {
                'category': 'UI/UX Designer',
                'expected_languages': ['HTML', 'CSS', 'JavaScript'],
                'expected_frameworks': ['React'],
                'expected_databases': [],
                'expected_tools': ['Figma', 'Sketch', 'Photoshop'],
                'expected_years': 5,
                'min_skills_total': 10
            },
            'product_manager.txt': {
                'category': 'Product Manager',
                'expected_languages': ['SQL'],
                'expected_frameworks': [],
                'expected_databases': [],
                'expected_tools': ['Jira', 'Google Analytics', 'Tableau'],
                'expected_years': 9,
                'min_skills_total': 8
            }
        }

    def run_all_tests(self):
        """Run all tests and generate comprehensive report"""
        print("🧪 Starting Resume Parser Testing")
        print("=" * 60)
        
        # Test directory path
        test_resumes_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'test_resumes'
        )
        
        if not os.path.exists(test_resumes_dir):
            print(f"❌ Test resumes directory not found: {test_resumes_dir}")
            return
        
        # Get all test resume files
        resume_files = [f for f in os.listdir(test_resumes_dir) if f.endswith('.txt')]
        
        if not resume_files:
            print(f"❌ No test resume files found in: {test_resumes_dir}")
            return
        
        print(f"📁 Found {len(resume_files)} test resume files")
        
        all_parsed_resumes = []
        
        # Parse each resume file
        for i, filename in enumerate(resume_files, 1):
            file_path = os.path.join(test_resumes_dir, filename)
            print(f"\n📄 Testing {i}/{len(resume_files)}: {filename}")
            
            # Parse resume
            parsed_result = self.parser.parse_resume_file(
                file_path=file_path,
                resume_id=i,
                category=self.expected_results.get(filename, {}).get('category', 'Unknown')
            )
            
            if parsed_result:
                all_parsed_resumes.append(parsed_result)
                # Validate results
                self.validate_parsing_results(filename, parsed_result)
            else:
                print(f"❌ Failed to parse {filename}")
        
        # Generate comprehensive report
        self.generate_test_report(all_parsed_resumes)
        
        # Create database and analytics
        if all_parsed_resumes:
            self.create_test_database(all_parsed_resumes)

    def validate_parsing_results(self, filename: str, parsed_result: Dict):
        """Validate parsing results against expected values"""
        print(f"   🔍 Validating results for {filename}")
        
        expected = self.expected_results.get(filename, {})
        if not expected:
            print(f"   ⚠️  No expected results defined for {filename}")
            return
        
        validation_results = {
            'filename': filename,
            'category': parsed_result['category'],
            'tests': {}
        }
        
        # Test 1: Years of experience
        expected_years = expected.get('expected_years')
        actual_years = parsed_result['years_experience']
        if expected_years and actual_years:
            years_match = abs(actual_years - expected_years) <= 1  # Allow 1 year difference
            validation_results['tests']['years_experience'] = {
                'expected': expected_years,
                'actual': actual_years,
                'passed': years_match
            }
            print(f"   {'✅' if years_match else '❌'} Years: Expected {expected_years}, Got {actual_years}")
        
        # Test 2: Programming languages
        expected_langs = expected.get('expected_languages', [])
        actual_langs = parsed_result['programming_languages']
        lang_matches = sum(1 for lang in expected_langs if any(lang.lower() in actual.lower() for actual in actual_langs))
        lang_score = lang_matches / len(expected_langs) if expected_langs else 1
        validation_results['tests']['programming_languages'] = {
            'expected': expected_langs,
            'actual': actual_langs,
            'matches': lang_matches,
            'score': lang_score,
            'passed': lang_score >= 0.5
        }
        print(f"   {'✅' if lang_score >= 0.5 else '❌'} Languages: {lang_matches}/{len(expected_langs)} matched ({lang_score:.1%})")
        
        # Test 3: Frameworks/Libraries
        expected_frameworks = expected.get('expected_frameworks', [])
        actual_frameworks = parsed_result['frameworks_libraries']
        if expected_frameworks:
            framework_matches = sum(1 for fw in expected_frameworks if any(fw.lower() in actual.lower() for actual in actual_frameworks))
            framework_score = framework_matches / len(expected_frameworks)
            validation_results['tests']['frameworks_libraries'] = {
                'expected': expected_frameworks,
                'actual': actual_frameworks,
                'matches': framework_matches,
                'score': framework_score,
                'passed': framework_score >= 0.4
            }
            print(f"   {'✅' if framework_score >= 0.4 else '❌'} Frameworks: {framework_matches}/{len(expected_frameworks)} matched ({framework_score:.1%})")
        
        # Test 4: Tools/Software
        expected_tools = expected.get('expected_tools', [])
        actual_tools = parsed_result['tools_software']
        if expected_tools:
            tool_matches = sum(1 for tool in expected_tools if any(tool.lower() in actual.lower() for actual in actual_tools))
            tool_score = tool_matches / len(expected_tools)
            validation_results['tests']['tools_software'] = {
                'expected': expected_tools,
                'actual': actual_tools,
                'matches': tool_matches,
                'score': tool_score,
                'passed': tool_score >= 0.3
            }
            print(f"   {'✅' if tool_score >= 0.3 else '❌'} Tools: {tool_matches}/{len(expected_tools)} matched ({tool_score:.1%})")
        
        # Test 5: Total skills extracted
        total_skills = len(parsed_result['raw_extracted_terms'])
        min_expected = expected.get('min_skills_total', 5)
        skills_adequate = total_skills >= min_expected
        validation_results['tests']['total_skills'] = {
            'expected_min': min_expected,
            'actual': total_skills,
            'passed': skills_adequate
        }
        print(f"   {'✅' if skills_adequate else '❌'} Total Skills: {total_skills} (min {min_expected})")
        
        # Test 6: Soft skills
        soft_skills = parsed_result['soft_skills']
        has_soft_skills = len(soft_skills) > 0
        validation_results['tests']['soft_skills'] = {
            'actual': soft_skills,
            'count': len(soft_skills),
            'passed': has_soft_skills
        }
        print(f"   {'✅' if has_soft_skills else '⚠️ '} Soft Skills: {len(soft_skills)} found")
        
        self.test_results.append(validation_results)

    def generate_test_report(self, parsed_resumes: List[Dict]):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        if not self.test_results:
            print("No test results to report")
            return
        
        # Overall statistics
        total_tests = len(self.test_results)
        passed_tests = 0
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total Resumes Tested: {total_tests}")
        
        # Category breakdown
        category_stats = {}
        for result in self.test_results:
            category = result['category']
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'passed': 0}
            category_stats[category]['total'] += 1
            
            # Count passed tests for this resume
            test_scores = []
            for test_name, test_result in result['tests'].items():
                if test_result.get('passed', False):
                    test_scores.append(1)
                else:
                    test_scores.append(0)
            
            if test_scores and sum(test_scores) / len(test_scores) >= 0.6:  # 60% pass rate
                category_stats[category]['passed'] += 1
                passed_tests += 1
        
        print(f"   Overall Pass Rate: {passed_tests}/{total_tests} ({passed_tests/total_tests:.1%})")
        
        # Detailed results by category
        print(f"\n📋 Results by Category:")
        for category, stats in category_stats.items():
            pass_rate = stats['passed'] / stats['total']
            print(f"   {category}: {stats['passed']}/{stats['total']} passed ({pass_rate:.1%})")
        
        # Skill extraction statistics
        print(f"\n🔍 Skill Extraction Analysis:")
        
        all_programming_languages = []
        all_frameworks = []
        all_tools = []
        all_databases = []
        all_soft_skills = []
        
        for resume in parsed_resumes:
            all_programming_languages.extend(resume['programming_languages'])
            all_frameworks.extend(resume['frameworks_libraries'])
            all_tools.extend(resume['tools_software'])
            all_databases.extend(resume['databases'])
            all_soft_skills.extend(resume['soft_skills'])
        
        print(f"   Programming Languages Found: {len(set(all_programming_languages))}")
        if all_programming_languages:
            top_langs = pd.Series(all_programming_languages).value_counts().head(5)
            print(f"   Top Languages: {list(top_langs.index)}")
        
        print(f"   Frameworks/Libraries Found: {len(set(all_frameworks))}")
        if all_frameworks:
            top_frameworks = pd.Series(all_frameworks).value_counts().head(5)
            print(f"   Top Frameworks: {list(top_frameworks.index)}")
        
        print(f"   Tools/Software Found: {len(set(all_tools))}")
        if all_tools:
            top_tools = pd.Series(all_tools).value_counts().head(5)
            print(f"   Top Tools: {list(top_tools.index)}")
        
        print(f"   Databases Found: {len(set(all_databases))}")
        print(f"   Soft Skills Found: {len(set(all_soft_skills))}")
        
        # Years of experience analysis
        years_data = [r['years_experience'] for r in parsed_resumes if r['years_experience'] is not None]
        if years_data:
            avg_years = sum(years_data) / len(years_data)
            print(f"   Average Years Experience: {avg_years:.1f}")
            print(f"   Experience Range: {min(years_data)} - {max(years_data)} years")
        
        # Individual test details
        print(f"\n📝 Detailed Test Results:")
        for result in self.test_results:
            filename = result['filename']
            category = result['category']
            tests = result['tests']
            
            print(f"\n   📄 {filename} ({category}):")
            for test_name, test_data in tests.items():
                status = "✅" if test_data.get('passed', False) else "❌"
                if 'score' in test_data:
                    print(f"      {status} {test_name}: {test_data['score']:.1%} ({test_data['matches']}/{len(test_data['expected'])})")
                elif 'actual' in test_data and 'expected' in test_data:
                    print(f"      {status} {test_name}: Expected {test_data['expected']}, Got {test_data['actual']}")
                else:
                    print(f"      {status} {test_name}: {test_data.get('count', 'N/A')}")

    def create_test_database(self, parsed_resumes: List[Dict]):
        """Create database with test results"""
        print(f"\n💾 Creating test database...")
        
        # Create tables
        tables = self.parser.create_normalized_tables(parsed_resumes)
        
        # Save to database
        db_file = 'test_resume_parsing_results.db'
        self.parser.create_sqlite_database(tables, db_file)
        
        # Generate analytics
        analytics = self.parser.generate_skill_analytics(tables)
        
        print(f"\n📊 Database Analytics:")
        for skill_type, stats in analytics.items():
            if stats['top_10']:
                print(f"   {skill_type.replace('_', ' ').title()}:")
                print(f"     Total: {stats['total_mentions']} mentions")
                print(f"     Unique: {stats['unique_skills']} skills")
                top_3 = [f"{skill}({count})" for skill, count in stats['top_10'][:3]]
                print(f"     Top 3: {', '.join(top_3)}")
        
        # Save test results to JSON
        results_file = 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'test_results': self.test_results,
                'analytics': analytics,
                'summary': {
                    'total_resumes': len(parsed_resumes),
                    'total_skills_extracted': sum(len(r['raw_extracted_terms']) for r in parsed_resumes),
                    'categories_tested': list(set(r['category'] for r in parsed_resumes))
                }
            }, f, indent=2)
        
        print(f"   Test results saved to: {results_file}")
        print(f"   Database saved to: {db_file}")

    def run_interactive_test(self, resume_text: str):
        """Run interactive test with custom resume text"""
        print("\n🔬 Interactive Resume Test")
        print("-" * 40)
        
        parsed_result = self.parser.parse_resume_text(resume_text, resume_id=999, category="Interactive Test")
        
        print(f"📊 Parsing Results:")
        print(f"   Years of Experience: {parsed_result['years_experience']}")
        print(f"   Programming Languages ({len(parsed_result['programming_languages'])}): {parsed_result['programming_languages']}")
        print(f"   Frameworks/Libraries ({len(parsed_result['frameworks_libraries'])}): {parsed_result['frameworks_libraries']}")
        print(f"   Tools/Software ({len(parsed_result['tools_software'])}): {parsed_result['tools_software']}")
        print(f"   Databases ({len(parsed_result['databases'])}): {parsed_result['databases']}")
        print(f"   Soft Skills ({len(parsed_result['soft_skills'])}): {parsed_result['soft_skills']}")
        print(f"   Other Skills ({len(parsed_result['other_skills'])}): {parsed_result['other_skills']}")
        print(f"   Experience Descriptions: {len(parsed_result['experience_descriptions'])} found")
        print(f"   Total Terms Extracted: {len(parsed_result['raw_extracted_terms'])}")
        
        return parsed_result

def main():
    """Main test execution"""
    tester = ResumeParserTester()
    
    # Check if we want to run interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        sample_text = """
        Senior Python Developer with 5 years of experience.
        Skilled in Django, Flask, PostgreSQL, and AWS.
        Experience with machine learning using TensorFlow and scikit-learn.
        Strong problem-solving skills and team leadership experience.
        """
        tester.run_interactive_test(sample_text)
    else:
        # Run full test suite
        tester.run_all_tests()
        
        print(f"\n✅ Testing completed successfully!")
        print(f"💡 Run with --interactive flag to test custom resume text")

if __name__ == "__main__":
    main()
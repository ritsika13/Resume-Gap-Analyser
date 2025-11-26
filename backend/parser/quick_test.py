# quick_csv_test.py - Quick test to see how NLP parser works on your CSV

import pandas as pd

def quick_test_csv_parsing():
    """Quick test to see NLP parsing in action"""
    print("🔬 QUICK CSV PARSING TEST")
    print("=" * 50)
    
    # Load your CSV
    try:
        df = pd.read_csv('csdataset.csv')
        print(f"✅ Loaded CSV: {len(df)} resumes found")
        print(f"Categories: {df['Category'].unique().tolist()}")
    except FileNotFoundError:
        print("❌ csdataset.csv not found in current directory")
        return
    
    # Import the NLP parser
    try:
        from csv_parser import NLPResumeParser
        parser = NLPResumeParser()
        print("✅ NLP Parser loaded successfully")
    except ImportError as e:
        print(f"❌ Cannot import parser: {e}")
        print("Make sure nlp_parser_no_spacy.py is in the current directory")
        return
    
    first_resume = df.iloc[0]
    resume_text = first_resume['Resume']
    category = first_resume['Category']
    
    print(f"\n📄 TESTING ON FIRST RESUME:")
    print(f"Category: {category}")
    print(f"Resume preview: {resume_text[:200]}...")
    
    print(f"\n🧠 NLP EXTRACTION:")
    print("-" * 30)
    
    # Extract years
    years = parser.extract_years_experience(resume_text)
    print(f"Years of experience: {years}")
    
    # Extract and classify skills
    skills = parser.extract_entities_and_skills(resume_text)
    
    print(f"\nSkills found by category:")
    for skill_type, skill_list in skills.items():
        if skill_list:
            print(f"  {skill_type.replace('_', ' ').title()}: {skill_list}")
    
    # Full parse
    parsed = parser.parse_single_resume(resume_text, category, 1)
    
    print(f"\n📊 SUMMARY:")
    print(f"Total skills extracted: {len(parsed['raw_extracted_terms'])}")
    print(f"Programming languages: {len(parsed['programming_languages'])}")
    print(f"Frameworks/libraries: {len(parsed['frameworks_libraries'])}")
    print(f"Tools/software: {len(parsed['tools_software'])}")
    print(f"Databases: {len(parsed['databases'])}")
    
    # Test a few more resumes
    print(f"\n🔄 TESTING 5 MORE RESUMES:")
    print("-" * 30)
    
    for i in range(1, 6):
        if i < len(df):
            resume = df.iloc[i]
            parsed = parser.parse_single_resume(resume['Resume'], resume['Category'], i+1)
            
            total_skills = len(parsed['raw_extracted_terms'])
            years = parsed['years_experience'] or 'Unknown'
            
            print(f"Resume {i+1} ({resume['Category']}): {total_skills} skills, {years} years exp")
            print(f"  Languages: {parsed['programming_languages'][:3]}")
            print(f"  Frameworks: {parsed['frameworks_libraries'][:3]}")
    
    print(f"\n✅ Quick test complete!")
    print(f"The parser successfully extracted skills from your resume data.")
    print(f"\nTo run full processing: python nlp_parser_no_spacy.py")
    print(f"For detailed testing: python test_nlp_on_csv.py")

if __name__ == "__main__":
    quick_test_csv_parsing()
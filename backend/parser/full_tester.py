# test_nlp_on_csv.py - Test the NLP parser on your CSV dataset

import pandas as pd
import json
import time
from datetime import datetime

def test_single_resume_parsing():
    """Test parsing on individual resumes to see detailed extraction"""
    print("🧪 TESTING SINGLE RESUME PARSING")
    print("=" * 60)
    
    # Load the dataset
    try:
        df = pd.read_csv('csdataset.csv')
        print(f"✅ Dataset loaded: {len(df)} resumes")
    except FileNotFoundError:
        print("❌ csdataset.csv not found. Make sure it's in the current directory.")
        return False
    
    # Import the parser
    try:
        from csv_parser import NLPResumeParser
        parser = NLPResumeParser()
        print("✅ NLP Parser initialized")
    except ImportError as e:
        print(f"❌ Cannot import NLP parser: {e}")
        print("Make sure nlp_parser_no_spacy.py is in the current directory")
        return False
    
    # Test on first 5 resumes from different categories
    test_indices = [0, 1, 2, 3, 4]  # First 5 resumes
    
    for idx in test_indices:
        if idx >= len(df):
            continue
            
        row = df.iloc[idx]
        resume_text = row['Resume']
        category = row['Category']
        
        print(f"\n{'='*60}")
        print(f"📄 TESTING RESUME {idx + 1}: {category}")
        print(f"{'='*60}")
        
        # Show original resume content (first 300 chars)
        print(f"\n📝 Original Resume Content:")
        print(f"{resume_text[:300]}...")
        
        print(f"\n🔍 NLP EXTRACTION RESULTS:")
        print("-" * 40)
        
        # Test individual extraction methods
        
        # 1. Years of experience
        years = parser.extract_years_experience(resume_text)
        print(f"⏰ Years of Experience: {years if years else 'Not found'}")
        
        # 2. Named entities
        entities = parser.extract_named_entities(resume_text)
        print(f"🏢 Named Entities: {entities[:5] if entities else 'None found'}")
        if len(entities) > 5:
            print(f"    ... and {len(entities) - 5} more")
        
        # 3. Technical terms
        tech_terms = parser.extract_technical_terms_nlp(resume_text)
        print(f"🔧 Technical Terms: {tech_terms[:5] if tech_terms else 'None found'}")
        if len(tech_terms) > 5:
            print(f"    ... and {len(tech_terms) - 5} more")
        
        # 4. TF-IDF terms
        tfidf_terms = parser.extract_skills_with_tfidf(resume_text)
        print(f"📊 TF-IDF Important Terms: {tfidf_terms[:5] if tfidf_terms else 'None found'}")
        if len(tfidf_terms) > 5:
            print(f"    ... and {len(tfidf_terms) - 5} more")
        
        # 5. Full skill extraction and classification
        skills = parser.extract_entities_and_skills(resume_text)
        print(f"\n🎯 CLASSIFIED SKILLS:")
        for skill_type, skill_list in skills.items():
            if skill_list:
                print(f"  {skill_type.replace('_', ' ').title()}: {skill_list}")
        
        # 6. Experience descriptions
        experience = parser.extract_experience_descriptions(resume_text)
        print(f"\n💼 Experience Descriptions:")
        for i, desc in enumerate(experience[:3], 1):
            print(f"  {i}. {desc}")
        if len(experience) > 3:
            print(f"  ... and {len(experience) - 3} more")
        
        # 7. Full parsing result
        print(f"\n📋 COMPLETE PARSING RESULT:")
        parsed = parser.parse_single_resume(resume_text, category, idx + 1)
        
        # Show summary
        total_skills = len(parsed['raw_extracted_terms'])
        print(f"  Resume ID: {parsed['id']}")
        print(f"  Category: {parsed['category']}")
        print(f"  Years Experience: {parsed['years_experience']}")
        print(f"  Total Skills Extracted: {total_skills}")
        print(f"  Programming Languages: {len(parsed['programming_languages'])}")
        print(f"  Frameworks/Libraries: {len(parsed['frameworks_libraries'])}")
        print(f"  Tools/Software: {len(parsed['tools_software'])}")
        print(f"  Databases: {len(parsed['databases'])}")
        print(f"  Soft Skills: {len(parsed['soft_skills'])}")
        
        # Wait for user input to continue (except for first resume)
        if idx > 0:
            input(f"\nPress Enter to continue to next resume...")
    
    return True

def test_batch_processing():
    """Test processing multiple resumes and show statistics"""
    print(f"\n🔄 TESTING BATCH PROCESSING")
    print("=" * 60)
    
    try:
        from nlp_parser_no_spacy import NLPResumeParser
        parser = NLPResumeParser()
        
        # Test with first 20 resumes for speed
        df = pd.read_csv('csdataset.csv').head(20)
        print(f"📊 Testing batch processing on {len(df)} resumes...")
        
        start_time = time.time()
        
        # Create a temporary CSV file
        df.to_csv('test_batch.csv', index=False)
        
        # Process the batch
        parsed_resumes, tables = parser.load_and_parse_dataset('test_batch.csv')
        
        end_time = time.time()
        
        print(f"✅ Batch processing completed!")
        print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
        print(f"📈 Average time per resume: {(end_time - start_time) / len(df):.3f} seconds")
        
        # Show statistics
        print(f"\n📊 BATCH PROCESSING STATISTICS:")
        print("-" * 40)
        
        # Count skills by category
        skill_stats = {}
        total_skills_found = 0
        
        for resume in parsed_resumes:
            for skill_type in ['programming_languages', 'frameworks_libraries', 'tools_software', 'databases']:
                if skill_type not in skill_stats:
                    skill_stats[skill_type] = {'total': 0, 'unique': set()}
                
                skills = resume[skill_type]
                skill_stats[skill_type]['total'] += len(skills)
                skill_stats[skill_type]['unique'].update(skills)
                total_skills_found += len(skills)
        
        print(f"Total skills extracted: {total_skills_found}")
        print(f"Average skills per resume: {total_skills_found / len(parsed_resumes):.1f}")
        
        for skill_type, stats in skill_stats.items():
            unique_count = len(stats['unique'])
            print(f"{skill_type.replace('_', ' ').title()}: {stats['total']} total, {unique_count} unique")
        
        # Show most common skills
        print(f"\n🏆 MOST COMMON SKILLS FOUND:")
        from collections import Counter
        
        for skill_type in ['programming_languages', 'frameworks_libraries']:
            all_skills = []
            for resume in parsed_resumes:
                all_skills.extend(resume[skill_type])
            
            if all_skills:
                skill_counts = Counter(all_skills)
                top_5 = skill_counts.most_common(5)
                print(f"{skill_type.replace('_', ' ').title()}: {top_5}")
        
        # Show sample results
        print(f"\n📝 SAMPLE PARSING RESULTS:")
        for i, resume in enumerate(parsed_resumes[:3]):
            print(f"\nResume {i+1} ({resume['category']}):")
            print(f"  Years: {resume['years_experience']}")
            print(f"  Languages: {resume['programming_languages'][:3]}")
            print(f"  Frameworks: {resume['frameworks_libraries'][:3]}")
            print(f"  Tools: {resume['tools_software'][:3]}")
        
        # Clean up
        import os
        os.remove('test_batch.csv')
        
        return True
        
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_categories():
    """Test parsing on specific job categories to see differences"""
    print(f"\n🎯 TESTING BY JOB CATEGORIES")
    print("=" * 60)
    
    try:
        df = pd.read_csv('csdataset.csv')
        from nlp_parser_no_spacy import NLPResumeParser
        parser = NLPResumeParser()
        
        # Get one resume from each category
        categories = df['Category'].unique()
        
        category_results = {}
        
        for category in categories[:5]:  # Test first 5 categories
            category_resumes = df[df['Category'] == category]
            if len(category_resumes) > 0:
                # Take the first resume from this category
                sample_resume = category_resumes.iloc[0]
                
                print(f"\n📂 CATEGORY: {category}")
                print("-" * 40)
                
                # Parse the resume
                parsed = parser.parse_single_resume(
                    sample_resume['Resume'], 
                    category, 
                    1
                )
                
                # Store results
                category_results[category] = {
                    'years': parsed['years_experience'],
                    'total_skills': len(parsed['raw_extracted_terms']),
                    'languages': parsed['programming_languages'],
                    'frameworks': parsed['frameworks_libraries'],
                    'tools': parsed['tools_software'],
                    'databases': parsed['databases']
                }
                
                # Show results
                print(f"Years Experience: {parsed['years_experience']}")
                print(f"Total Skills: {len(parsed['raw_extracted_terms'])}")
                print(f"Programming Languages: {parsed['programming_languages']}")
                print(f"Frameworks/Libraries: {parsed['frameworks_libraries']}")
                print(f"Tools/Software: {parsed['tools_software'][:5]}...")
                print(f"Databases: {parsed['databases']}")
        
        # Compare categories
        print(f"\n📊 CATEGORY COMPARISON:")
        print("-" * 40)
        
        for category, results in category_results.items():
            print(f"{category}:")
            print(f"  Avg Skills: {results['total_skills']}")
            print(f"  Languages: {len(results['languages'])}")
            print(f"  Frameworks: {len(results['frameworks'])}")
            print(f"  Tools: {len(results['tools'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in category testing: {e}")
        return False

def save_detailed_results():
    """Process a few resumes and save detailed results to files"""
    print(f"\n💾 SAVING DETAILED RESULTS")
    print("=" * 60)
    
    try:
        df = pd.read_csv('csdataset.csv')
        from nlp_parser_no_spacy import NLPResumeParser
        parser = NLPResumeParser()
        
        # Process first 10 resumes
        test_resumes = df.head(10)
        results = []
        
        for idx, row in test_resumes.iterrows():
            print(f"Processing resume {idx + 1}/10...")
            
            parsed = parser.parse_single_resume(
                row['Resume'], 
                row['Category'], 
                idx + 1
            )
            
            # Add original text for reference
            parsed['original_text'] = row['Resume'][:500] + "..."  # First 500 chars
            
            results.append(parsed)
        
        # Save to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nlp_parsing_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Detailed results saved to: {filename}")
        
        # Create summary report
        summary_filename = f"nlp_summary_{timestamp}.txt"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write("NLP RESUME PARSING SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset: csdataset.csv\n")
            f.write(f"Processed: {len(results)} resumes\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Skills statistics
            all_languages = []
            all_frameworks = []
            all_tools = []
            
            for result in results:
                all_languages.extend(result['programming_languages'])
                all_frameworks.extend(result['frameworks_libraries'])
                all_tools.extend(result['tools_software'])
            
            from collections import Counter
            
            f.write("TOP SKILLS DISCOVERED:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Programming Languages: {Counter(all_languages).most_common(5)}\n")
            f.write(f"Frameworks: {Counter(all_frameworks).most_common(5)}\n")
            f.write(f"Tools: {Counter(all_tools).most_common(5)}\n\n")
            
            # Individual results
            for i, result in enumerate(results, 1):
                f.write(f"RESUME {i}: {result['category']}\n")
                f.write(f"Years: {result['years_experience']}\n")
                f.write(f"Skills: {len(result['raw_extracted_terms'])}\n")
                f.write(f"Languages: {result['programming_languages']}\n")
                f.write(f"Frameworks: {result['frameworks_libraries']}\n\n")
        
        print(f"✅ Summary report saved to: {summary_filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False

def main():
    """Run comprehensive testing"""
    print("🚀 NLP RESUME PARSER TESTING SUITE")
    print("Testing on your csdataset.csv file")
    print("=" * 60)
    
    print("This will test how the NLP parser extracts skills from your resume data.")
    print("The test includes:")
    print("1. Individual resume parsing (detailed view)")
    print("2. Batch processing (performance and statistics)")
    print("3. Category-specific analysis")
    print("4. Detailed results export")
    
    choice = input("\nChoose test type:\n1. Quick test (first 5 resumes)\n2. Full test suite\n3. Just batch processing\nEnter choice (1/2/3): ")
    
    if choice == "1":
        print("\n🎯 Running quick test...")
        test_single_resume_parsing()
        
    elif choice == "2":
        print("\n🎯 Running full test suite...")
        
        # Test 1: Individual parsing
        success = test_single_resume_parsing()
        if not success:
            return
        
        # Test 2: Batch processing
        input("\nPress Enter to continue to batch processing test...")
        test_batch_processing()
        
        # Test 3: Category analysis
        input("\nPress Enter to continue to category analysis...")
        test_specific_categories()
        
        # Test 4: Save results
        save_choice = input("\nSave detailed results to files? (y/n): ")
        if save_choice.lower() == 'y':
            save_detailed_results()
            
    elif choice == "3":
        print("\n🎯 Running batch processing test...")
        test_batch_processing()
        
    else:
        print("Invalid choice. Running quick test...")
        test_single_resume_parsing()
    
    print("\n" + "="*60)
    print("🎉 TESTING COMPLETE!")
    print("="*60)
    print("\nThe NLP parser has been tested on your CSV data.")
    print("You can now run the full parser with: python nlp_parser_no_spacy.py")

if __name__ == "__main__":
    main()
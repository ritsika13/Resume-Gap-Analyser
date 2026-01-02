"""
Skill Filter - Removes noise words and validates extracted skills
"""

# Common words that are NOT skills (blacklist)
NOISE_WORDS = {
    # Generic descriptors
    'proficient', 'experienced', 'expertise', 'knowledge', 'skills', 'skill',
    'strong', 'excellent', 'good', 'basic', 'advanced', 'intermediate',
    'working', 'solid', 'deep', 'extensive', 'proven', 'demonstrated',

    # Generic tech terms
    'programming', 'languages', 'language', 'framework', 'frameworks',
    'library', 'libraries', 'tool', 'tools', 'software', 'platform', 'platforms',
    'database', 'databases', 'system', 'systems', 'application', 'applications',

    # Action words
    'development', 'developed', 'developing', 'design', 'designed', 'designing',
    'implementation', 'implemented', 'implementing', 'building', 'built',
    'creating', 'created', 'create', 'using', 'used', 'use',

    # General words
    'machine', 'learning', 'data', 'web', 'mobile', 'cloud', 'big',
    'full', 'stack', 'front', 'back', 'end', 'side', 'client', 'server',
    'based', 'oriented', 'driven', 'focused', 'related', 'various',

    # Partial phrases
    'manipulation', 'analysis', 'visualization', 'processing', 'management',
    'distributed', 'scalable', 'efficient', 'robust', 'scalability',
    'models', 'model',  # Too generic

    # Time/quantity
    'years', 'year', 'months', 'experience', 'projects', 'project',

    # Academic
    'course', 'courses', 'coursework', 'relevant', 'including',

    # Generic adjectives
    'technical', 'analytical', 'statistical', 'computational',
    'data-driven',  # Adjective, not skill

    # Job titles (not skills)
    'scientist', 'engineer', 'developer', 'designer', 'analyst', 'architect'
}

# Known valid skills (whitelist - expand as needed)
VALID_SKILLS = {
    # Programming Languages
    'python', 'java', 'javascript', 'typescript', 'c', 'c++', 'c#', 'go', 'rust',
    'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'julia', 'perl',
    'bash', 'shell', 'powershell', 'sql', 'html', 'css', 'sass', 'less',
    'objective-c', 'dart', 'elixir', 'haskell', 'clojure', 'groovy',

    # Frameworks & Libraries
    'react', 'angular', 'vue', 'svelte', 'nextjs', 'next.js', 'nuxt', 'gatsby',
    'django', 'flask', 'fastapi', 'spring', 'springboot', 'express', 'nodejs',
    'node.js', 'laravel', 'rails', 'rubyonrails', 'asp.net', 'dotnet',
    'react-native', 'flutter', 'ionic', 'xamarin', 'cordova',

    # Data Science / ML Libraries
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'scikitlearn', 'sklearn',
    'pandas', 'numpy', 'scipy', 'statsmodels', 'seaborn', 'matplotlib', 'plotly',
    'opencv', 'nltk', 'spacy', 'huggingface', 'transformers', 'langchain', 'openai',

    # Frontend Libraries
    'redux', 'mobx', 'vuex', 'pinia', 'rxjs', 'jquery', 'bootstrap',
    'tailwind', 'materialui', 'chakra', 'ant-design', 'shadcn',

    # Databases
    'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'elasticsearch',
    'cassandra', 'dynamodb', 'firebase', 'firestore', 'supabase', 'oracle',
    'mssql', 'sqlite', 'mariadb', 'couchdb', 'neo4j', 'influxdb', 'timescaledb',

    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'heroku', 'vercel', 'netlify', 'digitalocean',
    'docker', 'kubernetes', 'k8s', 'jenkins', 'gitlab', 'github', 'bitbucket',
    'circleci', 'travis', 'ansible', 'terraform', 'cloudformation', 'pulumi',
    'nginx', 'apache', 'serverless', 'lambda', 'ec2', 's3', 'rds',

    # Tools
    'git', 'svn', 'mercurial', 'jira', 'confluence', 'slack', 'notion',
    'vscode', 'vim', 'emacs', 'intellij', 'pycharm', 'webstorm', 'eclipse',
    'postman', 'insomnia', 'swagger', 'graphql', 'rest', 'grpc',
    'webpack', 'vite', 'parcel', 'rollup', 'babel', 'eslint', 'prettier',
    'jest', 'mocha', 'chai', 'pytest', 'junit', 'selenium', 'cypress', 'playwright',

    # Data & Analytics
    'tableau', 'powerbi', 'looker', 'metabase', 'superset', 'grafana',
    'airflow', 'dbt', 'spark', 'hadoop', 'kafka', 'rabbitmq', 'celery',
    'streamlit', 'dash', 'jupyter', 'colab', 'bigquery', 'snowflake', 'databricks',

    # Design
    'figma', 'sketch', 'adobexd', 'photoshop', 'illustrator', 'indesign',
    'after-effects', 'premiere', 'canva', 'framer',

    # CMS & E-commerce
    'wordpress', 'shopify', 'woocommerce', 'magento', 'drupal', 'contentful',
    'sanity', 'strapi', 'ghost',

    # Mobile
    'android', 'ios', 'swiftui', 'jetpack-compose',

    # AI/ML Specific
    'llm', 'gpt', 'bert', 'llama', 'claude', 'gemini', 'stable-diffusion',
    'yolo', 'resnet', 'vgg', 'gan', 'cnn', 'rnn', 'lstm', 'transformer',
}


def is_valid_skill(term: str) -> bool:
    """
    Determine if a term is a valid technical skill

    Args:
        term: Skill term to validate

    Returns:
        True if valid skill, False if noise word
    """
    if not term or len(term) < 2:
        return False

    term_lower = term.lower().strip()

    # Remove special characters for comparison
    term_clean = ''.join(c for c in term_lower if c.isalnum() or c in ['-', '.', '+', '#'])

    # Check if it's in the noise word blacklist
    if term_clean in NOISE_WORDS:
        return False

    # Check if any part of the noise words is the entire term
    if term_lower in NOISE_WORDS:
        return False

    # Check if it's in the known valid skills whitelist
    if term_clean in VALID_SKILLS:
        return True

    # Additional validation rules

    # Must have at least one letter
    if not any(c.isalpha() for c in term):
        return False

    # Too long (probably not a skill name)
    if len(term_clean) > 30:
        return False

    # Single letter skills (only C and R are valid)
    if len(term_clean) == 1 and term_clean not in ['c', 'r']:
        return False

    # Check for skill-like patterns
    skill_patterns = [
        term_clean.endswith('.js'),  # JavaScript frameworks
        term_clean.endswith('.py'),  # Python packages
        term_clean.endswith('++'),   # C++
        '#' in term_clean,           # C#, F#
        '-' in term_clean and len(term_clean) > 3,  # kebab-case (react-native)
        '.' in term_clean and len(term_clean.split('.')) == 2,  # dot notation (Next.js)
    ]

    if any(skill_patterns):
        return True

    # If it's an acronym (all caps, 2-5 letters), it might be valid
    if term.isupper() and 2 <= len(term) <= 5:
        # But not if it's a noise acronym
        noise_acronyms = {'CS', 'IT', 'AI', 'ML', 'DL', 'DS', 'DE', 'DA', 'UX', 'UI'}
        if term not in noise_acronyms:
            return True

    # Default: allow it if it passed all checks
    # (Conservative approach - better to have some false positives than miss real skills)
    return True


def filter_skills(skills: list) -> list:
    """
    Filter a list of skills to remove noise words

    Args:
        skills: List of skill terms

    Returns:
        Filtered list of valid skills
    """
    return [skill for skill in skills if is_valid_skill(skill)]


def filter_skill_dict(skill_dict: dict) -> dict:
    """
    Filter a dictionary of categorized skills

    Args:
        skill_dict: Dict with categories as keys, skill lists as values

    Returns:
        Filtered skill dictionary
    """
    filtered = {}
    for category, skills in skill_dict.items():
        filtered[category] = filter_skills(skills)
    return filtered

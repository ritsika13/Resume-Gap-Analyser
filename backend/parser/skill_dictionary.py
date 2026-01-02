"""
Comprehensive Skill Dictionary for Resume Parsing
Maps known technical skills to their categories
"""

SKILL_DICTIONARY = {
    'programming_languages': [
        # Common languages
        'python', 'java', 'javascript', 'typescript', 'c', 'c++', 'c#', 'csharp',
        'go', 'golang', 'rust', 'ruby', 'php', 'swift', 'kotlin', 'scala',
        'r', 'matlab', 'julia', 'perl', 'bash', 'shell', 'powershell',

        # Web languages
        'html', 'css', 'sass', 'scss', 'less',

        # Other languages
        'objective-c', 'objectivec', 'dart', 'elixir', 'haskell', 'clojure',
        'groovy', 'lua', 'fortran', 'cobol', 'assembly', 'vb.net', 'vba',
        'f#', 'fsharp', 'erlang', 'racket', 'scheme', 'lisp', 'prolog',

        # SQL variants
        'sql', 'plsql', 'pl/sql', 't-sql', 'tsql', 'mysql', 'postgresql', 'postgres'
    ],

    'frameworks_libraries': [
        # Frontend frameworks
        'react', 'react.js', 'reactjs', 'angular', 'vue', 'vue.js', 'vuejs',
        'svelte', 'ember', 'backbone', 'knockout', 'preact', 'solid',

        # React ecosystem
        'next.js', 'nextjs', 'gatsby', 'remix', 'react-native', 'react native',
        'redux', 'mobx', 'recoil', 'zustand', 'react-query', 'react query',
        'react-router', 'react router',

        # Vue ecosystem
        'nuxt', 'nuxt.js', 'nuxtjs', 'vuex', 'pinia', 'vue-router', 'vue router',

        # Backend frameworks
        'django', 'flask', 'fastapi', 'pyramid', 'tornado', 'bottle',
        'spring', 'spring boot', 'springboot', 'hibernate', 'struts',
        'express', 'express.js', 'expressjs', 'koa', 'hapi', 'nest', 'nestjs', 'nest.js',
        'node.js', 'nodejs', 'node',
        'laravel', 'symfony', 'codeigniter', 'yii', 'cakephp',
        'rails', 'ruby on rails', 'rubyonrails', 'sinatra',
        'asp.net', 'dotnet', '.net', '.net core', 'entity framework',
        'gin', 'echo', 'fiber', 'beego',

        # Mobile frameworks
        'flutter', 'ionic', 'xamarin', 'cordova', 'phonegap', 'capacitor',
        'react-native', 'swiftui', 'jetpack compose', 'compose',

        # Data science & ML libraries
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'scikitlearn',
        'pandas', 'numpy', 'scipy', 'statsmodels', 'xgboost', 'lightgbm', 'catboost',
        'opencv', 'cv2', 'pillow', 'scikit-image', 'skimage',
        'nltk', 'spacy', 'gensim', 'transformers', 'huggingface', 'hugging face',
        'langchain', 'llamaindex', 'haystack',
        'matplotlib', 'seaborn', 'plotly', 'bokeh', 'altair', 'ggplot',
        'theano', 'caffe', 'mxnet', 'chainer', 'jax', 'flax',

        # Testing frameworks
        'jest', 'mocha', 'chai', 'jasmine', 'karma', 'cypress', 'playwright',
        'selenium', 'webdriver', 'puppeteer', 'testcafe',
        'pytest', 'unittest', 'nose', 'doctest',
        'junit', 'testng', 'mockito', 'hamcrest',
        'rspec', 'minitest',

        # CSS frameworks
        'bootstrap', 'tailwind', 'tailwindcss', 'bulma', 'foundation',
        'materialize', 'material-ui', 'mui', 'chakra', 'chakra-ui',
        'ant-design', 'antd', 'semantic-ui', 'shadcn', 'daisyui',

        # Other libraries
        'jquery', 'lodash', 'underscore', 'moment', 'date-fns', 'dayjs',
        'axios', 'fetch', 'ajax', 'websocket', 'socket.io', 'socketio',
        'graphql', 'apollo', 'relay', 'urql',
        'rxjs', 'ramda', 'immutable', 'immer',
        'd3', 'd3.js', 'chart.js', 'chartjs', 'highcharts', 'echarts',
        'three.js', 'threejs', 'babylon.js', 'babylonjs', 'pixi.js', 'pixijs',
        'electron', 'tauri', 'nw.js',

        # Big data & streaming
        'spark', 'apache spark', 'pyspark', 'hadoop', 'mapreduce',
        'kafka', 'apache kafka', 'flink', 'storm', 'samza',
        'airflow', 'apache airflow', 'luigi', 'dagster', 'prefect',
        'dbt', 'great expectations',
    ],

    'tools_software': [
        # Version control
        'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial', 'perforce',

        # IDEs & Editors
        'vscode', 'visual studio code', 'vim', 'neovim', 'emacs',
        'intellij', 'pycharm', 'webstorm', 'phpstorm', 'goland', 'rider',
        'eclipse', 'netbeans', 'atom', 'sublime', 'sublime text',
        'xcode', 'android studio', 'visual studio',

        # DevOps & CI/CD
        'docker', 'kubernetes', 'k8s', 'helm', 'istio', 'linkerd',
        'jenkins', 'travis', 'circleci', 'github actions', 'gitlab ci',
        'azure devops', 'bamboo', 'teamcity',
        'ansible', 'terraform', 'pulumi', 'cloudformation', 'cdk',
        'chef', 'puppet', 'saltstack', 'vagrant',
        'prometheus', 'grafana', 'datadog', 'new relic', 'splunk',
        'elk', 'elasticsearch', 'logstash', 'kibana',

        # Cloud platforms
        'aws', 'amazon web services', 'azure', 'gcp', 'google cloud',
        'heroku', 'vercel', 'netlify', 'digitalocean', 'linode',
        'cloudflare', 'fastly', 'akamai',

        # AWS services
        'ec2', 's3', 'lambda', 'rds', 'dynamodb', 'sqs', 'sns',
        'cloudfront', 'route53', 'vpc', 'iam', 'cloudwatch',
        'ecs', 'eks', 'fargate', 'beanstalk', 'amplify',

        # Web servers
        'nginx', 'apache', 'iis', 'tomcat', 'jetty', 'gunicorn', 'uwsgi',

        # API tools
        'postman', 'insomnia', 'swagger', 'openapi', 'rest', 'restful',
        'graphql', 'grpc', 'soap', 'api gateway',

        # Build tools
        'webpack', 'vite', 'parcel', 'rollup', 'esbuild', 'turbopack',
        'babel', 'swc', 'typescript compiler', 'tsc',
        'maven', 'gradle', 'ant', 'sbt',
        'make', 'cmake', 'ninja', 'bazel',
        'npm', 'yarn', 'pnpm', 'pip', 'conda', 'poetry',

        # Linters & formatters
        'eslint', 'prettier', 'stylelint', 'tslint',
        'pylint', 'flake8', 'black', 'autopep8', 'mypy',
        'rubocop', 'standardrb',

        # Design tools
        'figma', 'sketch', 'adobe xd', 'adobexd', 'invision',
        'photoshop', 'illustrator', 'indesign', 'after effects',
        'premiere', 'premiere pro', 'lightroom',
        'canva', 'framer', 'principle', 'zeplin', 'abstract',

        # Data & Analytics
        'tableau', 'power bi', 'powerbi', 'looker', 'metabase',
        'superset', 'redash', 'grafana', 'qlik', 'sisense',
        'streamlit', 'dash', 'plotly dash', 'shiny',
        'jupyter', 'jupyter notebook', 'jupyterlab', 'colab',
        'google colab', 'kaggle', 'databricks',

        # Project management
        'jira', 'confluence', 'trello', 'asana', 'monday',
        'notion', 'clickup', 'basecamp', 'linear',
        'slack', 'teams', 'discord', 'zoom', 'meet',

        # Other tools
        'chrome devtools', 'firefox devtools', 'safari devtools',
        'wireshark', 'fiddler', 'charles proxy',
        'git bash', 'terminal', 'iterm', 'powershell', 'cmd',
        'regex', 'regular expressions', 'json', 'yaml', 'xml', 'csv',
        'markdown', 'latex', 'pandoc',
        'cursor', 'claude code', 'copilot', 'tabnine',
        'sandbox', 'bigquery sandbox',
    ],

    'databases': [
        # Relational databases
        'mysql', 'postgresql', 'postgres', 'mariadb', 'sqlite',
        'oracle', 'oracle db', 'sql server', 'mssql', 'ms sql',
        'db2', 'teradata', 'sybase', 'informix',

        # NoSQL databases
        'mongodb', 'couchdb', 'couchbase', 'cassandra',
        'redis', 'memcached', 'hazelcast',
        'elasticsearch', 'opensearch', 'solr',
        'neo4j', 'arangodb', 'orientdb', 'janusgraph',

        # Cloud databases
        'dynamodb', 'cosmosdb', 'firestore', 'firebase',
        'bigtable', 'bigquery', 'redshift', 'snowflake',
        'aurora', 'rds', 'cloud sql', 'azure sql',
        'supabase', 'planetscale', 'neon', 'cockroachdb',

        # Time series & specialized
        'influxdb', 'timescaledb', 'prometheus', 'clickhouse',
        'druid', 'pinot', 'rockset',

        # Graph databases
        'dgraph', 'memgraph', 'tigergraph',

        # Vector databases
        'pinecone', 'weaviate', 'milvus', 'qdrant', 'chroma',

        # ORMs & query builders
        'sqlalchemy', 'django orm', 'sequelize', 'prisma',
        'typeorm', 'knex', 'hibernate', 'mybatis',
        'active record', 'eloquent', 'doctrine',
    ]
}

# Skill variations and aliases
SKILL_ALIASES = {
    'react': ['reactjs', 'react.js'],
    'vue': ['vuejs', 'vue.js'],
    'next.js': ['nextjs'],
    'node.js': ['nodejs', 'node'],
    'scikit-learn': ['sklearn', 'scikitlearn'],
    'postgresql': ['postgres'],
    'c++': ['cpp'],
    'c#': ['csharp'],
    'typescript': ['ts'],
    'javascript': ['js'],
    'python': ['py'],
}


def normalize_skill(skill: str) -> str:
    """Normalize skill name for matching"""
    skill_lower = skill.lower().strip()

    # Remove common prefixes/suffixes
    skill_lower = skill_lower.replace('.js', 'js')
    skill_lower = skill_lower.replace('apache ', '')
    skill_lower = skill_lower.replace('google ', '')
    skill_lower = skill_lower.replace('microsoft ', '')

    # Handle special characters
    skill_lower = skill_lower.replace(' ', '').replace('-', '').replace('_', '')

    return skill_lower


def get_all_skills_flat():
    """Get a flat list of all skills across categories"""
    all_skills = []
    for category, skills in SKILL_DICTIONARY.items():
        all_skills.extend(skills)
    return list(set(all_skills))

import pandas as pd
import sqlite3
import re
import os
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")

# NLP Libraries
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

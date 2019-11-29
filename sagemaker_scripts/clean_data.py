import unidecode, re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin

def clean_string(x
                 , regex_string = ['[^\w\s]','\\n']
                 , replacement = ' '):
    x = x.lower()
    x = x.encode("latin1", errors="ignore").decode('latin1')
    x = unidecode.unidecode(x)
    for i in regex_string:
        x = re.sub(i, replacement, x)
    x = re.sub(' +', ' ', x).strip()
    return(x)

def tokenize(x, delimeter = ' '):
    x = x.split(sep = delimeter)
    return(x)

def clean_tokens(x
                 , stop_words = stopwords.words('english')
                 , min_string_length = 2
                 , stem = True
                 , lemmatize_pos = None
                 , sort=True):
    x = [w for w in x if w not in stop_words]
    x = [w for w in x if len(w) >= min_string_length]
    if stem:
        x = [SnowballStemmer("english", ignore_stopwords=False).stem(w) for w in x]
    if lemmatize_pos is not None: # lemmatize_pos = 'v'
        x = [WordNetLemmatizer().lemmatize(w, pos=lemmatize_pos) for w in x]
    if sort:
        x = sorted(x)
    return(x)

def clean_all(x
              , regex_string = ['[^\w\s]','\\n']
              , replacement = ' '
              , delimeter = ' '
              , stop_words = stopwords.words('english')
              , min_string_length = 2
              , stem = True
              , lemmatize_pos = None
              , sort=False):
    x = clean_string(x
                     , regex_string = regex_string
                     , replacement = replacement)
    x = tokenize(x
                , delimeter = delimeter)
    x = clean_tokens(x
                     , stop_words = stop_words
                     , min_string_length = min_string_length
                     , stem = stem
                     , lemmatize_pos = lemmatize_pos
                     , sort = sort)
    return(x)

class StringClean(BaseEstimator, TransformerMixin):    
    def __init__(self
                 , regex_string = ['[^\w\s]','\\n']
                 , replacement = ' '
                 , stop_words = stopwords.words('english')
                 , min_string_length = 2
                 , stem = True
                 , lemmatize_pos = None):
        self.regex_string = regex_string
        self.replacement = replacement  
        self.stop_words = stop_words
        self.min_string_length = min_string_length
        self.stem = stem
        self.lemmatize_pos = lemmatize_pos
        
    def fit(self, X, y=None):
        return(self)

    def transform(self, X, y=None):
        X = [clean_all(x) for x in X]
        return(X)    
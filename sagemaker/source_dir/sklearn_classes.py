################################################################################
## TECH CLASSIFICATION: CLASSES
################################################################################

# For defining sklearn classes
from sklearn.base import BaseEstimator, TransformerMixin 

## Text cleaning
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nltk'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'unidecode==1.0.22'])
import unidecode, re
# from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

## Embedding methods
import numpy as np
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])
from scipy.sparse import csr_matrix, hstack # for stacking dimensionality reduction matricies
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import TruncatedSVD, NMF # PCA will not work on sparse matricies
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gensim'])
from gensim.models.doc2vec import TaggedDocument as td
from gensim.models import Doc2Vec as d2v #, phrases as bigram # Use sklearn tfidf vectorizer instead, as ngram > 2

## Classification
from sklearn.calibration import CalibratedClassifierCV, LinearSVC

########################################
## TEXT CLEANING
########################################

def clean_string(x
                 , regex_string = ['[^\w\s]','\\n']
                 , replacement = ' '):
    x = x.lower()
    x = x.encode("latin1", errors="ignore").decode('latin1')
    for i in regex_string:
        x = re.sub(i, replacement, x)
    x = re.sub(' +', ' ', x).strip()
    return(x)

def tokenize(x, delimeter = ' '):
    x = x.split(sep = delimeter)
    return(x)

def clean_tokens(x
                 , stop_words = ['TBC'] # stopwords.words('english')
                 , min_string_length = 2
                 , stem = True
                 , lemmatize_pos = None
                 , sort=True):
    x = [w for w in x if w not in stop_words]
    x = [w for w in x if len(w) >= min_string_length]
    x = [unidecode.unidecode(w) for w in x]
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
              , stop_words = ['TBC'] # stopwords.words('english')
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

## building class so later we can add to the text pipeline

class StringClean(BaseEstimator, TransformerMixin):    
    def __init__(self
                 , regex_string = ['[^\w\s]','\\n']
                 , replacement = ' '
                 , stop_words = ['TBC'] # stopwords.words('english')
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
        X = [clean_all(x
                       , regex_string = self.regex_string
                       , replacement = self.replacement
                       , stop_words = self.stop_words
                       , min_string_length = self.min_string_length
                       , stem = self.stem
                       , lemmatize_pos= self.lemmatize_pos) for x in X]
        return(X)


########################################
## CHOOSING COLUMNS (HETEROGENEOUS FEATURES)
########################################

class col_chooser(BaseEstimator, TransformerMixin):
    
    """Choose which heterogeneous feature to feed into the pipeline"""
    
    def __init__(self, key = ''):
        self.key = key
        
    def fit(self, X, y=None):
        return(self)
    
    def transform(self, X):
        X = X.fillna('')
        try:
            return(X[self.key])
        except:
            return(None)


########################################
## EMBEDDING 1: TF-IDF + TRUNCATED SVD
########################################

# Stored in the main guard

# tfidf_tsvd_struct_pipeline = Pipeline([('tfidf', TfidfVectorizer(preprocessor = ' '.join))
#                                        , ('tsvd', TruncatedSVD(n_components=50))])


########################################
## EMBEDDING 2: Doc2Vec
########################################

class d2v_struct_pipeline(BaseEstimator, TransformerMixin):

    """Create D2V vectorized features from text"""
    
    # https://arxiv.org/pdf/1405.4053v2.pdf
    # https://arxiv.org/pdf/1301.3781.pdf
    
    # https://medium.com/@amarbudhiraja/understanding-document-embeddings-of-doc2vec-bfe7237a26da
    
    def __init__(self, vector_size=200, window=10, min_count=1, dm=1): # learning_rate=0.02, epochs=20
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.dm = dm
#         self.learning_rate = learning_rate
#         self.epochs = epochs
    
    def fit(self, X, *_, **args):
        tagged_docs = list(map(lambda i, line: td(line, [i]), list(range(len(X))), X))
        self.d2v_dm = d2v(tagged_docs
                          , vector_size=self.vector_size
                          , window=self.window
                          , min_count=self.min_count
                          , dm=self.dm
                          , **args)
        return self

    def transform(self, X, *_):
        d2v_dm_m = [self.d2v_dm.infer_vector(x) for x in X]
        return(csr_matrix(d2v_dm_m))


########################################
## CLASSIFICATION
########################################        

# IN FUTURE, SHOULD BE A FEATURE UNION OF DIFFERENT CLASSIFIERS

classify = CalibratedClassifierCV(LinearSVC(class_weight = 'balanced', max_iter = 10000))
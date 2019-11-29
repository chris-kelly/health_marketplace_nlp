import numpy as np

from scipy.sparse import csr_matrix, hstack # for stacking dimensionality reduction matricies

from gensim.models.doc2vec import TaggedDocument as td
from gensim.models import Doc2Vec as d2v #, phrases as bigram # Use sklearn tfidf vectorizer instead, as ngram > 2

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import TruncatedSVD, NMF # PCA will not work on sparse matricies

from sklearn.base import BaseEstimator, TransformerMixin

class col_chooser(BaseEstimator, TransformerMixin):
    
    """Choose which heterogeneous feature to feed into the pipeline"""
    
    def __init__(self, key = ''):
        self.key = key
        
    def fit(self, X, y=None):
        return(self)
    
    def transform(self, X):
        try:
            return(X[self.key])
        except:
            return(None)

class tfidf_tsvd_struct_pipeline(BaseEstimator, TransformerMixin):

    """Create TF_IDF vectorized features from text"""
    
    def __init__(self, n_components=100, norm='l2', ngram_range=(1, 2), preprocessor = ' '.join):
        self.n_components = n_components
        self.norm = norm
        self.ngram_range = ngram_range
        self.preprocessor = preprocessor
        
    def fit(self, X, *_, **args):
        if X is not None:
            self.tf_idf = TfidfVectorizer(norm = self.norm
                                          , ngram_range = self.ngram_range
                                          , preprocessor = self.preprocessor
                                          , **args).fit(X) # norm='l2', ngram_range=(1, 2)
            # self.n_components = n_components
            if (self.n_components != None):
                self.t_m = TruncatedSVD(self.n_components).fit(self.tf_idf.transform(X))
            return(self)
        else:
            return(self)

    def transform(self, X, *_):
        if self.tf_idf is not None:
            if (self.n_components != None):
                t_m = csr_matrix(self.t_m.transform(self.tf_idf.transform(X)))
            else:
                t_m = csr_matrix(self.tf_idf.transform(X))
            return(t_m)
        else:
            return(None)


class d2v_struct_pipeline(BaseEstimator, TransformerMixin):

    """Create D2V vectorized features from text"""
    
    # https://arxiv.org/pdf/1405.4053v2.pdf
    # https://arxiv.org/pdf/1301.3781.pdf
    
    # https://medium.com/@amarbudhiraja/understanding-document-embeddings-of-doc2vec-bfe7237a26da
    
    def __init__(self, vector_size=100, window=10, min_count=1, dm=1): # learning_rate=0.02, epochs=20
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
'''
Process that predicts if an abstract is a technology or not
The pipeline is composed of the following components:

Features: a feature union of heterogenous data sources, undergoing TF-IDF+TSVD and Doc2Vec
Classifier: a Linear SVC (should be an ensemble)
'''

from io import StringIO
import subprocess, sys, os, argparse
# Install boto3
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 's3fs']) 
# Install for encoders and worker (needed to output a json)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sagemaker-containers']) # for encoders and worker (needed to output a json)

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import TruncatedSVD, NMF # PCA will not work on sparse matricies

## Classification
from sklearn.calibration import CalibratedClassifierCV, LinearSVC

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

from sklearn.externals import joblib

from sklearn_classes import StringClean, col_chooser, d2v_struct_pipeline

# stuff to make sagemaker work
import pandas as pd, numpy as np
from sagemaker_containers.beta.framework import encoders, worker
import json
from ast import literal_eval

# From the SageMaker documentation:
# Because the SageMaker imports your training script, you should put your
# training code in a main guard (if __name__=='__main__':) if you are using
# the same script to host your model, so that SageMaker does not inadvertently
# run your training code at the wrong point in execution.

if __name__ == "__main__":
    # PARSE ARGUMENTS
    arg_parser = argparse.ArgumentParser()

    # Hyperparameters (TBC)
    # arg_parser.add_argument('--n-estimators', type=int, default=400)

    # Data and model directories
    arg_parser.add_argument("--target_col", type=str, default="label") # target variable
    arg_parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    arg_parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    # --model_dir is a required command line argument when using SageMaker.
    arg_parser.add_argument("--model_dir",type=str, default=os.environ.get('SM_MODEL_DIR'))
    args = arg_parser.parse_args()  # Read command line arguments

    # GLOBAL PARAMETERS
    OUT_PATH = os.path.join(args.model_dir, "model.joblib")
    TRAIN_PATH = os.path.join(args.train, "train.csv")
    TEST_PATH = os.path.join(args.test, "test.csv")

    # GET TRAIN AND TEST DATA
    train_df = pd.read_csv(TRAIN_PATH) # , converters={'field_name': literal_eval}
    X_train = train_df.drop(columns=[args.target_col])
    y_train = train_df[args.target_col]

    test_df = pd.read_csv(TEST_PATH) #  # , converters={'field_name': literal_eval}
    X_test = test_df.drop(columns=[args.target_col])
    y_test = test_df[args.target_col]

    # BUILD THE PIPELINE

    ########################################
    ## EMBEDDING 1: TF-IDF + TRUNCATED SVD
    ########################################

    tfidf_tsvd_struct_pipeline = Pipeline([('tfidf', TfidfVectorizer(preprocessor = ' '.join))
                                           , ('tsvd', TruncatedSVD(n_components=200))])

    ########################################
    ## UNION EMBEDDING TYPES
    ########################################

    unsupervised_union = \
    FeatureUnion([("tfidf_svd", tfidf_tsvd_struct_pipeline)
#                   , ("d2v1", d2v_struct_pipeline(dm=1))
                  , ("d2v0", d2v_struct_pipeline(dm=0))])

    ########################################
    ## HETEROGENEOUS INPUTS FOR EMBEDDINGS
    ########################################

    text_inputs = \
    FeatureUnion([('text_1', Pipeline([('col', col_chooser(key = 'abstract'))
                                       , ('str_clean', StringClean())
                                       , ('comb', unsupervised_union)]))
#                   ,('text_2', Pipeline([('col', col_chooser(key = 'other_field'))
#                                         , ('str_clean', StringClean())
#                                         , ('comb', unsupervised_union)]))
                 ])

    ########################################
    ## CLASSIFICATION
    ########################################

    classify = CalibratedClassifierCV(LinearSVC(class_weight = 'balanced', max_iter = 10000))

    ########################################
    ## EMBEDDING + CLASSIFICATION
    ########################################

    pipeline = Pipeline([("features", text_inputs)
                         , ("classifer", classify)]) # memory=cachedir

    scoring = {'f1_macro': 'f1_macro'
               , 'roc_curve': 'roc_auc'
               , 'precision': 'precision_macro'
               , 'recall': 'recall_macro'}
    
    ########################################
    ## GRIDSEARCH
    ######################################## 
    
    param_grid = dict(features__text_1__comb__tfidf_svd__tsvd__n_components=[200]
#                       , features__text_1__comb__d2v1__vector_size=[200]
                      , features__text_1__comb__d2v0__vector_size=[200]
#                       , features__text_2__comb__tfidf_svd__tsvd__n_components=[200]
#                       , features__text_2__comb__d2v1__vector_size=[200]
#                       , features__text_2__comb__d2v0__vector_size=[200]
                      , classifer__cv = [5]
                      , classifer__base_estimator__loss = ['hinge']
                     )

    grid_search = GridSearchCV(pipeline
                               , param_grid=param_grid
                               , cv=5
                               , iid = False
                               , scoring = scoring
                               , refit='f1_macro'
                               , return_train_score=True
                               , verbose=False)
    
    # fit the pipeline with gridsearch
    locale_model = grid_search.fit(X = X_train, y = y_train)
    # y_pred = locale_model.predict(X_test)

    # save trained model
    joblib.dump(locale_model, OUT_PATH)


def model_fn(model_dir):
    """
    Deserialize fitted model
    """
    pipeline = joblib.load(os.path.join(model_dir, "model.joblib"))
    return(pipeline)


def input_fn(request_body, request_content_type):
    """
    Parse input data payload
    This function currently only supports csv input but can be modified to support json.
    """
    if request_content_type == 'text/csv':
        try:
            df = pd.read_csv(request_body)
            return(df)
        except FileNotFoundError:
            df2 = pd.read_csv(StringIO(request_body),
                          names=['id','abstract'])
            return(df2)
    else:
        raise ValueError("{} not supported by script".format(request_content_type))

def predict_fn(input_data, pipeline):
    prediction = pipeline.predict(input_data)
    pred_proba = pipeline.predict_proba(input_data)
    return(np.array(prediction))


def output_fn(prediction, accept):
    """
    Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    TheContentType and mimetype need to be set as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})
        json_output = {"instances": instances}
        response1 = worker.Response(json.dumps(json_output), accept, mimetype=accept)
        return(response1)
    elif accept == 'text/csv':
        response2 = worker.Response(encoders.encode(prediction, accept), accept, mimetype=accept)
        return(response2) 
    else:
        raise RuntimeError("{} accept type is not supported by this script.".format(accept))
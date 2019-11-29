from clean_data import StringClean
from embedding import col_chooser, tfidf_tsvd_struct_pipeline, d2v_struct_pipeline
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import make_scorer, precision_score, recall_score

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV

import pandas as pd
from sklearn.externals import joblib

# Sagemaker passes command line arguments to your script when it runs so we import the
# `argparse` module to receive these arguments
# import argparse

# From the SageMaker documentation:
# Because the SageMaker imports your training script, you should put your training code
# in a main guard (if __name__=='__main__':) if you are using the same script to host
# your model, so that SageMaker does not inadvertently run your training code at the
# wrong point in execution.
if __name__ == "__main__":
	
	# arg_parser = argparse.ArgumentParser()
	# arg_parser.add_argument("--arg_name", type=str, default="arg_value")
	# args = arg_parser.parse_args()  # Read command line arguments

	########################################
	## PERMIT MULTIPLE UNSUPERVISED EMBEDDING OPTIONS IN PIPELINE
	########################################

	unsupervised_union = \
	FeatureUnion([("tfidf_svd", tfidf_tsvd_struct_pipeline(preprocessor = ' '.join))
	              , ("d2v1", d2v_struct_pipeline(dm=1))
	              , ("d2v0", d2v_struct_pipeline(dm=0))])

	text_inputs = \
	FeatureUnion([('text_1', Pipeline([('col', col_chooser(key = 'text')) # key = 'abstract'
	                                   , ('comb', unsupervised_union)]))
	             ])

	classifier = CalibratedClassifierCV(LinearSVC(class_weight = 'balanced')) # SVC(kernel="linear")

	pipeline = Pipeline([("features", text_inputs)
	                     , ("classifer", classifier)]) # memory=cachedir

	########################################
	## SCORING FOR GRIDSEARCH
	########################################

	# DEFINE PERFORMANCE METRICS
	scoring = {'f1_macro': 'f1_macro'
	           , 'roc_curve': 'roc_auc'
	           , 'precision': 'precision_macro'
	           , 'recall': 'recall_macro'}

	# DEFINE GRIDSEARCH
	param_grid = dict(features__text_1__comb__tfidf_svd__n_components=[400]
	                  , features__text_1__comb__d2v1__vector_size=[300]
	                  , features__text_1__comb__d2v0__vector_size=[400]
	#                   , features__transformer_weights = [{'text_1': 1, 'text_2': 1, 'text_3': 0, 'text_4': 0}]
	                 )

	grid_search = GridSearchCV(pipeline
	                           , param_grid=param_grid
	                           , cv=5
	                           , scoring = scoring
	                           , refit='roc_curve' #, refit = False # 
	                           , return_train_score=True
	                           , verbose=False)

	########################################
	## IMPORT DATA
	########################################	

	df = pd.read_csv('../csv/tech_no_tech.csv')
	df.columns = ['id', 'label', 'abstract']
	df['text'] = StringClean().fit_transform(df.abstract)

	X_train, X_test, y_train, y_test = train_test_split(df, df['label'], test_size=0.2, random_state=0)

	gs_fit = grid_search.fit(X = X_train.to_records()
							, y = y_train)

	y_pred = gs_fit.predict(X_test)

	joblib.dump(gs_fit, OUT_PATH)

# df = pd.read_csv('../csv/tech_no_tech.csv')
# df.columns = ['id', 'label', 'abstract']
# df['text'] = StringClean().fit_transform(df.abstract)

# X_train, X_test, y_train, y_test = train_test_split(df, df['label'], test_size=0.2, random_state=0)

# gs_fit = grid_search.fit(X = X_train.to_records()
#                          , y = y_train)
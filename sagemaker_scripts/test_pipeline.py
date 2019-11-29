## LIBRARIES
import pandas as pd, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

# GLOBAL PARAMETERS
DATA_DIR = os.getcwd()
# TRAIN_PATH = DATA_DIR + "/train/train_set.csv"
# TEST_PATH = DATA_DIR + "/test/test_set.csv"
OUT_PATH = DATA_DIR + "/model/model.joblib"
TARGET_COLUMN = "quality"

# GET TRAIN AND TEST DATA
df = pd.read_csv(DATA_DIR + '/data.csv', header=0, sep=';')
X = df.drop(columns=[TARGET_COLUMN], axis=1)
y = df[TARGET_COLUMN]
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.2, random_state=42)

# CREATE AND FIT PIPELINE
pipeline = Pipeline([("scaling", StandardScaler())
                     , ("classifer", LinearSVC())])
pipeline.fit(X_train, y_train)

# PREDICT AND EVALUATE ON TEST SET
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred)

# SAVE TRAINED MODEL
joblib.dump(pipeline, OUT_PATH)
print("Job complete")
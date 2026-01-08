import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

data = {
    "study_hours": [2, 4, 6, 8, 10, 3, 5, 7],
    "attendance": [60, 65, 70, 80, 90, 55, 75, 85],
    "prev_score": [40, 45, 55, 65, 80, 35, 60, 75],
    "pass": [0, 0, 1, 1, 1, 0, 1, 1]}

df = pd.DataFrame(data)

X = df.drop('pass', axis = 1)
y = df['pass']

pipeline = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])

pipeline.fit(X,y)

joblib.dump(pipeline, 'student_pipeline.pkl')
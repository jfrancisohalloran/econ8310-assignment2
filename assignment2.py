import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
train_df = pd.read_csv(train_url)

train_df = train_df.drop(['id', 'DateTime'], axis=1)

X_train = train_df.drop('meal', axis=1)
y_train = train_df['meal']

model = RandomForestClassifier(random_state=42)
modelFit = model.fit(X_train, y_train)

test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
test_df = pd.read_csv(test_url)

test_df = test_df.drop(['id', 'DateTime'], axis=1)

predictions = modelFit.predict(test_df)

pred = predictions.astype(int)

print(pred)

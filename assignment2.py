import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
df_train = pd.read_csv(train_url)
df_train = df_train.drop(['id', 'DateTime'], axis=1)
X_train = df_train.drop('meal', axis=1)
y_train = df_train['meal']

cv_models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBClassifier': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}
cv_scores = {}
print("Cross-Validation Accuracy Scores:")
for name, candidate in cv_models.items():
    scores = cross_val_score(candidate, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores[name] = np.mean(scores)
    print(f"{name}: {np.mean(scores):.4f}")

best_model_name = max(cv_scores, key=cv_scores.get)
print("\nBest model based on CV accuracy:", best_model_name)

model = DecisionTreeClassifier(random_state=42)
modelFit = model.fit(X_train, y_train)

test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
df_test = pd.read_csv(test_url)
df_test = df_test.drop(['id', 'DateTime'], axis=1)

if 'meal' in df_test.columns:
    df_test = df_test.drop('meal', axis=1)

predictions = modelFit.predict(df_test) 

predictions = predictions.flatten()

pred = predictions.astype(int).tolist()
print("Length of pred:", len(pred))
print("First 10 predictions:", pred[:10])

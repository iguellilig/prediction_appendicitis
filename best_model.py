import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from xgboost import XGBClassifier
import numpy as np
from imblearn.over_sampling import SMOTE
import joblib

# random seed
seed = 123

# Read original dataset
data_df = pd.read_csv("data/data.csv")
data_df.sample(frac=1, random_state=seed)
data_df = data_df.dropna(subset=["Diagnosis"])
data_df.fillna(-1, inplace = True)
data_df = data_df[['Age', 
             'BMI', 
             'Sex', 
             'Height',
             'Weight',
             'Appendix_Diameter',
             'Migratory_Pain', 
             'Lower_Right_Abd_Pain',
             'Contralateral_Rebound_Tenderness',
             'Coughing_Pain',
             'Loss_of_Appetite',
             'Body_Temperature',
             'WBC_Count',
             'Neutrophil_Percentage',
             'Segmented_Neutrophils',
             'Neutrophilia',
             'RBC_Count',
             'Hemoglobin',
             'RDW',
             'Thrombocyte_Count',
             'Ketones_in_Urine',
             'RBC_in_Urine',
             'WBC_in_Urine',
             'CRP',
             'Dysuria',
             'Stool',
             'Peritonitis',
             'Psoas_Sign',
             'Diagnosis'
            ]]
    
val_map = [('Sex', {'male':0, 'female':1}), 
                    ('Migratory_Pain', {'yes': 2, 'no':3}), 
                    ('Lower_Right_Abd_Pain', {'yes': 2, 'no':3}),
                    ('Contralateral_Rebound_Tenderness', {'yes': 2, 'no':3}),
                    ('Coughing_Pain', {'yes': 2, 'no':3}),
                    ('Loss_of_Appetite', {'yes': 2, 'no':3}),
                    ('Neutrophilia', {'yes': 2, 'no':3}),
                    ('Ketones_in_Urine', {'no': 3, '+':4, '++':5, '+++':6}),
                    ('RBC_in_Urine', {'no': 3, '+':4, '++':5, '+++':6}),
                    ('WBC_in_Urine', {'no': 3, '+':4, '++':5, '+++':6}),
                    ('Dysuria', {'yes': 2, 'no':3}),
                    ('Stool', {'normal': 7, 'constipation':8, 'diarrhea':9, 'constipation, diarrhea':10}),
                    ('Peritonitis', {'no': 3, 'local':11, 'generalized':12}),
                    ('Psoas_Sign', {'yes': 2, 'no':3}),
                    ('Diagnosis', {'appendicitis':1, 'no appendicitis': 0})]
for column, themap in val_map:
    data_df[column] = data_df[column].replace(themap)
print(data_df)
    
#X = X.astype(float)
#X = X.round(3)

X = data_df[['Age', 
             'BMI', 
             'Sex', 
             'Height',
             'Weight',
             'Appendix_Diameter',
             'Migratory_Pain', 
             'Lower_Right_Abd_Pain',
             'Contralateral_Rebound_Tenderness',
             'Coughing_Pain',
             'Loss_of_Appetite',
             'Body_Temperature',
             'WBC_Count',
             'Neutrophil_Percentage',
             'Segmented_Neutrophils',
             'Neutrophilia',
             'RBC_Count',
             'Hemoglobin',
             'RDW',
             'Thrombocyte_Count',
             'Ketones_in_Urine',
             'RBC_in_Urine',
             'WBC_in_Urine',
             'CRP',
             'Dysuria',
             'Stool',
             'Peritonitis',
             'Psoas_Sign'
            ]]
             
y = data_df[['Diagnosis']]


# split data into train and test sets
# 70% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y)

X_train["Diagnosis"] = y_train["Diagnosis"]


#Random oversampling
minority_class = X_train[X_train["Diagnosis"] == 0]
majority_class = X_train[X_train["Diagnosis"] == 1]
minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class),random_state=42)
balanced_data = pd.concat([majority_class, minority_upsampled])
X_train = balanced_data
y_train = X_train["Diagnosis"]
X_train = X_train.drop(columns=["Diagnosis"])


X_train.to_csv("data/train_sample.csv")
y_train.to_csv("data/train_diagnosis.csv")
X_test.to_csv("data/test_sample.csv")
y_test.to_csv("data/test_diagnosis.csv")

# create an instance of the random forest classifier
#clf = XGBClassifier(n_estimators=500, max_depth=2, learning_rate=0.01, objective='binary:logistic')
clf = RandomForestClassifier(n_estimators=200)
'''clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)'''


# train the classifier on the training data
clf.fit(X_train, y_train)

# predict on the test set
y_pred = clf.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# save the model to disk
joblib.dump(clf, "best_model.sav")



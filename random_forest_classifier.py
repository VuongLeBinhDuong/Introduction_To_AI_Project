import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import re

process_df = pd.read_parquet("loan_status_processed_df.parquet")

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score

X_features = process_df.drop(['loan_status', 'issue_d', 'earliest_cr_line'], axis = 1)
y_target = process_df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size = 0.3, random_state = 2412)

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy = "minority", random_state = 81)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rfc_model = RandomForestClassifier(n_estimators = 100, bootstrap = True, random_state = 1000)
rfc_model.fit(X_train_resampled, y_train_resampled)

import pickle
pickle.dump(rfc_model,open('rfc_model.pkl','wb'))
model=pickle.load(open('rfc_model.pkl','rb'))
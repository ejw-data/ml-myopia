# Summary of models

# Produce a table of all the model parameters
# Only need to input a few model parameters as lists.
# each list can contain tuples that indicate how the object should be
# used as a 
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from statistics import mean, mode

from imblearn.pipeline import Pipeline
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_validate, validation_curve
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

summary=[]
holdout=[]
# 5 fold cross validation
cv = StratifiedKFold(n_splits=5)

# create process steps
pipes = [
    ("scaler", StandardScaler()),
    ("svc", SVC(random_state=2))
]

def model_compare(X_train, y_train, X_test, y_test, labels, pipes, cv):
    # create pipeline
    pipeline = Pipeline(pipes)

    # Train the scaler with the X_train data.
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    train_class_report = classification_report(y_train, y_train_pred, output_dict=True)
    y_test_pred = pipeline.predict(X_test)
    test_class_report = classification_report(y_test, y_test_pred, output_dict=True)
    holdout.append({"class":labels[0], train_class_report['0']})
    holdout.append({"class":labels[1], train_class_report['1']})
    # evaluation metrics
    scoring = ('f1', 'recall', 'precision', 'roc_auc')

    # evaluate pipeline
    scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)

    pipelength = len(pipes)
    summary.append({
        'pipe1': pipes[0][1], 
        'pipe2': pipes[1][1] if pipelength >= 2 else "",
        'pipe3': pipes[2][1] if pipelength >= 3 else "",
        'pipe4': pipes[3][1] if pipelength >= 4 else "",
        'pipe5': pipes[4][1] if pipelength >= 5 else "",
        'cv': cv,
        'f1':  mean(scores['test_f1']),
        'recall': mean(scores['test_recall']),
        'precision': mean(scores['test_precision']),
        'ROC AUC': mean(scores['test_roc_auc'])
    })

    return summary


# set gridsearch parameters
params = {
    "svc__C": [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 1000],
    "pca__n_components":[4,5,6,7]
}

# gridsearch with 5 fold cross validation setup
grid = GridSearchCV(pipeline, params);
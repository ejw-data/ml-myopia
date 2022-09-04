# Summary of models

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
from sklearn.metrics import classification_report, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
# from sklearn.metrics import average_precision_score

#  Add gridsearch in the future???
# # set gridsearch parameters
# params = {
#     "svc__C": [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 1000],
#     "pca__n_components":[4,5,6,7]
# }

# # gridsearch with 5 fold cross validation setup
# grid = GridSearchCV(pipeline, params);


summary=[]

def model_compare(X_train, y_train, X_test, y_test, labels, pipes, cv):
    # create pipeline
    pipeline = Pipeline(pipes)

    # Train the scaler with the X_train data.
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    train_class_report = classification_report(y_train, y_train_pred, output_dict=True)
    y_test_pred = pipeline.predict(X_test)
    test_class_report = classification_report(y_test, y_test_pred, output_dict=True)

    # evaluation metrics
    scoring = ('f1', 'recall', 'precision', 'roc_auc')

    # evaluate pipeline
    scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
    
    y_predicted_probability = pipeline.predict_proba(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test.to_list(), y_predicted_probability[:, 1])
    prc_auc0 = auc(recall, precision)

    pipelength = len(pipes)
    
    cross_validation={
        'type': 'cross validation',
        'pipe1': pipes[0][1], 
        'pipe2': pipes[1][1] if pipelength >= 2 else "",
        'pipe3': pipes[2][1] if pipelength >= 3 else "",
        'pipe4': pipes[3][1] if pipelength >= 4 else "",
        'pipe5': pipes[4][1] if pipelength >= 5 else "",
        'cv': cv,
        'f1-score':  mean(scores['test_f1']),
        'recall': mean(scores['test_recall']),
        'precision': mean(scores['test_precision']),
        'ROC_AUC': mean(scores['test_roc_auc']), 
        'Precision_Recall_AUC': round(prc_auc0,2)
    }
    
    summary.append(cross_validation)
    
    report_name = ['train', 'train', 'test', 'test']
    report_class = [0, 1, 0, 1]
    reports = [train_class_report['0'], train_class_report['1'], train_class_report['0'], train_class_report['1'] ]
    
    for report in reports:
        report_index = reports.index(report)
        report_class = labels[report_index]
        report_type = report_name[report_index]
        threshold ={
            'type': f'threshold_{report_type}_{report_class}',
            'pipe1': pipes[0][1], 
            'pipe2': pipes[1][1] if pipelength >= 2 else "",
            'pipe3': pipes[2][1] if pipelength >= 3 else "",
            'pipe4': pipes[3][1] if pipelength >= 4 else "",
            'pipe5': pipes[4][1] if pipelength >= 5 else "",
            'cv': cv
        }
        for key, value in report.items():
            threshold[key] = value
    
        threshold['number_samples'] = threshold['support']
        del threshold['support']
        
        summary.append(threshold)
    
    return summary




def model_compareZ(X_train, y_train, X_test, y_test, labels, pipes, cv):
    # create pipeline
    pipeline = Pipeline(pipes)

    # Train the scaler with the X_train data.
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    train_class_report = classification_report(y_train, y_train_pred, output_dict=True)
    y_test_pred = pipeline.predict(X_test)
    test_class_report = classification_report(y_test, y_test_pred, output_dict=True)
    
    train_class_report_0 = train_class_report['0']
    train_class_report_1 = train_class_report['1']
    test_class_report_0 = train_class_report['0']
    test_class_report_1 = train_class_report['1']

    # evaluation metrics
    scoring = ('f1', 'recall', 'precision', 'roc_auc')

    # evaluate pipeline
    scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)

    pipelength = len(pipes)
    
    cross_validation={
        'pipe1': pipes[0][1], 
        'pipe2': pipes[1][1] if pipelength >= 2 else "",
        'pipe3': pipes[2][1] if pipelength >= 3 else "",
        'pipe4': pipes[3][1] if pipelength >= 4 else "",
        'pipe5': pipes[4][1] if pipelength >= 5 else "",
        'cv': cv,
        'f1':  mean(scores['test_f1']),
        'recall': mean(scores['test_recall']),
        'precision': mean(scores['test_precision']),
        'ROC_AUC': mean(scores['test_roc_auc']), 
    }
    
    for key, value in train_class_report_0.items():
        cross_validation[f"train_0_{key}"] = value
        
    for key, value in train_class_report_1.items():
        cross_validation[f"train_1_{key}"] = value
        
    for key, value in test_class_report_0.items():
        cross_validation[f"test_0_{key}"] = value
                         
    for key, value in test_class_report_1.items():
        cross_validation[f"test_1_{key}"] = value
                         
    summary.append(cross_validation)

    return summary
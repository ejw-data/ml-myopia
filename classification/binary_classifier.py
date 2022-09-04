# Summary of models

# import numpy as np
import pandas as pd
from IPython.display import display, HTML
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



summary=[]

def model_compare(scenario, X_train, y_train, X_test, y_test, labels, pipes, cv="", tuner_params=""):
    # create pipeline
    pipeline = Pipeline(pipes)
    pipelength = len(pipes)
    # Holdout Evaluation Method
            
    # No tuning
    if not tuner_params:
        # Train the scaler with the X_train data.
        grid = GridSearchCV(pipeline, {});
        grid.fit(X_train, y_train)
        y_train_pred = grid.predict(X_train)
        y_test_pred = grid.predict(X_test)

    # With tuning and cross validation
    else:
        grid = GridSearchCV(pipeline, tuner_params);
        grid.fit(X_train, y_train)
        y_train_pred = grid.predict(X_train)
        y_test_pred = grid.predict(X_test)

        
    train_class_report = classification_report(y_train, y_train_pred, output_dict=True)
    test_class_report = classification_report(y_test, y_test_pred, output_dict=True)
    
    report_name = ['train', 'train', 'test', 'test']
    report_class = [0, 1, 0, 1]
    reports = [train_class_report['0'], train_class_report['1'], test_class_report['0'], test_class_report['1'] ]

    for index, report in enumerate(reports): 
        
        threshold ={
            'scenario':scenario,
            'type': f'threshold_{report_name[index]}_{report_class[index]}',
            'pipe1': pipes[0][1], 
            'pipe2': pipes[1][1] if pipelength >= 2 else "",
            'pipe3': pipes[2][1] if pipelength >= 3 else "",
            'pipe4': pipes[3][1] if pipelength >= 4 else "",
            'pipe5': pipes[4][1] if pipelength >= 5 else "",
        }
        
        for key, value in report.items():
            threshold[key] = value
            
        threshold['number_samples'] = threshold['support']
        del threshold['support']
        
        if tuner_params:
            threshold['cv'] = grid
            threshold['tuning_params'] = grid.best_params_
            
        if report_name[index]=='train':
            threshold['accuracy'] = train_class_report['accuracy']

        else:
            threshold['accuracy'] = test_class_report['accuracy']
        
        summary.append(threshold)
             

    # Cross Validation Evaluation Method
    # evaluation metrics
    scoring = ('f1', 'recall', 'precision', 'roc_auc', 'accuracy')
    if tuner_params:
        pipeline.set_params(**grid.best_params_)
        scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=1)
    else:  
        scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=1)
    
    # Calculcate Precision-Recall AUC
    y_predicted_probability = grid.predict_proba(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test.to_list(), y_predicted_probability[:, 1])
    prc_auc0 = auc(recall, precision)

    cross_validation={
        'scenario': scenario,
        'type': 'cross_validation',
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
        'Precision_Recall_AUC': round(prc_auc0,2),
        'accuracy': mean(scores['test_accuracy'])
    }
    
    if tuner_params:
        cross_validation['tuning_params']=grid.best_params_
        
    current = pd.DataFrame(cross_validation, index=[0])
    display(current)
    
    summary.append(cross_validation)
    
    return summary
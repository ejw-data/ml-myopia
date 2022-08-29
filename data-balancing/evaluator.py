# libraries
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
# Pipeline from imblearn - I could probably use scikitlearn instead
from imblearn.pipeline import Pipeline

from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc

from scikitplot.metrics import plot_roc, plot_precision_recall, plot_cumulative_gain, plot_lift_curve


# steps = [('over', balancer), ('model', model)]
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
def model_evaluate(steps, cv, Xtrain, ytrain, Xtest, ytest):
    
    # Pipeline
    pipeline = Pipeline(steps=steps)
    # Cross Validation Report
    cv_report(cv, pipeline, Xtrain, ytrain)
    # Cross Validation Folds
    cv_splits(cv, Xtrain, ytrain)
    # Holdout Testing
    holdout_report(pipeline, Xtrain, ytrain, Xtest, ytest)
    # Holdout Plot metrics 
    thresholds, tpr0, fpr0 = holdout_plots(pipeline, Xtest, ytest)  
    # Display thresholds
    holdout_thresholds(thresholds, tpr0, fpr0)



def cv_report(cv, pipeline, Xtrain, ytrain):
    
    scoring = ('f1', 'recall', 'precision', 'roc_auc')
    scores = cross_validate(pipeline, Xtrain, ytrain, scoring=scoring, cv=cv, n_jobs=-1)
    print(f"-----"*10)
    print("\n")
    print(f"Cross Validation Prediction Scores - average of 30 runs")
    print('Mean f1: %.3f' % mean(scores['test_f1']))
    print('Mean recall: %.3f' % mean(scores['test_recall']))
    print('Mean precision: %.3f' % mean(scores['test_precision']))
    print('Mean ROC AUC: %.3f' % mean(scores['test_roc_auc']))
    print("\n")


def cv_splits(cv, Xtrain, ytrain):

    # print('Resample dataset shape', Counter(y_smote))
    print(f"-----"*10)
    print("\n")
    print(f"Cross Validation Datasets")
    X_train_np = Xtrain.to_numpy()
    y_train_np = ytrain.to_numpy()
    for train_index, test_index in cv.split(X_train_np, y_train_np):
        # select rows
        # train_X, test_X = X_train_np[train_index], X_train_np[test_index]
        train_y, test_y = y_train_np[train_index], y_train_np[test_index]
        # summarize train and test composition
        train_0, train_1 = len(train_y[train_y==0]), len(train_y[train_y==1])
        test_0, test_1 = len(test_y[test_y==0]), len(test_y[test_y==1])
        print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
    print("\n")

def holdout_report(pipeline, Xtrain, ytrain, Xtest, ytest):
    pipeline.fit(Xtrain, ytrain)
    y_pred = pipeline.predict(Xtest)
    print(f"-----"*10)
    print("\n")
    print(f"Holdout Prediction Scores - single sample never used in training")
    print(classification_report(ytest, y_pred))
    print("\n")

def holdout_plots(pipeline, Xtest, ytest):
    print(f"-----"*10)
    print("\n")
    y_predicted_probability = pipeline.predict_proba(Xtest)

    # ROC
    fpr0, tpr0, thresholds = roc_curve(ytest, y_predicted_probability[:, 1])
    roc_auc0 = auc(fpr0, tpr0)
    print(f"ROC AUC:  {round(roc_auc0,2)}\n")
    plot_roc(ytest.to_list(), y_predicted_probability, figsize=(10, 4), classes_to_plot=[1], plot_micro=0)
    plt.show()
    print("\n")

    #Precision-Recall
    precision, recall, thresholds = precision_recall_curve(ytest.to_list(), y_predicted_probability[:, 1])
    prc_auc0 = auc(recall, precision)
    print(f"Precision-Recall Curve AUC:  {round(prc_auc0,2)}\n")
    plot_precision_recall(ytest.to_list(), y_predicted_probability, figsize=(10, 4))
    plt.show()
    print("\n")

    plot_cumulative_gain(ytest.to_list(), y_predicted_probability, figsize=(10, 4))
    plt.show()
    print("\n")

    plot_lift_curve(ytest.to_list(), y_predicted_probability, figsize=(10, 4))
    plt.show() 
    print("\n")

    return thresholds, tpr0, fpr0

def holdout_thresholds(thresholds, tpr0, fpr0):
    print(f"-----"*10)
    print("\n")
    print(f"Print Thresholds\n")
    print(f"{'Thresholds':>10} {'TPR':>10} {'FPR':>10}")
    threshold_list = list(zip(np.round_(thresholds,3), np.round_(tpr0,2), np.round_(fpr0,2)))
    [print(f"{row[0]:>10} {row[1]:>10} {row[2]:>10}") for row in threshold_list]
    print("\n")
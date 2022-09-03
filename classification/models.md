# Model Selection

### Notes from model testing experiments
*  model-KNN.ipynb
    Unbalanced Data:
    *  k=5
    *  ROC AUC: 0.68 
    *  Precision: 0.27
    *  Recall: 0.11
    Summary:  The cross validation shows that only a small percent of the predictions are correct.    

    Balanced Data with SMOTE:
    *  k = 5
    *  ROC AUC:  0.69
    *  Precision:  0.25
    *  Recall:  0.56
    Summary:  This model has some predictive power.  It is better at avoiding classification of false negatives. It's still not very good.    

*  model-SVC-gridsearch.ipynb  
    Unbalanced Data:
    *  Summary:  Training and testing had similar responses for the holdout data but the cross validation scores showed that all the scores were low.  

    Unbalanced Data with GridSearch
    Summary:  I laughted at this result since the optimized value for C was 1 which is the default to it will evaluate to the same values as the non-tuned model.    

    Unbalanced Data with PCA and GridSearch 
    Summary:  PCA worked best with 4 components.  There was a jump in precision using PCA with unbalanced data.  

    Balanced Data with SMOTE
    Summary:  Saw a significant increase in recall and modest gains in precision relative to the unbalanced experiment.   

    Balanced Data with SMOTE and PCA and Gridsearch
    Summary:  Recall increased but precision did not change much.  

    Balanced Data with SMOTE and SVC class weighting and Gridsearch
    Summary:  The model has a lot of differences between train and test datasets.  This experiment should be repeated with more data.  When running a stratified cross validation I received a lower precision and slightly lower recall scores as compared to not using weighting.  

    Balanced Data with SMOTE, PCA, SVC class weighting and Gridsearch
    Summary:  I think this model is overfitting.  I have high train scores but low test scores and the numbers just don't make sense.  I limited the values that the hyperparameters would test with to see if the overfitting could be stopped.  I was able to get some values that made fore sense but only by using high C values.  When running a separate cross validation, I was able to get precision and recall values a bit higher than past models.  I'm not sure about this model since I manually forced the C regularization term to be very high.  The class weighting basically did not help in any way since this model is nearly identical to the version without weighting and in some instances I think the weighting is contributing to the overfitting.  

    Final Summary:  I think that linear SVC just has a hard time fitting this model well.  The SVC parameter that allows for balancing may be fine to use when that is the only imbalance correcting done but otherwise it seems to contribute more to overfitting.  In general PCA helps a little bit but the biggest contributor to improving the model with this dataset is using oversampling.   

*  model-multimodel-compare.ipynb  
    Model Report indicated these are the best models:  
    *  NearestCentroid
    *  DesicisonTreeClassifier
    *  AdaBoostClassifier
    *  Perceptron
    *  LogisticRegression
    All show 60%+ Accuracy when balanced
    and show 77% Accuracy with unbalanced.  F1 Scores are typically over 80%.  
    My main preference is that it would include the Precision and Recall scores since I like to look at those individually instead of looking at the weighed combination of the two with the f1 score.  The ROC AUC can also be a bit misleading for models that have similar scores since they can have dramatically different precision and recall scores.  I am thinking about including a precision-recall metric to all my evaluations since that may be more helpful with imbalanced data.  

*  model-keras.ipynb 
    Sequential Model with 2 hidden layers with each one having 16 nodes.  
    *  Precision:  0.57
    *  Recall:  0.33